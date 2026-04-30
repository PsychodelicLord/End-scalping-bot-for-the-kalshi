"""
Kalshi High-Probability Trading Bot
====================================
Connects via WebSocket, monitors all markets in real-time,
and bets on markets priced at or above a configurable probability
threshold using Kelly Criterion position sizing.

Setup:
  pip install websockets cryptography httpx

Usage:
  1. Set your KEY_ID, PRIVATE_KEY_PATH, BANKROLL, and thresholds below
  2. Run: python kalshi_bot.py
  3. Use --dry-run to simulate without placing real orders
"""

import asyncio
import base64
import json
import os
import time
import logging
import argparse
import httpx
from datetime import datetime, timezone
from dataclasses import dataclass, field
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding

# ─────────────────────────────────────────────
#  CONFIG  —  all read from environment variables
#  Set these in Railway → Variables tab
# ─────────────────────────────────────────────
KEY_ID           = os.environ["KALSHI_KEY_ID"]
# Private key can be stored as a multiline env var (the full PEM string)
_raw_key         = os.environ["KALSHI_PRIVATE_KEY"].replace("\\n", "\n").encode()
PRIVATE_KEY      = serialization.load_pem_private_key(_raw_key, password=None)

BANKROLL         = float(os.getenv("BANKROLL",         "500.00"))
MIN_PROBABILITY  = float(os.getenv("MIN_PROBABILITY",  "0.90"))
MAX_KELLY_FRAC   = float(os.getenv("MAX_KELLY_FRAC",   "0.05"))
MIN_LIQUIDITY    = float(os.getenv("MIN_LIQUIDITY",    "10.0"))
MAX_OPEN_MARKETS = int(os.getenv("MAX_OPEN_MARKETS",   "10"))
DRY_RUN          = os.getenv("DRY_RUN", "false").lower() == "true"

# Kalshi endpoints
WS_URL   = "wss://api.elections.kalshi.com/trade-api/ws/v2"
REST_URL = "https://api.elections.kalshi.com/trade-api/v2"

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("kalshi_bot")


# ─────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────
@dataclass
class MarketSnapshot:
    ticker: str
    yes_bid: float  = 0.0
    yes_ask: float  = 0.0
    liquidity: float = 0.0  # estimated ask-side depth in USD

@dataclass
class Position:
    ticker: str
    contracts: int
    cost_per_contract: float   # USD paid per contract
    placed_at: str


@dataclass
class BotState:
    bankroll: float
    positions: dict = field(default_factory=dict)   # ticker -> Position
    market_data: dict = field(default_factory=dict) # ticker -> MarketSnapshot
    total_pnl: float = 0.0
    bets_placed: int = 0
    bets_won: int = 0
    bets_lost: int = 0


# ─────────────────────────────────────────────
#  AUTH HELPERS
# ─────────────────────────────────────────────
def sign(private_key, text: str) -> str:
    sig = private_key.sign(
        text.encode(),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode()


def make_headers(private_key, method: str, path: str) -> dict:
    ts = str(int(time.time() * 1000))
    msg = ts + method.upper() + path.split("?")[0]
    return {
        "Content-Type": "application/json",
        "KALSHI-ACCESS-KEY": KEY_ID,
        "KALSHI-ACCESS-SIGNATURE": sign(private_key, msg),
        "KALSHI-ACCESS-TIMESTAMP": ts,
    }


# ─────────────────────────────────────────────
#  KELLY CRITERION
# ─────────────────────────────────────────────
def kelly_bet(bankroll: float, prob: float, ask: float) -> float:
    """
    Full Kelly for a binary bet:
      b  = (1 - ask) / ask  (net odds on a $ask wager that pays $1)
      f* = (b*p - (1-p)) / b
    Returns dollar amount to wager, capped at MAX_KELLY_FRAC of bankroll.
    """
    if ask <= 0 or ask >= 1:
        return 0.0
    b = (1.0 - ask) / ask          # net profit per dollar risked
    f_star = (b * prob - (1 - prob)) / b
    f_star = max(0.0, f_star)      # never negative
    f_star = min(f_star, MAX_KELLY_FRAC)
    return round(bankroll * f_star, 2)


# ─────────────────────────────────────────────
#  REST: PLACE ORDER
# ─────────────────────────────────────────────
async def place_order(
    client: httpx.AsyncClient,
    private_key,
    ticker: str,
    contracts: int,
    limit_price: int,   # in cents, e.g. 93 cents → 93
    dry_run: bool = False,
) -> dict | None:
    if dry_run:
        log.info(f"[DRY RUN] Would buy {contracts} × {ticker} YES @ ${limit_price/100:.2f}")
        return {"order": {"order_id": "dry-run", "status": "resting"}}

    path = "/trade-api/v2/portfolio/orders"
    body = {
        "ticker": ticker,
        "action": "buy",
        "side": "yes",
        "type": "limit",
        "count": contracts,
        "yes_price": limit_price,
        "time_in_force": "immediate_or_cancel",
    }
    headers = make_headers(private_key, "POST", path)
    try:
        resp = await client.post(REST_URL + "/portfolio/orders",
                                 headers=headers, json=body, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log.error(f"Order failed for {ticker}: {e}")
        return None


# ─────────────────────────────────────────────
#  REST: GET BALANCE
# ─────────────────────────────────────────────
async def get_balance(client: httpx.AsyncClient, private_key) -> float:
    path = "/trade-api/v2/portfolio/balance"
    headers = make_headers(private_key, "GET", path)
    try:
        resp = await client.get(REST_URL + "/portfolio/balance",
                                headers=headers, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return data.get("balance", 0) / 100.0   # cents → dollars
    except Exception as e:
        log.warning(f"Could not fetch balance: {e}")
        return 0.0


# ─────────────────────────────────────────────
#  DECISION ENGINE
# ─────────────────────────────────────────────
async def evaluate_market(
    snap: MarketSnapshot,
    state: BotState,
    client: httpx.AsyncClient,
    private_key,
    dry_run: bool,
):
    ticker = snap.ticker
    ask    = snap.yes_ask

    # Already holding this market
    if ticker in state.positions:
        return

    # Too many open positions
    if len(state.positions) >= MAX_OPEN_MARKETS:
        return

    # Below threshold
    if ask < MIN_PROBABILITY:
        return

    # Thin liquidity
    if snap.liquidity < MIN_LIQUIDITY:
        return

    # Kelly sizing
    wager = kelly_bet(state.bankroll, ask, ask)  # treat ask as our probability estimate
    if wager < 0.10:
        log.debug(f"Skipping {ticker}: Kelly wager ${wager:.2f} too small")
        return

    # Convert to integer contracts (each contract costs `ask` dollars)
    contracts = max(1, int(wager / ask))
    limit_price_cents = int(ask * 100)

    log.info(
        f"🎯  {ticker}  ask={ask:.2f}  wager=${wager:.2f}  "
        f"contracts={contracts}  kelly_frac={wager/state.bankroll:.2%}"
    )

    result = await place_order(client, private_key, ticker,
                               contracts, limit_price_cents, dry_run)
    if result:
        state.positions[ticker] = Position(
            ticker=ticker,
            contracts=contracts,
            cost_per_contract=ask,
            placed_at=datetime.now(timezone.utc).isoformat(),
        )
        state.bankroll -= wager
        state.bets_placed += 1
        print_dashboard(state)


# ─────────────────────────────────────────────
#  DASHBOARD
# ─────────────────────────────────────────────
def print_dashboard(state: BotState):
    print("\n" + "─" * 60)
    print(f"  💰  Bankroll:    ${state.bankroll:.2f}")
    print(f"  📊  Bets placed: {state.bets_placed}  "
          f"(W:{state.bets_won} / L:{state.bets_lost})")
    print(f"  📈  Total P&L:   ${state.total_pnl:+.2f}")
    print(f"  🔓  Open positions ({len(state.positions)}):")
    for pos in state.positions.values():
        cost = pos.contracts * pos.cost_per_contract
        print(f"       {pos.ticker:35s}  {pos.contracts}× @ ${pos.cost_per_contract:.2f}  "
              f"cost=${cost:.2f}")
    print("─" * 60 + "\n")


# ─────────────────────────────────────────────
#  WEBSOCKET LISTENER
# ─────────────────────────────────────────────
async def run_bot():
    try:
        import websockets
    except ImportError:
        log.error("Install websockets:  pip install websockets cryptography httpx")
        return

    private_key = PRIVATE_KEY
    dry_run = DRY_RUN
    state = BotState(bankroll=BANKROLL)
    msg_id = 1

    log.info(f"Starting Kalshi bot  |  bankroll=${BANKROLL}  "
             f"threshold={MIN_PROBABILITY:.0%}  dry_run={dry_run}")

    async with httpx.AsyncClient() as http_client:
        # Sync starting bankroll from API (skip in dry-run)
        if not dry_run:
            live_bal = await get_balance(http_client, private_key)
            if live_bal > 0:
                state.bankroll = live_bal
                log.info(f"Live balance: ${state.bankroll:.2f}")

        reconnect_delay = 1.0

        while True:
            try:
                ws_headers = make_headers(private_key, "GET", "/trade-api/ws/v2")
                async with websockets.connect(
                    WS_URL,
                    additional_headers=ws_headers,
                    ping_interval=20,
                    ping_timeout=30,
                ) as ws:
                    log.info("✅  WebSocket connected")
                    reconnect_delay = 1.0

                    # Subscribe to the global ticker feed (all markets)
                    sub = {"id": msg_id, "cmd": "subscribe",
                           "params": {"channels": ["ticker"]}}
                    await ws.send(json.dumps(sub))
                    msg_id += 1

                    # Also subscribe to fills so we can track settlement
                    sub_fills = {"id": msg_id, "cmd": "subscribe",
                                 "params": {"channels": ["fill"]}}
                    await ws.send(json.dumps(sub_fills))
                    msg_id += 1

                    async for raw in ws:
                        data = json.loads(raw)
                        mtype = data.get("type")

                        # ── ticker update ────────────────────────────
                        if mtype == "ticker":
                            msg = data.get("msg", {})
                            log.info(f"TICK {ticker} ask={yes_ask} bid={yes_bid}")
                            ticker = msg.get("market_ticker", "")
                            yes_ask = float(msg.get("yes_ask_dollars") or 0)
                            yes_bid = float(msg.get("yes_bid_dollars") or 0)
                            liquidity = float(msg.get("liquidity") or 0)

                            snap = MarketSnapshot(
                                ticker=ticker,
                                yes_ask=yes_ask,
                                yes_bid=yes_bid,
                                liquidity=liquidity if liquidity else 999.0,
                            )
                            state.market_data[ticker] = snap

                            await evaluate_market(
                                snap, state, http_client, private_key, dry_run
                            )

                        # ── fill (settlement) ────────────────────────
                        elif mtype == "fill":
                            msg = data.get("msg", {})
                            ticker = msg.get("market_ticker", "")
                            count = int(msg.get("count", 0))
                            yes_price = float(msg.get("yes_price", 0)) / 100.0

                            if ticker in state.positions:
                                pos = state.positions[ticker]
                                proceeds = count * yes_price
                                cost = count * pos.cost_per_contract
                                pnl = proceeds - cost
                                state.total_pnl += pnl
                                state.bankroll += proceeds

                                if pnl > 0:
                                    state.bets_won += 1
                                    log.info(f"✅  WON  {ticker}  pnl=${pnl:+.2f}")
                                else:
                                    state.bets_lost += 1
                                    log.warning(f"❌  LOST {ticker}  pnl=${pnl:+.2f}")

                                del state.positions[ticker]
                                print_dashboard(state)

                        # ── errors ───────────────────────────────────
                        elif mtype == "error":
                            log.warning(f"WS error: {data}")

            except Exception as e:
                log.error(f"WebSocket error: {e}  — reconnecting in {reconnect_delay}s")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 60)


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        log.info("Bot stopped.")

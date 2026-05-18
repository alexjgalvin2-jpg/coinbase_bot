"""
Coinbase Trading Bot — Coinbase Advanced Trade API
Strategy: Momentum-based spot trading on crypto pairs 24/7.
Buys when technical conditions are met, sells when target or stop is hit.
Uses Coinbase Advanced Trade REST API with JWT authentication.
"""

import os
import time
import json
import logging
import requests
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np
import jwt

# =============================================================================
#  CONFIG
# =============================================================================
CB_API_KEY    = os.getenv("COINBASE_API_KEY")
CB_API_SECRET = os.getenv("COINBASE_API_SECRET")

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

STATE_FILE     = "coinbase_state.json"
TRADE_LOG_FILE = "coinbase_trades.json"

PAPER_MODE = True   # ← set False only when ready for real money

# Trading parameters
CAPITAL_PER_TRADE  = 200.0    # USD per trade
MIN_PROFIT_PCT     = 1.5      # minimum expected profit % to enter
STOP_LOSS_PCT      = 1.0      # stop loss % below entry
TAKE_PROFIT_PCT    = 2.0      # take profit % above entry
MAX_POSITIONS      = 5        # max open positions at once
POLL_INTERVAL      = 60       # seconds between scans

# Technical filter thresholds
RSI_MIN            = 52       # RSI must be above this to buy
RSI_PERIOD         = 14
SMA_PERIOD         = 20       # price must be above 20-period SMA
VOLUME_MULT        = 1.3      # volume must be 1.3x average

# Symbols to trade (Coinbase Advanced format)
SYMBOLS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD",
    "LINK-USD", "MATIC-USD", "DOT-USD", "ADA-USD",
    "DOGE-USD", "UNI-USD", "ATOM-USD", "LTC-USD",
]

BASE_URL = "https://api.coinbase.com"

# =============================================================================
#  LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("coinbase_bot.log", encoding="utf-8")
    ]
)
log = logging.getLogger("CoinbaseBot")


# =============================================================================
#  TELEGRAM
# =============================================================================
def send_telegram(msg: str):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=10)
    except Exception as e:
        log.warning("Telegram failed: %s", e)


# =============================================================================
#  TRADE LOGGER
# =============================================================================
def log_trade(symbol: str, side: str, price: float, qty: float,
              pnl: Optional[float] = None, reason: str = ""):
    try:
        try:
            with open(TRADE_LOG_FILE) as f:
                trades = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            trades = []
        trades.append({
            "symbol":    symbol,
            "side":      side,
            "price":     price,
            "qty":       qty,
            "pnl":       round(pnl, 2) if pnl is not None else None,
            "reason":    reason,
            "timestamp": datetime.now().isoformat(),
        })
        with open(TRADE_LOG_FILE, "w") as f:
            json.dump(trades, f, indent=2)
    except Exception as e:
        log.warning("log_trade failed: %s", e)


# =============================================================================
#  COINBASE ADVANCED TRADE CLIENT
# =============================================================================
class CoinbaseClient:
    def __init__(self):
        self.api_key    = CB_API_KEY
        self.api_secret = CB_API_SECRET
        log.info("Coinbase client initialised")

    def _get_jwt(self) -> str:
        """Generate a JWT token for Coinbase Advanced Trade API authentication."""
        import time as t
        payload = {
            "sub":  self.api_key,
            "iss":  "coinbase-cloud",
            "nbf":  int(t.time()),
            "exp":  int(t.time()) + 120,
            "aud":  ["retail_rest_api_proxy"],
        }
        private_key = self.api_secret.replace("\\n", "\n")
        token = jwt.encode(payload, private_key, algorithm="ES256",
                           headers={"kid": self.api_key, "nonce": str(int(t.time()))})
        return token

    def _request(self, method: str, path: str,
                 params: dict = None, body: dict = None) -> Optional[dict]:
        try:
            token   = self._get_jwt()
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type":  "application/json",
            }
            url = BASE_URL + path
            if method == "GET":
                resp = requests.get(url, headers=headers, params=params, timeout=10)
            else:
                resp = requests.post(url, headers=headers, json=body, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            log.warning("API %s %s failed: %s", method, path, e)
            return None

    def get_accounts(self) -> list:
        """Get all Coinbase accounts."""
        data = self._request("GET", "/api/v3/brokerage/accounts")
        if data:
            return data.get("accounts", [])
        return []

    def get_usd_balance(self) -> float:
        """Get available USD balance."""
        accounts = self.get_accounts()
        for acct in accounts:
            if acct.get("currency") == "USD":
                return float(acct.get("available_balance", {}).get("value", 0))
        return 0.0

    def get_candles(self, symbol: str, granularity: str = "ONE_HOUR",
                    limit: int = 50) -> Optional[pd.DataFrame]:
        """Fetch OHLCV candles for a symbol."""
        try:
            end   = int(datetime.utcnow().timestamp())
            # Map granularity to seconds
            gran_map = {
                "ONE_MINUTE": 60, "FIVE_MINUTE": 300,
                "FIFTEEN_MINUTE": 900, "ONE_HOUR": 3600,
                "SIX_HOUR": 21600, "ONE_DAY": 86400,
            }
            secs  = gran_map.get(granularity, 3600)
            start = end - (limit * secs)

            data = self._request("GET", f"/api/v3/brokerage/products/{symbol}/candles", params={
                "start":       str(start),
                "end":         str(end),
                "granularity": granularity,
                "limit":       limit,
            })
            if not data or "candles" not in data:
                return None

            candles = data["candles"]
            if not candles:
                return None

            df = pd.DataFrame(candles)
            df = df.rename(columns={
                "start": "timestamp", "open": "open", "high": "high",
                "low": "low", "close": "close", "volume": "volume"
            })
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)
            return df
        except Exception as e:
            log.warning("get_candles(%s) failed: %s", symbol, e)
            return None

    def get_best_bid_ask(self, symbol: str) -> Optional[dict]:
        """Get current best bid and ask price."""
        data = self._request("GET", f"/api/v3/brokerage/best_bid_ask",
                             params={"product_ids": symbol})
        if data and "pricebooks" in data:
            for pb in data["pricebooks"]:
                if pb.get("product_id") == symbol:
                    bids = pb.get("bids", [])
                    asks = pb.get("asks", [])
                    if bids and asks:
                        return {
                            "bid": float(bids[0]["price"]),
                            "ask": float(asks[0]["price"]),
                        }
        return None

    def place_order(self, symbol: str, side: str, quote_size: float = None,
                    base_size: float = None) -> Optional[dict]:
        """Place a market order. Use quote_size for buys (USD amount), base_size for sells."""
        import uuid
        body = {
            "client_order_id": str(uuid.uuid4()),
            "product_id":      symbol,
            "side":            side.upper(),
            "order_configuration": {
                "market_market_ioc": {}
            }
        }
        if side.upper() == "BUY" and quote_size:
            body["order_configuration"]["market_market_ioc"]["quote_size"] = str(round(quote_size, 2))
        elif side.upper() == "SELL" and base_size:
            body["order_configuration"]["market_market_ioc"]["base_size"] = str(base_size)

        if PAPER_MODE:
            log.info("📄 PAPER ORDER skipped (real API call blocked): %s %s", side.upper(), symbol)
            return {"order_id": "paper-" + body["client_order_id"], "paper": True}

        data = self._request("POST", "/api/v3/brokerage/orders", body=body)
        if data and data.get("success"):
            return data.get("success_response", {})
        log.warning("Order failed: %s", data)
        return None

    def get_order(self, order_id: str) -> Optional[dict]:
        """Get order details by ID."""
        return self._request("GET", f"/api/v3/brokerage/orders/historical/{order_id}")

    def list_open_positions(self) -> dict:
        """Return current crypto holdings as {symbol: {qty, avg_price}}."""
        accounts = self.get_accounts()
        positions = {}
        for acct in accounts:
            currency = acct.get("currency", "")
            if currency == "USD":
                continue
            qty = float(acct.get("available_balance", {}).get("value", 0))
            if qty > 0:
                symbol = f"{currency}-USD"
                if symbol in SYMBOLS:
                    positions[symbol] = {"qty": qty, "currency": currency}
        return positions


# =============================================================================
#  INDICATOR HELPERS
# =============================================================================
def calc_rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def calc_sma(close: pd.Series, period: int) -> float:
    return close.rolling(period).mean().iloc[-1]

def calc_macd(close: pd.Series):
    fast = close.ewm(span=12, adjust=False).mean()
    slow = close.ewm(span=26, adjust=False).mean()
    line = fast - slow
    sig  = line.ewm(span=9, adjust=False).mean()
    hist = line - sig
    return line.iloc[-1], sig.iloc[-1], hist.iloc[-1]


# =============================================================================
#  SIGNAL ENGINE
# =============================================================================
def analyse(client: CoinbaseClient, symbol: str) -> dict:
    result = {
        "symbol":      symbol,
        "signal":      None,
        "reason":      [],
        "entry_price": None,
        "stop_loss":   None,
        "take_profit": None,
    }

    # Get hourly candles
    df = client.get_candles(symbol, granularity="ONE_HOUR", limit=50)
    if df is None or len(df) < SMA_PERIOD + 5:
        result["reason"].append("insufficient candle data")
        return result

    close  = df["close"]
    volume = df["volume"]

    last_close   = close.iloc[-1]
    sma_val      = calc_sma(close, SMA_PERIOD)
    rsi_val      = calc_rsi(close, RSI_PERIOD)
    avg_vol      = volume.rolling(20).mean().iloc[-1]
    last_vol     = volume.iloc[-1]
    macd_l, macd_s, macd_h = calc_macd(close)

    above_sma    = last_close > sma_val
    rsi_ok       = rsi_val > RSI_MIN
    volume_ok    = last_vol >= avg_vol * VOLUME_MULT
    macd_ok      = macd_l > macd_s and macd_h > 0

    # Get live price
    quote = client.get_best_bid_ask(symbol)
    if quote:
        entry = quote["ask"]
    else:
        entry = last_close

    stop_loss   = entry * (1 - STOP_LOSS_PCT / 100)
    take_profit = entry * (1 + TAKE_PROFIT_PCT / 100)

    if above_sma and rsi_ok and volume_ok and macd_ok:
        result["signal"]      = "buy"
        result["entry_price"] = entry
        result["stop_loss"]   = stop_loss
        result["take_profit"] = take_profit
        result["reason"]      = [
            f"price {entry:.4f} > SMA{SMA_PERIOD} {sma_val:.4f}",
            f"RSI {rsi_val:.1f} > {RSI_MIN}",
            f"volume spike {last_vol/avg_vol:.1f}x",
            f"MACD bullish (hist {macd_h:.4f})",
        ]
        return result

    # Exit signal — below SMA
    if not above_sma:
        result["signal"] = "exit"
        result["reason"] = [f"price {last_close:.4f} below SMA{SMA_PERIOD} {sma_val:.4f}"]
        return result

    flags = []
    if not above_sma:  flags.append(f"below SMA{SMA_PERIOD}")
    if not rsi_ok:     flags.append(f"RSI {rsi_val:.1f} < {RSI_MIN}")
    if not volume_ok:  flags.append(f"weak volume ({last_vol/avg_vol:.1f}x)")
    if not macd_ok:    flags.append("MACD bearish")
    result["reason"] = flags
    return result


# =============================================================================
#  BOT
# =============================================================================
class CoinbaseBot:
    def __init__(self):
        self.client     = CoinbaseClient()
        self.positions  = self._load_state()

    def _load_state(self) -> dict:
        try:
            with open(STATE_FILE) as f:
                state = json.load(f)
            log.info("Loaded state: %d open position(s)", len(state))
            return state
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_state(self):
        with open(STATE_FILE, "w") as f:
            json.dump(self.positions, f, indent=2)

    def _check_stops(self, live_positions: dict):
        """Check stop-loss and take-profit for all open positions."""
        for symbol, info in list(self.positions.items()):
            quote = self.client.get_best_bid_ask(symbol)
            if not quote:
                continue

            current_price = quote["bid"]
            entry         = info["entry"]
            stop          = info["stop_loss"]
            target        = info["take_profit"]
            qty           = info["qty"]

            if current_price <= stop:
                log.info("STOP-LOSS hit %s @ %.4f (entry=%.4f)", symbol, current_price, entry)
                order = self.client.place_order(symbol, "sell", base_size=qty)
                if order:
                    pnl = (current_price - entry) * qty
                    send_telegram(
                        f"🛑 STOP-LOSS hit [{symbol}]\n"
                        f"Entry: ${entry:.4f} → Exit: ${current_price:.4f}\n"
                        f"P&L: ${pnl:+.2f}"
                    )
                    log_trade(symbol, "sell", current_price, qty, pnl, "stop-loss")
                    del self.positions[symbol]
                    self._save_state()

            elif current_price >= target:
                log.info("TAKE-PROFIT hit %s @ %.4f (entry=%.4f)", symbol, current_price, entry)
                order = self.client.place_order(symbol, "sell", base_size=qty)
                if order:
                    pnl = (current_price - entry) * qty
                    send_telegram(
                        f"✅ TAKE-PROFIT hit [{symbol}]\n"
                        f"Entry: ${entry:.4f} → Exit: ${current_price:.4f}\n"
                        f"P&L: ${pnl:+.2f}"
                    )
                    log_trade(symbol, "sell", current_price, qty, pnl, "take-profit")
                    del self.positions[symbol]
                    self._save_state()

    def run(self):
        try:
            log.info("=== Coinbase Bot STARTED ===")

            # Verify connection
            balance = self.client.get_usd_balance()
            log.info("USD Balance: $%.2f", balance)

            send_telegram(
                f"🟡 Coinbase Bot is online!\n"
                f"Mode: {'PAPER 🧪' if PAPER_MODE else 'LIVE 💰'}\n"
                f"USD Balance: ${balance:,.2f}\n"
                f"Capital per trade: ${CAPITAL_PER_TRADE}\n"
                f"Scanning {len(SYMBOLS)} crypto pairs\n"
                f"Stop loss: {STOP_LOSS_PCT}% | Take profit: {TAKE_PROFIT_PCT}%"
            )

            while True:
                try:
                    live_positions = self.client.list_open_positions()
                    log.info("-- Scan -- balance=$%.2f  open=%d/%d --",
                             self.client.get_usd_balance(), len(self.positions), MAX_POSITIONS)

                    # Check stops/targets for open positions
                    self._check_stops(live_positions)

                    # Scan for new opportunities
                    for symbol in SYMBOLS:
                        analysis = analyse(self.client, symbol)
                        sig      = analysis["signal"]

                        # Exit signal for held position
                        if symbol in self.positions and sig == "exit":
                            info  = self.positions[symbol]
                            quote = self.client.get_best_bid_ask(symbol)
                            price = quote["bid"] if quote else info["entry"]
                            qty   = info["qty"]
                            order = self.client.place_order(symbol, "sell", base_size=qty)
                            if order:
                                pnl = (price - info["entry"]) * qty
                                send_telegram(
                                    f"🔴 SELL [{symbol}]\n"
                                    f"Reason: {'; '.join(analysis['reason'])}\n"
                                    f"P&L: ${pnl:+.2f}"
                                )
                                log_trade(symbol, "sell", price, qty, pnl,
                                          "; ".join(analysis["reason"]))
                                del self.positions[symbol]
                                self._save_state()
                            continue

                        # Skip if already holding or at max positions
                        if symbol in self.positions:
                            continue
                        if len(self.positions) >= MAX_POSITIONS:
                            log.info("Max positions reached, skipping %s", symbol)
                            continue

                        if sig != "buy":
                            if analysis["reason"]:
                                log.debug("SKIP %s — %s", symbol,
                                          "; ".join(analysis["reason"]))
                            continue

                        # Check we have enough USD
                        usd_balance = self.client.get_usd_balance()
                        if usd_balance < CAPITAL_PER_TRADE:
                            log.info("Insufficient USD balance ($%.2f), skipping %s",
                                     usd_balance, symbol)
                            continue

                        entry = analysis["entry_price"]
                        qty   = round(CAPITAL_PER_TRADE / entry, 8)

                        log.info("BUY %s qty=%.6f entry=%.4f SL=%.4f TP=%.4f | %s",
                                 symbol, qty, entry,
                                 analysis["stop_loss"], analysis["take_profit"],
                                 "; ".join(analysis["reason"]))

                        order = self.client.place_order(
                            symbol, "buy", quote_size=CAPITAL_PER_TRADE
                        )
                        if order:
                            send_telegram(
                                f"🟢 BUY [{symbol}]\n"
                                f"Entry: ${entry:.4f}\n"
                                f"Amount: ${CAPITAL_PER_TRADE}\n"
                                f"Stop-loss: ${analysis['stop_loss']:.4f}\n"
                                f"Take-profit: ${analysis['take_profit']:.4f}\n"
                                f"Reason: {'; '.join(analysis['reason'])}"
                            )
                            log_trade(symbol, "buy", entry, qty, reason="; ".join(analysis["reason"]))
                            self.positions[symbol] = {
                                "entry":       entry,
                                "qty":         qty,
                                "stop_loss":   analysis["stop_loss"],
                                "take_profit": analysis["take_profit"],
                            }
                            self._save_state()

                except Exception as e:
                    log.error("Scan error: %s", e, exc_info=True)
                    send_telegram(f"⚠️ Coinbase Bot scan error: {e}")

                time.sleep(POLL_INTERVAL)

        except Exception as e:
            log.error("FATAL: %s", e, exc_info=True)
            send_telegram(f"💀 Coinbase Bot crashed: {e}")
            raise


# =============================================================================
#  ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    bot = CoinbaseBot()
    bot.run()

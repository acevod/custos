"""
Data puller for Bitget Reality Spot Stock API - 10 rTokens covering a
mix of volatility profiles. Public endpoint, no API key required.
https://www.bitget.com/docs/catalog/market/market-data#get-tickers

M3/M6 fixes vs the original version:
  - Reuses a single requests.Session across all 10 symbols instead of
    opening a fresh TLS connection per request.
  - Retries each request with exponential backoff on 429 / 5xx /
    connection errors, instead of giving up after one attempt. A
    transient rate-limit no longer silently blanks a token for a whole
    4-hour cycle.
"""

import time

import requests
from datetime import datetime, timezone

BASE_URL = "https://api.bitget.com/api/v3/market"

MAX_RETRIES = 3          # total attempts per symbol
BACKOFF_BASE_SECONDS = 2 # 2s, 4s between retries

# M6 fix: one shared session (connection pooling) for all symbols.
SESSION = requests.Session()

SYMBOLS = {
    "NVDA": "rNVDAUSDT",
    "TSLA": "rTSLAUSDT",
    "AAPL": "rAAPLUSDT",
    "AMZN": "rAMZNUSDT",
    "GOOGL": "rGOOGLUSDT",
    "SPY": "rSPYUSDT",
    "QQQ": "rQQQUSDT",
    "KO": "rKOUSDT",
    "MCD": "rMCDUSDT",
    "PYPL": "rPYPLUSDT",
}


def fetch_ticker(symbol: str) -> dict:
    """Fetch full ticker: lastPrice, bid1/ask1 price+size, volume24h.
    Retries up to MAX_RETRIES times with linear backoff on 429, 5xx
    and connection-level errors; re-raises the last error when the
    retries are exhausted (the caller converts it to a per-symbol
    error entry, exactly as before)."""
    url = f"{BASE_URL}/tickers"
    params = {"category": "SPOT", "symbol": symbol}
    last_err = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = SESSION.get(url, params=params, timeout=10)
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < MAX_RETRIES:
                    time.sleep(BACKOFF_BASE_SECONDS * attempt)
                    continue
            resp.raise_for_status()
            rows = resp.json().get("data", [])
            return rows[0] if rows else {}
        except requests.exceptions.RequestException as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(BACKOFF_BASE_SECONDS * attempt)
                continue
            raise
    raise last_err if last_err else RuntimeError("unreachable")


def calculate_spread_pct(bid: float | None, ask: float | None) -> float | None:
    """Spread from bid/ask, as a percentage of mid price."""
    if bid is None or ask is None:
        return None
    mid = (bid + ask) / 2
    if mid == 0:
        return None
    return round((ask - bid) / mid * 100, 4)


def fetch_bitget_data() -> list[dict]:
    """
    Main entry point - called from the central data puller.
    Returns a list of dicts, one entry per rToken.
    """
    results = []
    for underlying, symbol in SYMBOLS.items():
        try:
            ticker = fetch_ticker(symbol)
            if not ticker:
                raise ValueError("empty ticker response - symbol may be wrong or not listed")

            bid = float(ticker.get("bid1Price", 0)) or None
            ask = float(ticker.get("ask1Price", 0)) or None
            bid_size = float(ticker.get("bid1Size", 0)) or None
            ask_size = float(ticker.get("ask1Size", 0)) or None

            results.append({
                "underlying": underlying,
                "symbol": symbol,
                "price": float(ticker.get("lastPrice", 0)) or None,
                "bid": bid,
                "ask": ask,
                "bid_size": bid_size,
                "ask_size": ask_size,
                "volume_24h": float(ticker.get("volume24h", 0)) or None,
                "spread_pct": calculate_spread_pct(bid, ask),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "bitget_api_v3",
                "status": "ok",
            })
        except Exception as e:
            results.append({
                "underlying": underlying,
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "bitget_api_v3",
                "status": "error",
                "error": str(e),
            })
    return results


if __name__ == "__main__":
    import json
    print(json.dumps(fetch_bitget_data(), indent=2))

"""
Data puller for Binance bStocks (NVDAB, TSLAB, AAPLB, AMZNB).
Issued by BTech Holdings Limited, a Binance affiliate, on BNB Chain.
Public endpoint: https://api.binance.com/api/v3/ticker/24hr
No API key required. This single endpoint gives price, bid/ask, and
volume together, unlike /ticker/bookTicker which omits volume.

Symbol verification status (tested live via browser):
  NVDABUSDT  - confirmed (includes volume via /ticker/24hr)
  TSLABUSDT  - confirmed at bStocks launch announcement
  AAPLBUSDT  - confirmed via Binance news (Apple added to lineup)
  AMZNBUSDT  - confirmed via Binance news (Amazon added to lineup)
  GOOGLBUSDT - confirmed (tested live)
"""

import requests
from datetime import datetime, timezone

BASE_URL = "https://api.binance.com/api/v3"

SYMBOLS = {
    "NVDA": "NVDABUSDT",
    "TSLA": "TSLABUSDT",
    "AAPL": "AAPLBUSDT",
    "AMZN": "AMZNBUSDT",
    "GOOGL": "GOOGLBUSDT",
}


def fetch_ticker_24hr(symbol: str) -> dict:
    """Fetch 24hr ticker: lastPrice, bidPrice/askPrice, volume, etc."""
    url = f"{BASE_URL}/ticker/24hr"
    params = {"symbol": symbol}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def calculate_spread_pct(bid: float | None, ask: float | None) -> float | None:
    """Spread from bid/ask, as a percentage of mid price."""
    if bid is None or ask is None:
        return None
    mid = (bid + ask) / 2
    if mid == 0:
        return None
    return round((ask - bid) / mid * 100, 4)


def fetch_binance_data() -> list[dict]:
    """
    Main entry point - called from the central data puller.
    Returns a list of dicts, one entry per underlying stock.
    """
    results = []
    for underlying, symbol in SYMBOLS.items():
        try:
            ticker = fetch_ticker_24hr(symbol)
            if not ticker or "lastPrice" not in ticker:
                raise ValueError("empty/invalid ticker response - symbol may be wrong or not listed")

            bid = float(ticker.get("bidPrice", 0)) or None
            ask = float(ticker.get("askPrice", 0)) or None

            results.append({
                "issuer": "binance",
                "underlying": underlying,
                "symbol": symbol,
                "price": float(ticker.get("lastPrice", 0)) or None,
                "bid": bid,
                "ask": ask,
                "volume_24h": float(ticker.get("volume", 0)) or None,
                "spread_pct": calculate_spread_pct(bid, ask),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "binance_api_v3",
                "status": "ok",
            })
        except Exception as e:
            results.append({
                "issuer": "binance",
                "underlying": underlying,
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "binance_api_v3",
                "status": "error",
                "error": str(e),
            })
    return results


if __name__ == "__main__":
    import json
    print(json.dumps(fetch_binance_data(), indent=2))

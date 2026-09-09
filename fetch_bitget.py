"""
Data puller for Bitget Reality Spot Stock API (rNVDA, rTSLA, rAAPL).
Public endpoint, verified against official docs:
https://www.bitget.com/docs/catalog/market/market-data#get-tickers
No API key required - the Ticker endpoint already includes bid1/ask1
(top of book), so we don't need the separate Order Book endpoint
(which is whitelist-only).
"""

import requests
from datetime import datetime, timezone

BASE_URL = "https://api.bitget.com/api/v3/market"

SYMBOLS = {
    "NVDA": "rNVDAUSDT",
    "TSLA": "rTSLAUSDT",
    "AAPL": "rAAPLUSDT",
}


def fetch_ticker(symbol: str) -> dict:
    """Fetch full ticker: lastPrice, bid1/ask1, volume24h, etc."""
    url = f"{BASE_URL}/tickers"
    params = {"category": "SPOT", "symbol": symbol}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    rows = data.get("data", [])
    return rows[0] if rows else {}


def calculate_spread_pct(ticker: dict) -> float | None:
    """Spread from bid1/ask1, already present directly in the ticker."""
    try:
        bid = float(ticker["bid1Price"])
        ask = float(ticker["ask1Price"])
        mid = (bid + ask) / 2
        if mid == 0:
            return None
        return round((ask - bid) / mid * 100, 4)
    except (KeyError, ValueError, TypeError, ZeroDivisionError):
        return None


def fetch_bitget_data() -> list[dict]:
    """
    Main entry point - called from the central data puller.
    Returns a list of dicts, one entry per underlying stock.
    """
    results = []
    for underlying, symbol in SYMBOLS.items():
        try:
            ticker = fetch_ticker(symbol)
            if not ticker:
                raise ValueError("empty ticker response - symbol may be wrong or not listed")

            spread = calculate_spread_pct(ticker)

            results.append({
                "issuer": "bitget",
                "underlying": underlying,
                "symbol": symbol,
                "price": float(ticker.get("lastPrice", 0)) or None,
                "bid": float(ticker.get("bid1Price", 0)) or None,
                "ask": float(ticker.get("ask1Price", 0)) or None,
                "volume_24h": float(ticker.get("volume24h", 0)) or None,
                "spread_pct": spread,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "bitget_api_v3",
                "status": "ok",
            })
        except Exception as e:
            # Don't let one symbol's failure crash the whole batch -
            # record the error and move on to the next symbol.
            results.append({
                "issuer": "bitget",
                "underlying": underlying,
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "bitget_api_v3",
                "status": "error",
                "error": str(e),
            })
    return results


if __name__ == "__main__":
    # Manual test: run this file directly to check the connection.
    import json
    print(json.dumps(fetch_bitget_data(), indent=2))

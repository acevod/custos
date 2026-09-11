"""
Data puller for Bitget Reality Spot Stock API - 10 rTokens covering a
mix of volatility profiles (mega-cap tech, ETFs, consumer staples,
fintech) so the abnormal-movement scoring has real contrast to work
with. Public endpoint, no API key required.
https://www.bitget.com/docs/catalog/market/market-data#get-tickers
"""

import requests
from datetime import datetime, timezone

BASE_URL = "https://api.bitget.com/api/v3/market"

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
    """Fetch full ticker: lastPrice, bid1/ask1 price+size, volume24h."""
    url = f"{BASE_URL}/tickers"
    params = {"category": "SPOT", "symbol": symbol}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    rows = data.get("data", [])
    return rows[0] if rows else {}


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

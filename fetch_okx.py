"""
Data puller for xStocks as listed on OKX (Unified Tokenized Stocks:
XNVDA, XTSLA, XAAPL, XAMZN, XGOOGL). Underlying token is issued by
Backed Finance (same xStocks framework used on Bybit), OKX just
provides a shared order book across issuers.

Public endpoint: https://us.okx.com/api/v5/market/ticker
No API key required. IMPORTANT: uses the us.okx.com domain
specifically, not openapi.okx.com - OKX's tokenized stocks product
explicitly excludes US/EU users on the main domain, but us.okx.com
is the dedicated endpoint for US-registered traffic and returned
valid data when tested (including from a US-hosted network, unlike
Bybit which blocks US/China IPs outright with no regional
alternative for this product).
"""

import requests
from datetime import datetime, timezone

BASE_URL = "https://us.okx.com/api/v5/market"

SYMBOLS = {
    "NVDA": "XNVDA-USDT",
    "TSLA": "XTSLA-USDT",
    "AAPL": "XAAPL-USDT",
    "AMZN": "XAMZN-USDT",
    "GOOGL": "XGOOGL-USDT",
}


def fetch_ticker(symbol: str) -> dict:
    """Fetch ticker: last, bidPx/askPx, vol24h, etc."""
    url = f"{BASE_URL}/ticker"
    params = {"instId": symbol}
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


def fetch_okx_data() -> list[dict]:
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

            bid = float(ticker.get("bidPx", 0)) or None
            ask = float(ticker.get("askPx", 0)) or None

            results.append({
                "issuer": "okx",
                "underlying": underlying,
                "symbol": symbol,
                "price": float(ticker.get("last", 0)) or None,
                "bid": bid,
                "ask": ask,
                "volume_24h": float(ticker.get("vol24h", 0)) or None,
                "spread_pct": calculate_spread_pct(bid, ask),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "okx_api_v5",
                "status": "ok",
            })
        except Exception as e:
            results.append({
                "issuer": "okx",
                "underlying": underlying,
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "okx_api_v5",
                "status": "error",
                "error": str(e),
            })
    return results


if __name__ == "__main__":
    import json
    print(json.dumps(fetch_okx_data(), indent=2))

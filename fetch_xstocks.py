"""
Data puller for xStocks (Backed Finance) public API.
Endpoint verified against official docs:
https://docs.xstocks.fi/apis/openapi/assets
No API key required for public endpoints.

IMPORTANT NOTE: the price-data endpoint only returns a single quote
number, no bid/ask. The mint/redeem spread (xChange RFQ) is an
authenticated endpoint that requires a Backed client account - not
used here. Instead, a "spread proxy" is computed downstream from
our own stored price history (short-term volatility).
"""

import requests
from datetime import datetime, timezone

BASE_URL = "https://api.xstocks.fi/api/v2"

# xStocks symbols use a lowercase "x" suffix
SYMBOLS = {
    "NVDA": "NVDAx",
    "TSLA": "TSLAx",
    "AAPL": "AAPLx",
}


def fetch_price(symbol: str) -> dict:
    """Fetch the current indicative price. Response: {"quote": number}."""
    url = f"{BASE_URL}/public/assets/{symbol}/price-data"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


def fetch_asset_info(symbol: str) -> dict:
    """
    Optional: extra info (trading status, etc.) from the
    'Get Asset by Symbol' endpoint. Useful for checking isTradingHalted.
    """
    url = f"{BASE_URL}/public/assets/{symbol}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


def fetch_xstocks_data() -> list[dict]:
    """
    Main entry point - called from the central data puller.
    Returns a list of dicts, one entry per underlying stock.
    spread_pct is intentionally None here - it gets computed
    separately in health_score.py from price history (volatility proxy).
    """
    results = []
    for underlying, symbol in SYMBOLS.items():
        try:
            price_data = fetch_price(symbol)
            quote = price_data.get("quote")
            if quote is None:
                raise ValueError("quote is null - trading is likely halted")

            # Extra info - a failure here must not fail the whole entry.
            trading_halted = None
            try:
                info = fetch_asset_info(symbol)
                trading_halted = info.get("isTradingHalted")
            except Exception:
                pass

            results.append({
                "issuer": "xstocks",
                "underlying": underlying,
                "symbol": symbol,
                "price": float(quote),
                "spread_pct": None,  # computed from history, not real-time
                "trading_halted": trading_halted,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "xstocks_api_v2",
                "status": "ok",
            })
        except Exception as e:
            results.append({
                "issuer": "xstocks",
                "underlying": underlying,
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "xstocks_api_v2",
                "status": "error",
                "error": str(e),
            })
    return results


if __name__ == "__main__":
    import json
    print(json.dumps(fetch_xstocks_data(), indent=2))

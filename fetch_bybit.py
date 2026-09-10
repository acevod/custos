"""
Data puller for xStocks as listed on Bybit Spot (NVDAX, TSLAX, AAPLX,
AMZNX, GOOGLX). Underlying token is issued by Backed Finance on
Solana; Bybit is used here purely as the data source because it
exposes a standard public order book (bid/ask/volume), unlike
xStocks' own API which only returns a single price with no spread
or volume data.

Public endpoint: https://api.bybit.com/v5/market/tickers
No API key required for the data itself. HOWEVER: Bybit blocks
requests from many cloud/datacenter IPs (including GitHub Actions
runners) with a 403 at the CDN level - this is a network-level
block, not an auth issue. To work around it, requests are routed
through a Cloudflare Worker reverse proxy (see
cloudflare-worker-bybit-proxy.js) whenever BYBIT_PROXY_BASE_URL is
set. If that env var is not set, this falls back to calling Bybit
directly - useful for local testing from a non-blocked IP (e.g. a
home connection), but will likely fail with 403 when run from
GitHub Actions unless the proxy is configured.

Symbol verification status:
  All 5 (NVDAX, TSLAX, AAPLX, AMZNX, GOOGLX) confirmed via Bybit's
  official xStocks listing announcement. MSFTX checked and NOT
  available on Bybit (confirmed by manual test) - that's why MSFT
  was dropped from the 5-stock lineup in favor of GOOGL.
"""

import os
import requests
from datetime import datetime, timezone

_PROXY_BASE = os.environ.get("BYBIT_PROXY_BASE_URL")  # e.g. https://custos-bybit-proxy.<user>.workers.dev
_DIRECT_BASE = "https://api.bybit.com"
BASE_URL = f"{(_PROXY_BASE or _DIRECT_BASE).rstrip('/')}/v5/market"

SYMBOLS = {
    "NVDA": "NVDAXUSDT",
    "TSLA": "TSLAXUSDT",
    "AAPL": "AAPLXUSDT",
    "AMZN": "AMZNXUSDT",
    "GOOGL": "GOOGLXUSDT",
}


def fetch_ticker(symbol: str) -> dict:
    """Fetch ticker: lastPrice, bid1Price/ask1Price, volume24h, etc."""
    url = f"{BASE_URL}/tickers"
    params = {"category": "spot", "symbol": symbol}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    rows = data.get("result", {}).get("list", [])
    return rows[0] if rows else {}


def calculate_spread_pct(bid: float | None, ask: float | None) -> float | None:
    """Spread from bid/ask, as a percentage of mid price."""
    if bid is None or ask is None:
        return None
    mid = (bid + ask) / 2
    if mid == 0:
        return None
    return round((ask - bid) / mid * 100, 4)


def fetch_bybit_data() -> list[dict]:
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

            bid = float(ticker.get("bid1Price", 0)) or None
            ask = float(ticker.get("ask1Price", 0)) or None

            results.append({
                "issuer": "bybit_xstocks",
                "underlying": underlying,
                "symbol": symbol,
                "price": float(ticker.get("lastPrice", 0)) or None,
                "bid": bid,
                "ask": ask,
                "volume_24h": float(ticker.get("volume24h", 0)) or None,
                "spread_pct": calculate_spread_pct(bid, ask),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "bybit_api_v5",
                "status": "ok",
            })
        except Exception as e:
            results.append({
                "issuer": "bybit_xstocks",
                "underlying": underlying,
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "bybit_api_v5",
                "status": "error",
                "error": str(e),
            })
    return results


if __name__ == "__main__":
    import json
    print(json.dumps(fetch_bybit_data(), indent=2))

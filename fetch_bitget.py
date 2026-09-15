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
import math

import requests
from datetime import datetime, timezone

BASE_URL = "https://api.bitget.com/api/v3/market"

MAX_RETRIES = 3          # total attempts per symbol
BACKOFF_BASE_SECONDS = 2 # 2s, 4s between retries
MAX_MARKET_DATA_AGE_SECONDS = 15 * 60

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


def _finite_positive(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value > 0


def _finite_non_negative(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value >= 0


def _parse_source_timestamp(raw_ts) -> datetime:
    """Parse Bitget's millisecond epoch timestamp into an aware UTC datetime."""
    try:
        ts_ms = int(raw_ts)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid Bitget ticker timestamp: {raw_ts!r}") from exc
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)


def calculate_spread_pct(bid: float | None, ask: float | None) -> float | None:
    """Spread from bid/ask, as a percentage of mid price."""
    if not _finite_positive(bid) or not _finite_positive(ask) or ask < bid:
        return None
    mid = (bid + ask) / 2
    if not math.isfinite(mid) or mid <= 0:
        return None
    spread = (ask - bid) / mid * 100
    return round(spread, 4) if math.isfinite(spread) else None


def validate_ticker(ticker: dict) -> tuple[dict, datetime]:
    """Validate the complete executable market-data boundary before scoring."""
    if not isinstance(ticker, dict):
        raise ValueError("ticker response is not an object")

    source_dt = _parse_source_timestamp(ticker.get("ts"))
    age_seconds = (datetime.now(timezone.utc) - source_dt).total_seconds()
    if age_seconds < -60:
        raise ValueError(f"Bitget ticker timestamp is too far in the future: {age_seconds:.1f}s")
    if age_seconds > MAX_MARKET_DATA_AGE_SECONDS:
        raise ValueError(f"stale Bitget ticker: {age_seconds:.0f}s old (max {MAX_MARKET_DATA_AGE_SECONDS}s)")

    def number(field: str, positive: bool = False, allow_none: bool = False):
        raw = ticker.get(field)
        if raw in (None, "") and allow_none:
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid numeric field {field}: {raw!r}") from exc
        valid = _finite_positive(value) if positive else _finite_non_negative(value)
        if not valid:
            raise ValueError(f"invalid numeric field {field}: {value!r}")
        return value

    price = number("lastPrice", positive=True)
    bid = number("bid1Price", positive=True)
    ask = number("ask1Price", positive=True)
    bid_size = number("bid1Size", allow_none=True)
    ask_size = number("ask1Size", allow_none=True)
    volume = number("volume24h", allow_none=True)

    if ask < bid:
        raise ValueError(f"invalid order book: ask {ask} < bid {bid}")

    return {
        "price": price, "bid": bid, "ask": ask,
        "bid_size": bid_size, "ask_size": ask_size,
        "volume_24h": volume,
        "spread_pct": calculate_spread_pct(bid, ask),
        "source_timestamp": source_dt.isoformat(),
        "data_age_seconds": round(max(0.0, age_seconds), 3),
    }, source_dt


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

            market, source_dt = validate_ticker(ticker)
            results.append({
                "underlying": underlying,
                "symbol": symbol,
                **market,
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

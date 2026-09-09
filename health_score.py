"""
Composite Health Score calculation for issuer risk monitoring.

Now that all three issuers (Bitget, Binance, Bybit-xStocks) are
sourced from standard exchange order books, every issuer has the
same shape of data (price, bid, ask, volume) - no proxy metrics
needed anymore.

Score is a weighted combination of:
  1. Premium/Discount (30%) - how far an issuer's price deviates
                              from the cross-issuer consensus
                              (median) price
  2. Spread             (25%) - real bid/ask spread, same formula
                              for all three issuers
  3. Volume Trend        (20%) - current volume vs its own recent
                              baseline, same formula for all three
  4. Weekend/After-Hours (15%) - flags reduced-liquidity windows
  5. Sentiment            (10%) - optional, supplied externally from
                              the bitget-signal news/sentiment skills

All component scores are normalized to 0.0 (unhealthy) - 1.0 (healthy).
Weights and thresholds here are heuristic / manually tuned, not the
result of backtesting - this is disclosed openly in the project
write-up rather than presented as a validated model.
"""

from datetime import datetime, timezone
from statistics import mean, median

WEIGHTS = {
    "premium_discount": 0.30,
    "spread": 0.25,
    "volume_trend": 0.20,
    "weekend": 0.15,
    "sentiment": 0.10,
}

HISTORY_WINDOW = 20  # how many recent points used for volume baseline


# ── Component 1: Premium / Discount ───────────────────────────

def calculate_consensus_price(issuer_prices: dict[str, float | None]) -> float | None:
    """
    With 3+ issuers, 'reference price' is the median across all
    issuers currently reporting a valid price - more robust than
    picking one issuer arbitrarily as ground truth, and more robust
    than a mean (a single outlier issuer can't drag the consensus).
    """
    valid = [p for p in issuer_prices.values() if p is not None]
    if not valid:
        return None
    return median(valid)


def score_premium_discount(issuer_price: float | None, consensus_price: float | None) -> float | None:
    """
    Deviation of one issuer's price from the cross-issuer consensus.
    Heuristic curve: 0% deviation -> score 1.0, 2%+ deviation -> ~0.
    """
    if issuer_price is None or consensus_price is None or consensus_price == 0:
        return None
    deviation_pct = abs(issuer_price - consensus_price) / consensus_price * 100
    score = max(0.0, 1.0 - (deviation_pct / 2.0))
    return round(score, 4)


# ── Component 2: Spread ───────────────────────────────────────

def score_spread(spread_pct: float | None) -> float | None:
    """
    Heuristic curve: 0% spread -> 1.0, 1%+ spread -> ~0.
    Same formula for all three issuers now (all have real bid/ask).
    """
    if spread_pct is None:
        return None
    score = max(0.0, 1.0 - (spread_pct / 1.0))
    return round(score, 4)


# ── Component 3: Volume Trend ─────────────────────────────────

def score_volume_trend(current_volume: float | None, volume_history: list[float]) -> float | None:
    """
    Compares current volume against the mean of recent history.
    A sharp drop vs baseline is treated as a liquidity warning sign.
    Same formula for all three issuers now.
    """
    if current_volume is None or len(volume_history) < 3:
        return None
    baseline = mean(volume_history)
    if baseline == 0:
        return None
    ratio = current_volume / baseline

    # ratio >= 1.0 (volume at/above baseline) -> score 1.0
    # ratio approaching 0 (volume collapsing) -> score approaching 0
    score = min(1.0, max(0.0, ratio))
    return round(score, 4)


# ── Component 4: Weekend / After-Hours ────────────────────────

def score_weekend_afterhours(timestamp: datetime) -> float:
    """
    Flags reduced-liquidity windows. This does not fail the score
    outright - it lowers it slightly to reflect thinner backing
    liquidity during these windows (issuers typically supply
    weekend/after-hours liquidity internally rather than routing to
    NASDAQ/NYSE, which behaves differently from regular market hours).

    Simplified UTC-based check - does not account for US market
    holidays or DST transitions precisely; good enough as a
    hackathon-scope heuristic.
    """
    weekday = timestamp.weekday()  # 0=Mon ... 6=Sun
    hour_utc = timestamp.hour

    is_weekend = weekday >= 5  # Sat, Sun
    # US market hours roughly 13:30-20:00 UTC (9:30am-4pm ET, ignoring DST)
    is_after_hours = not (13 <= hour_utc < 20)

    if is_weekend:
        return 0.6
    if is_after_hours:
        return 0.8
    return 1.0


# ── Composite score ────────────────────────────────────────────

def calculate_composite_score(components: dict) -> dict:
    """
    Combines component scores into one weighted composite.
    Missing components are excluded and weights are renormalized
    over whatever is actually available - this avoids silently
    treating "no data" as "healthy" or "unhealthy".

    components: dict with keys matching WEIGHTS, values are
    float 0-1 or None.

    Returns dict with the final score plus which components
    were actually used (for transparency in the log).
    """
    used = {k: v for k, v in components.items() if v is not None}
    if not used:
        return {"score": None, "components_used": [], "note": "no data available"}

    total_weight = sum(WEIGHTS[k] for k in used)
    weighted_sum = sum(WEIGHTS[k] * v for k, v in used.items())
    final_score = weighted_sum / total_weight

    return {
        "score": round(final_score, 4),
        "components_used": list(used.keys()),
        "components_raw": used,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def classify_score(score: float | None) -> str:
    """Maps a numeric score to a dashboard-friendly label."""
    if score is None:
        return "unknown"
    if score >= 0.8:
        return "healthy"
    if score >= 0.5:
        return "watch"
    return "red_flag"


def score_all_issuers(
    issuer_data: dict[str, dict],
    volume_histories: dict[str, list[float]],
    now: datetime,
) -> dict[str, dict]:
    """
    Convenience wrapper: scores every issuer for one underlying stock
    in a single call, sharing the same consensus price across all of
    them.

    issuer_data: {"bitget": {"price":.., "spread_pct":.., "volume_24h":..}, ...}
    volume_histories: {"bitget": [recent volumes...], ...}
    """
    consensus_price = calculate_consensus_price(
        {name: d.get("price") for name, d in issuer_data.items()}
    )

    results = {}
    for issuer_name, data in issuer_data.items():
        components = {
            "premium_discount": score_premium_discount(data.get("price"), consensus_price),
            "spread": score_spread(data.get("spread_pct")),
            "volume_trend": score_volume_trend(
                data.get("volume_24h"), volume_histories.get(issuer_name, [])
            ),
            "weekend": score_weekend_afterhours(now),
            "sentiment": data.get("sentiment_score"),  # wired in later from LLM layer
        }
        result = calculate_composite_score(components)
        result["label"] = classify_score(result["score"])
        results[issuer_name] = result

    return results


if __name__ == "__main__":
    import json

    now = datetime.now(timezone.utc)

    # Fabricated example: 3 issuers reporting on the same underlying (NVDA)
    issuer_data = {
        "bitget": {"price": 224.75, "spread_pct": 0.067, "volume_24h": 77_826_397},
        "binance": {"price": 224.60, "spread_pct": 0.05, "volume_24h": 12_500_000},
        "bybit_xstocks": {"price": 224.68, "spread_pct": 0.08, "volume_24h": 9_800_000},
    }
    volume_histories = {
        "bitget": [70_000_000, 75_000_000, 80_000_000],
        "binance": [11_000_000, 12_000_000, 13_000_000],
        "bybit_xstocks": [9_000_000, 9_500_000, 10_000_000],
    }

    results = score_all_issuers(issuer_data, volume_histories, now)
    print(json.dumps(results, indent=2))

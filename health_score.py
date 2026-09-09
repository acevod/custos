"""
Composite Health Score calculation for issuer risk monitoring.

Score is a weighted combination of:
  1. Premium/Discount  (30%) - how far issuer price deviates from
                               the cross-issuer reference price
  2. Spread             (25%) - bid/ask spread (Bitget: real,
                               xStocks: volatility-based proxy)
  3. Flow Trend          (20%) - Bitget: trading volume vs baseline.
                               xStocks: circulating-supply change vs
                               baseline (proxy for redemption flow,
                               since the public API has no volume field)
  4. Weekend/After-Hours (15%) - flags reduced-liquidity windows
  5. Sentiment            (10%) - optional, supplied externally from
                               the bitget-signal news/sentiment skills

All component scores are normalized to 0.0 (unhealthy) - 1.0 (healthy).
Weights and thresholds here are heuristic / manually tuned, not the
result of backtesting - this is disclosed openly in the project
write-up rather than presented as a validated model.
"""

from datetime import datetime, timezone
from statistics import mean, stdev

WEIGHTS = {
    "premium_discount": 0.30,
    "spread": 0.25,
    # "flow_trend" is trading volume for Bitget, circulating-supply
    # change for xStocks - same weight slot, different underlying
    # metric per issuer (see score_volume_trend / score_supply_flow).
    "flow_trend": 0.20,
    "weekend": 0.15,
    "sentiment": 0.10,
}

# How many recent history points to use for the xStocks volatility
# proxy and for the Bitget volume baseline.
HISTORY_WINDOW = 20


# ── Component 1: Premium / Discount ───────────────────────────

def score_premium_discount(bitget_price: float | None, xstocks_price: float | None) -> float | None:
    """
    Since we don't pull an independent 'ground truth' NVDA price,
    the two issuer prices are compared against each other as the
    reference. Large deviation between them = one side is mispriced
    relative to the other, which is itself a health signal.
    Returns None if either price is missing.
    """
    if bitget_price is None or xstocks_price is None:
        return None
    mid = (bitget_price + xstocks_price) / 2
    if mid == 0:
        return None
    deviation_pct = abs(bitget_price - xstocks_price) / mid * 100

    # Heuristic curve: 0% deviation -> score 1.0, 2%+ deviation -> score ~0
    score = max(0.0, 1.0 - (deviation_pct / 2.0))
    return round(score, 4)


# ── Component 2: Spread ───────────────────────────────────────

def score_spread(spread_pct: float | None) -> float | None:
    """
    Direct spread score (used for Bitget, which has real bid/ask).
    Heuristic curve: 0% spread -> 1.0, 1%+ spread -> ~0.
    """
    if spread_pct is None:
        return None
    score = max(0.0, 1.0 - (spread_pct / 1.0))
    return round(score, 4)


def calculate_volatility_proxy(price_history: list[float]) -> float | None:
    """
    Substitute for xStocks spread, since public API has no bid/ask.
    Uses coefficient of variation (stdev / mean) of recent prices as
    a rough liquidity-stress signal - the reasoning being that
    degraded liquidity tends to show up as increased short-term price
    noise, not just a wider bid/ask. This is an approximation, not a
    direct measurement, and is documented as such in the write-up.
    Needs at least 3 data points to be meaningful.
    """
    if len(price_history) < 3:
        return None
    m = mean(price_history)
    if m == 0:
        return None
    cv_pct = (stdev(price_history) / m) * 100
    return round(cv_pct, 4)


# ── Component 3: Flow Trend (volume for Bitget, supply for xStocks) ──

def score_volume_trend(current_volume: float | None, volume_history: list[float]) -> float | None:
    """
    Bitget-specific: compares current trading volume against the mean
    of recent history. A sharp drop vs baseline is a liquidity warning.
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


def score_supply_flow(current_supply: float | None, supply_history: list[float]) -> float | None:
    """
    xStocks-specific substitute for volume_trend, since the public API
    has no trading-volume field. Circulating supply rises on mint and
    falls on redeem, so this measures how much supply has shrunk
    relative to its recent baseline - a direct proxy for redemption
    pressure, which is arguably closer to what we actually care about
    (issuer/counterparty risk) than raw trading volume would be.

    Only the downside (supply shrinking) is treated as a risk signal;
    supply growing (more minting) scores as healthy/neutral.
    """
    if current_supply is None or len(supply_history) < 3:
        return None
    baseline = mean(supply_history)
    if baseline == 0:
        return None
    ratio = current_supply / baseline

    # ratio >= 1.0 (supply stable or growing) -> score 1.0
    # ratio dropping below 1.0 (net redemptions) -> score drops
    score = min(1.0, max(0.0, ratio))
    return round(score, 4)


# ── Component 4: Weekend / After-Hours ────────────────────────

def score_weekend_afterhours(timestamp: datetime) -> float:
    """
    Flags reduced-liquidity windows. This does not fail the score
    outright - it lowers it slightly to reflect thinner backing
    liquidity during these windows, per Bitget's own disclosure that
    weekend/after-hours liquidity is internally supplied and behaves
    differently from regular market-hours routing to NASDAQ/NYSE.

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


if __name__ == "__main__":
    # Manual smoke test with fabricated numbers.
    import json

    now = datetime.now(timezone.utc)

    bitget_price = 224.75
    xstocks_price = 224.68
    fake_price_history = [224.1, 224.3, 224.5, 224.68, 224.2]
    fake_volume_history = [70_000_000, 75_000_000, 80_000_000]
    fake_supply_history = [10_000_000, 10_050_000, 10_020_000]

    print("--- Bitget composite (uses trading volume) ---")
    bitget_components = {
        "premium_discount": score_premium_discount(bitget_price, xstocks_price),
        "spread": score_spread(0.067),  # from Bitget bid/ask example
        "flow_trend": score_volume_trend(77_826_397, fake_volume_history),
        "weekend": score_weekend_afterhours(now),
        "sentiment": None,  # not wired up yet - comes from LLM layer
    }
    bitget_result = calculate_composite_score(bitget_components)
    bitget_result["label"] = classify_score(bitget_result["score"])
    print(json.dumps(bitget_result, indent=2))

    print("\n--- xStocks composite (uses circulating supply proxy) ---")
    xstocks_spread_proxy_pct = calculate_volatility_proxy(fake_price_history)
    xstocks_components = {
        "premium_discount": score_premium_discount(bitget_price, xstocks_price),
        "spread": score_spread(xstocks_spread_proxy_pct),
        "flow_trend": score_supply_flow(9_950_000, fake_supply_history),
        "weekend": score_weekend_afterhours(now),
        "sentiment": None,
    }
    xstocks_result = calculate_composite_score(xstocks_components)
    xstocks_result["label"] = classify_score(xstocks_result["score"])
    print(json.dumps(xstocks_result, indent=2))

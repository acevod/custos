"""
Composite Health Score for single-issuer (Bitget) rToken structural
risk monitoring. Every component is self-referential - each rToken
is scored against ITS OWN historical baseline, never against price
direction or against other rTokens. This keeps the system a
structural/liquidity risk monitor, not a directional stock picker:
the score should be able to flag a token even while its price is
rising, and stay calm during a price drop if the wrapper's own
liquidity mechanics look normal.

Components (all normalized 0.0 unhealthy - 1.0 healthy):
  1. Spread            (25%) - real bid/ask spread
  2. Order book depth   (20%) - bid+ask size vs this token's own
                                recent baseline (thin book = fragile
                                liquidity even if spread looks tight)
  3. Volume trend        (20%) - current volume vs its own baseline
  4. Abnormal movement    (20%) - how large the latest price move is
                                relative to this token's own recent
                                volatility, direction-agnostic (a
                                sharp move up counts the same as a
                                sharp move down)
  5. Weekend/after-hours   (15%) - Bitget supplies liquidity
                                internally outside NASDAQ/NYSE hours,
                                which changes (not necessarily
                                worsens) the liquidity mechanism

Weights are heuristic / manually tuned, not backtested - disclosed
as such in the project write-up.
"""

from datetime import datetime, timezone
from statistics import mean, stdev

WEIGHTS = {
    "spread": 0.25,
    "depth": 0.20,
    "volume_trend": 0.20,
    "abnormal_movement": 0.20,
    "weekend": 0.15,
}

HISTORY_WINDOW = 20  # rolling window size for all self-baselines


# ── Component 1: Spread ───────────────────────────────────────

def score_spread(spread_pct: float | None) -> float | None:
    """Heuristic curve: 0% spread -> 1.0, 1%+ spread -> ~0."""
    if spread_pct is None:
        return None
    return round(max(0.0, 1.0 - (spread_pct / 1.0)), 4)


# ── Component 2: Order book depth ─────────────────────────────

def score_depth(bid_size: float | None, ask_size: float | None,
                 depth_history: list[float]) -> float | None:
    """
    Total top-of-book size (bid+ask) vs this token's own recent
    average. A thin book relative to its own normal depth is a
    fragility signal even when the spread itself still looks tight.
    """
    if bid_size is None or ask_size is None or len(depth_history) < 3:
        return None
    current_depth = bid_size + ask_size
    baseline = mean(depth_history)
    if baseline == 0:
        return None
    ratio = current_depth / baseline
    return round(min(1.0, max(0.0, ratio)), 4)


# ── Component 3: Volume trend ─────────────────────────────────

def score_volume_trend(current_volume: float | None, volume_history: list[float]) -> float | None:
    """Current volume vs its own recent baseline."""
    if current_volume is None or len(volume_history) < 3:
        return None
    baseline = mean(volume_history)
    if baseline == 0:
        return None
    ratio = current_volume / baseline
    return round(min(1.0, max(0.0, ratio)), 4)


# ── Component 4: Abnormal movement (direction-agnostic) ───────

def score_abnormal_movement(price_history: list[float]) -> float | None:
    """
    Compares the most recent price change to this token's own
    typical volatility (stdev of recent % returns). Direction is
    ignored on purpose - a sharp move up is scored the same as a
    sharp move down, since the point is detecting unusual mechanical
    behavior in the wrapper, not predicting where price goes next.
    Needs at least 4 price points (3 returns) to be meaningful.
    """
    if len(price_history) < 4:
        return None

    returns = []
    for i in range(1, len(price_history)):
        prev, curr = price_history[i - 1], price_history[i]
        if prev:
            returns.append((curr - prev) / prev * 100)

    if len(returns) < 3:
        return None

    historical_returns = returns[:-1]
    latest_return = returns[-1]

    vol = stdev(historical_returns) if len(historical_returns) >= 2 else None
    if not vol or vol == 0:
        return None

    z_score = abs(latest_return) / vol
    # z >= 3 (3 standard deviations) -> score 0; z == 0 -> score 1.0
    return round(max(0.0, 1.0 - (z_score / 3.0)), 4)


# ── Component 5: Weekend / after-hours ────────────────────────

def score_weekend_afterhours(timestamp: datetime) -> float:
    """
    Small, deliberate penalty (not a hard drop) reflecting that
    Bitget supplies liquidity internally outside regular NASDAQ/NYSE
    hours - a different mechanism, not automatically a worse one.
    Simplified UTC check, doesn't account for US market holidays/DST.
    """
    weekday = timestamp.weekday()  # 0=Mon ... 6=Sun
    hour_utc = timestamp.hour
    is_weekend = weekday >= 5
    is_after_hours = not (13 <= hour_utc < 20)

    if is_weekend:
        return 0.6
    if is_after_hours:
        return 0.8
    return 1.0


# ── Composite score ────────────────────────────────────────────

def calculate_composite_score(components: dict) -> dict:
    """
    Weighted composite over whatever components have data. Missing
    components are excluded and weights renormalized over what's
    available - avoids treating "no data yet" as healthy or unhealthy.
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
    if score is None:
        return "unknown"
    if score >= 0.8:
        return "healthy"
    if score >= 0.5:
        return "watch"
    return "red_flag"


def score_stock(data: dict, history: dict) -> dict:
    """
    Convenience wrapper: scores one stock given its latest fetch
    entry and its own historical series.
    data: single entry from fetch_bitget_data() (one stock)
    history: {"volume": [...], "price": [...], "depth": [...]}
    """
    now = datetime.now(timezone.utc)
    components = {
        "spread": score_spread(data.get("spread_pct")),
        "depth": score_depth(data.get("bid_size"), data.get("ask_size"),
                              history.get("depth", [])),
        "volume_trend": score_volume_trend(data.get("volume_24h"),
                                            history.get("volume", [])),
        "abnormal_movement": score_abnormal_movement(history.get("price", [])),
        "weekend": score_weekend_afterhours(now),
    }
    result = calculate_composite_score(components)
    result["label"] = classify_score(result["score"])
    return result


if __name__ == "__main__":
    import json

    # Fabricated example: healthy token with enough history
    fake_data = {"spread_pct": 0.07, "bid_size": 65, "ask_size": 140, "volume_24h": 78_000_000}
    fake_history = {
        "volume": [70_000_000, 75_000_000, 80_000_000],
        "price": [224.1, 224.3, 224.5, 224.68, 224.75],
        "depth": [180, 190, 200],
    }
    print(json.dumps(score_stock(fake_data, fake_history), indent=2))

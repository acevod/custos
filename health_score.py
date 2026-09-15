"""
Composite Health Score for single-issuer (Bitget) rToken structural
risk monitoring. Three of the five components are self-referential -
scored against each rToken's OWN historical baseline (depth, volume
trend, abnormal movement). Spread uses a fixed absolute threshold
rather than a historical baseline, and weekend/after-hours is
calendar-based rather than data-driven - both are disclosed as such
below rather than folded into a blanket "all self-referential" claim.
None of the five compares a token against price direction or against
other rTokens: the score should be able to flag a token even while
its price is rising, and stay calm during a price drop if the
wrapper's own liquidity mechanics look normal.

Components (all normalized 0.0 unhealthy - 1.0 healthy):
  1. Spread              (25%) - real bid/ask spread against a fixed
                                absolute threshold, NOT compared to
                                this token's own historical spread
  2. Top-of-book size     (20%) - best-bid + best-ask size vs this
                                token's own recent baseline. This is
                                top-of-book liquidity only, not true
                                multi-level order-book depth (Bitget's
                                deeper Reality order book is a
                                separate, access-gated endpoint) -
                                a thin top-of-book can still coexist
                                with a thick book further down
  3. Volume trend          (20%) - current volume vs its own baseline
  4. Abnormal movement      (20%) - how large the latest price move is
                                relative to this token's own recent
                                volatility, direction-agnostic (a
                                sharp move up counts the same as a
                                sharp move down)
  5. Weekend/after-hours     (15%) - a fixed calendar-based penalty
                                (not derived from this token's own
                                data) reflecting that Bitget supplies
                                liquidity internally outside NASDAQ/
                                NYSE hours, which changes (not
                                necessarily worsens) the mechanism.
                                Hardcoded UTC hours - does not account
                                for DST or US market holidays.

Weights are heuristic / manually tuned, not backtested - disclosed
as such in the project write-up.
"""

from datetime import datetime, timezone
from statistics import mean, stdev
import math

WEIGHTS = {
    "spread": 0.25,
    "depth": 0.20,
    "volume_trend": 0.20,
    "abnormal_movement": 0.20,
    "weekend": 0.15,
}

HISTORY_WINDOW = 42  # ~7 days at 4-hour cadence - spans a full weekday+weekend cycle


# ── Component 1: Spread ───────────────────────────────────────

def score_spread(spread_pct: float | None) -> float | None:
    """Heuristic curve: 0% spread -> 1.0, 1%+ spread -> ~0."""
    if spread_pct is None or not math.isfinite(spread_pct) or spread_pct < 0:
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
    if (bid_size is None or ask_size is None or
            not math.isfinite(bid_size) or not math.isfinite(ask_size) or
            bid_size < 0 or ask_size < 0 or len(depth_history) < 3):
        return None
    clean_history = [x for x in depth_history if math.isfinite(x) and x >= 0]
    if len(clean_history) < 3:
        return None
    current_depth = bid_size + ask_size
    baseline = mean(clean_history)
    if baseline == 0:
        return None
    ratio = current_depth / baseline
    return round(min(1.0, max(0.0, ratio)), 4)


# ── Component 3: Volume trend ─────────────────────────────────

def score_volume_trend(current_volume: float | None, volume_history: list[float]) -> float | None:
    """Current volume vs its own recent baseline."""
    if (current_volume is None or not math.isfinite(current_volume) or
            current_volume < 0 or len(volume_history) < 3):
        return None
    clean_history = [x for x in volume_history if math.isfinite(x) and x >= 0]
    if len(clean_history) < 3:
        return None
    baseline = mean(clean_history)
    if baseline == 0:
        return None
    ratio = current_volume / baseline
    return round(min(1.0, max(0.0, ratio)), 4)


# ── Component 4: Abnormal movement (direction-agnostic) ───────

def score_abnormal_movement(price_history: list[float], current_price: float | None) -> float | None:
    """
    Compares the latest price move (from the most recent historical
    reading to current_price) against this token's own typical
    volatility (stdev of past % returns). Direction is ignored on
    purpose - a sharp move up is scored the same as a sharp move
    down, since the point is detecting unusual mechanical behavior
    in the wrapper, not predicting where price goes next.

    price_history must NOT include the current reading - it's the
    pure past baseline, kept uncontaminated the same way depth/volume
    are. current_price is this run's fresh reading, passed separately.
    Needs at least 3 past prices (2 historical returns) plus a
    current price to be meaningful.
    """
    if (current_price is None or not math.isfinite(current_price) or
            current_price <= 0 or len(price_history) < 3):
        return None

    clean_prices = [p for p in price_history if math.isfinite(p) and p > 0]
    if len(clean_prices) < 3:
        return None

    historical_returns = []
    for i in range(1, len(clean_prices)):
        prev, curr = clean_prices[i - 1], clean_prices[i]
        if prev:
            historical_returns.append((curr - prev) / prev * 100)

    if len(historical_returns) < 2:
        return None

    last_hist_price = clean_prices[-1]
    if not last_hist_price:
        return None
    latest_return = (current_price - last_hist_price) / last_hist_price * 100

    vol = stdev(historical_returns)
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
    history: {"volume": [...], "price": [...], "depth": [...]} -
    IMPORTANT: this must be the baseline BEFORE this run's values are
    added (the caller is responsible for scoring first, then updating
    history afterward - see main.py's run()). Passing a history that
    already includes the current reading silently biases volume_trend
    and depth toward ~1.0, since the current value would be part of
    its own baseline average.
    """
    now = datetime.now(timezone.utc)
    components = {
        "spread": score_spread(data.get("spread_pct")),
        "depth": score_depth(data.get("bid_size"), data.get("ask_size"),
                              history.get("depth", [])),
        "volume_trend": score_volume_trend(data.get("volume_24h"),
                                            history.get("volume", [])),
        "abnormal_movement": score_abnormal_movement(history.get("price", []), data.get("price")),
        "weekend": score_weekend_afterhours(now),
    }
    result = calculate_composite_score(components)
    # A partial score is useful for observability, but it must not be
    # presented as fully healthy when the action gate has not matured.
    if len(result.get("components_used", [])) < 4:
        result["label"] = "warming_up"
    else:
        result["label"] = classify_score(result["score"])
    return result


if __name__ == "__main__":
    import json

    # Fabricated example: healthy token with enough PAST history.
    # Note: fake_history contains only PAST values - the current
    # reading lives in fake_data, kept separate (see score_stock's
    # docstring on why history must not include the current run).
    fake_data = {"price": 224.75, "spread_pct": 0.07, "bid_size": 65, "ask_size": 140, "volume_24h": 78_000_000}
    fake_history = {
        "volume": [70_000_000, 75_000_000, 80_000_000],
        "price": [223.5, 224.1, 224.3, 224.5],
        "depth": [180, 190, 200],
    }
    print(json.dumps(score_stock(fake_data, fake_history), indent=2))

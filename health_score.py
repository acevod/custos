"""
Composite Health Score for single-issuer (Bitget) rToken structural
risk monitoring. Two of the four components are self-referential -
scored against each rToken's OWN historical baseline (depth and
abnormal movement). Spread uses a fixed absolute threshold
rather than a historical baseline, and weekend/after-hours is
calendar-based rather than data-driven - both are disclosed as such
below rather than folded into a blanket "all self-referential" claim.
None of the four compares a token against price direction or against
other rTokens: the score should be able to flag a token even while
its price is rising, and stay calm during a price drop if the
wrapper's own liquidity mechanics look normal.

Components (all normalized 0.0 unhealthy - 1.0 healthy):
  1. Spread              (30%) - real bid/ask spread against a fixed
                                absolute threshold, NOT compared to
                                this token's own historical spread
  2. Top-of-book size     (25%) - best-bid + best-ask size vs this
                                token's own recent baseline. This is
                                top-of-book liquidity only, not true
                                multi-level order-book depth (Bitget's
                                deeper Reality order book is a
                                separate, access-gated endpoint) -
                                a thin top-of-book can still coexist
                                with a thick book further down
  3. Abnormal movement      (25%) - how large the latest price move is
                                relative to this token's own recent
                                volatility, direction-agnostic (a
                                sharp move up counts the same as a
                                sharp move down)
  4. Weekend/after-hours     (20%) - a fixed calendar-based penalty
                                (not derived from this token's own
                                data). Two different, verified regimes:
                                weekday pre/after-market hours (0.8)
                                still largely mirror the underlying
                                stock's own (lower) activity in that
                                window, per Bitget's hourly candles -
                                same liquidity source, just thinner.
                                The weekend window, Fri 20:00 ET ->
                                Sun 20:00 ET (0.6), is where Bitget's
                                rToken market genuinely stops following
                                the stock and runs on its own internal
                                liquidity instead - a different
                                mechanism, confirmed against 5 weekends
                                of candles and Labor Day 2026. Uses
                                America/New_York (DST-aware) and a
                                small holiday table (see
                                US_MARKET_HOLIDAYS - extend it yearly).

Volume is deliberately NOT a component: Bitget's `volume24h` is a
counter that resets at 16:00 UTC and, on weekdays, mirrors the volume of
the underlying US stock rather than trading on Bitget itself (verified
against Bitget's own hourly candles), so it says nothing about the
wrapper's liquidity.

Weights are heuristic / manually tuned, not backtested - disclosed
as such in the project write-up.
"""

from datetime import date, datetime, timedelta, timezone
from statistics import median, stdev
import math

try:  # zoneinfo needs system tzdata; fall back to a fixed UTC-4/5 guess if absent
    from zoneinfo import ZoneInfo
    _NY = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover - depends on the host's tzdata
    _NY = None

WEIGHTS = {
    "spread": 0.30,
    "depth": 0.25,
    "abnormal_movement": 0.25,
    "weekend": 0.20,
}
# A score needs ALL components before it may drive an action / a health label.
MIN_COMPONENTS = len(WEIGHTS)

HISTORY_WINDOW = 42  # ~7-9 days at the real (jittery) 4-6h cadence - spans a weekday+weekend cycle

# ── Data-quality guards (audit C-1) ───────────────────────────
# Regime-aware baselines: top-of-book size can behave very differently when the
# US market is open, closed on a weekday night, or in the weekend window
# (observed: 10-100x lower sizes on the first weekend). A reading is
# therefore compared only with history from the SAME regime; the calendar
# component already prices the closed-market effect, so comparing against a
# mixed baseline would double-count it. With too few same-regime points the
# component reports "no signal" (None) instead of guessing.
DEPTH_FULL_SCORE_RATIO = 0.5            # >= 50% of median top-of-book is fully healthy
DEPTH_ZERO_SCORE_RATIO = 0.05
DEPTH_SMOOTH_POINTS = 3                 # median of (current + last N-1 readings)
MIN_REGIME_POINTS = 3                   # need this many same-regime readings for a regime baseline
MAX_RETURN_GAP_HOURS = 12.0             # don't score moves measured across a longer gap
VOL_FLOOR_PCT_PER_SQRT_HOUR = 0.10      # floor on the volatility estimate (avoids z blow-ups)


# ── Component 1: Spread ───────────────────────────────────────

def score_spread(spread_pct: float | None) -> float | None:
    """Heuristic curve: 0% spread -> 1.0, 1%+ spread -> ~0."""
    if spread_pct is None or not math.isfinite(spread_pct) or spread_pct < 0:
        return None
    return round(max(0.0, 1.0 - (spread_pct / 1.0)), 4)


# ── Component 2: Order book depth ─────────────────────────────

def _clean(values):
    return [x for x in values if isinstance(x, (int, float)) and math.isfinite(x) and x >= 0]


def _linear(ratio: float, zero_at: float, full_at: float) -> float:
    """0.0 at/below zero_at, 1.0 at/above full_at, linear between."""
    if ratio >= full_at:
        return 1.0
    if ratio <= zero_at:
        return 0.0
    return (ratio - zero_at) / (full_at - zero_at)


def _regime(ts: datetime) -> str:
    """'open' (regular US market hours), 'night' (weekday, market closed)
    or 'weekend' (weekend / market holiday). Liquidity levels differ
    enough between these that a reading is only comparable within the
    same regime - 'night' still runs on the same mechanism as 'open'
    (thinner activity, not a different source); only 'weekend' is a
    genuinely different, Bitget-internal mechanism (see
    _is_native_window)."""
    calendar_score = score_weekend_afterhours(ts)
    if calendar_score == 1.0:
        return "open"
    return "night" if calendar_score == 0.8 else "weekend"


def _regime_baseline(values: list, ts_history: list | None, now: datetime | None):
    """Baseline values comparable to a reading taken at `now`.
    When timestamps are aligned 1:1 with `values` (and all valid), only
    readings from the SAME regime (open / night / weekend) are used;
    with fewer than MIN_REGIME_POINTS of them, returns [] (= no signal)
    rather than comparing a weekend reading with weekday liquidity.
    Without usable timestamps it falls back to the whole history."""
    if ts_history is None or now is None or len(ts_history) != len(values):
        return values
    stamps = [_parse_ts(t) for t in ts_history]
    if any(t is None for t in stamps):
        return values
    target = _regime(now)
    same = [v for v, t in zip(values, stamps) if _regime(t) == target]
    return same if len(same) >= MIN_REGIME_POINTS else []


def score_depth(bid_size: float | None, ask_size: float | None,
                 depth_history: list[float], ts_history: list | None = None,
                 now: datetime | None = None) -> float | None:
    """
    Total top-of-book size (bid+ask) vs this token's own recent MEDIAN.
    Top-of-book size is a single noisy snapshot (heavy-tailed: values can
    swing 10-100x between runs with no liquidity event), so:
      - the baseline is the median, not the mean (robust to outliers)
      - the current value is smoothed with the last few readings
      - the curve is tolerant: >= 50% of the median is fully healthy.
    """
    if (bid_size is None or ask_size is None or
            not math.isfinite(bid_size) or not math.isfinite(ask_size) or
            bid_size < 0 or ask_size < 0):
        return None
    clean_history = _clean(_regime_baseline(depth_history, ts_history, now))
    if len(clean_history) < 3:
        return None
    baseline = median(clean_history)
    if baseline == 0:
        return None
    current_depth = bid_size + ask_size
    recent = clean_history[-(DEPTH_SMOOTH_POINTS - 1):] + [current_depth]
    smoothed = median(recent)
    ratio = smoothed / baseline
    return round(_linear(ratio, DEPTH_ZERO_SCORE_RATIO, DEPTH_FULL_SCORE_RATIO), 4)


# ── Component 3: Abnormal movement (direction-agnostic) ───────

def _parse_ts(value):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def score_abnormal_movement(price_history: list[float], current_price: float | None,
                             ts_history: list | None = None,
                             current_ts: datetime | None = None) -> float | None:
    """
    Compares the latest price move (from the most recent historical
    reading to current_price) against this token's own typical
    volatility. Direction is ignored on purpose.

    price_history must NOT include the current reading. When timestamps
    are available (ts_history aligned 1:1 with price_history, plus
    current_ts), returns are normalized by sqrt(elapsed hours) so runs
    spaced 3h or 8h apart are comparable, moves measured across a gap
    longer than MAX_RETURN_GAP_HOURS are skipped, and the volatility
    estimate has a floor (VOL_FLOOR_PCT_PER_SQRT_HOUR). Without
    timestamps it falls back to the legacy equal-spacing behaviour.
    Needs at least 3 past prices (2 historical returns) plus a current
    price to be meaningful.
    """
    if (current_price is None or not math.isfinite(current_price) or
            current_price <= 0 or len(price_history) < 3):
        return None

    have_ts = (ts_history is not None and current_ts is not None
               and len(ts_history) == len(price_history))
    parsed_ts = [_parse_ts(t) for t in ts_history] if have_ts else []
    if have_ts and any(t is None for t in parsed_ts[-3:]):
        have_ts = False  # legacy/unknown timestamps for the recent points

    pairs = []  # (price, ts or None)
    for i, p in enumerate(price_history):
        if isinstance(p, (int, float)) and math.isfinite(p) and p > 0:
            pairs.append((p, parsed_ts[i] if have_ts else None))
    if len(pairs) < 3:
        return None

    def norm_return(prev, curr):
        (p0, t0), (p1, t1) = prev, curr
        ret = (p1 - p0) / p0 * 100
        if not have_ts:
            return ret
        hours = (t1 - t0).total_seconds() / 3600.0
        if hours <= 0 or hours > MAX_RETURN_GAP_HOURS:
            return None
        return ret / math.sqrt(hours)

    historical_returns = []
    for i in range(1, len(pairs)):
        r = norm_return(pairs[i - 1], pairs[i])
        if r is not None:
            historical_returns.append(r)
    if len(historical_returns) < 2:
        return None

    cur_ts = current_ts if current_ts and current_ts.tzinfo else (
        current_ts.replace(tzinfo=timezone.utc) if current_ts else None)
    latest = norm_return(pairs[-1], (current_price, cur_ts))
    if latest is None:
        return None

    vol = stdev(historical_returns)
    if have_ts:
        vol = max(vol, VOL_FLOOR_PCT_PER_SQRT_HOUR)
    if not vol:
        return None

    z_score = abs(latest) / vol
    # z >= 3 (3 standard deviations) -> score 0; z == 0 -> score 1.0
    return round(max(0.0, 1.0 - (z_score / 3.0)), 4)


# ── Component 4: Weekend / after-hours (calendar) ─────────────

# NYSE full-day closures. EXTEND EVERY YEAR (and verify against the NYSE
# calendar) - dates outside this table are treated as normal trading days.
US_MARKET_HOLIDAYS = {
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7),
    date(2026, 11, 26), date(2026, 12, 25),
}


def _to_new_york(timestamp: datetime) -> datetime:
    ts = timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
    if _NY is not None:
        return ts.astimezone(_NY)
    # Fallback without tzdata: approximate US DST (2nd Sun Mar - 1st Sun Nov)
    from datetime import timedelta
    utc = ts.astimezone(timezone.utc)
    dst = 3 <= utc.month <= 10 and not (utc.month == 3 and utc.day < 8) and not (utc.month == 11)
    return utc + timedelta(hours=-4 if dst else -5)


def _is_native_window(ny: datetime) -> bool:
    """The period in which Bitget's rToken liquidity stops following the US
    stock market's 24/5 activity. The liquidity "session day" rolls over at
    20:00 ET (when the overnight session starts), so a moment belongs to the
    session day of (ET time + 4h). It is native when that session day is a
    weekend day or a market holiday:
      * weekend: Fri 20:00 ET -> Sun 20:00 ET
      * holiday: 20:00 ET the evening BEFORE -> 20:00 ET on the holiday
    Verified against 5 weekends of Bitget hourly candles and Labor Day 2026
    (native from Fri 04 Sep 20:00 ET until Mon 07 Sep 20:00 ET)."""
    session_day = (ny + timedelta(hours=4)).date()
    return session_day.weekday() >= 5 or session_day in US_MARKET_HOLIDAYS


def score_weekend_afterhours(timestamp: datetime) -> float:
    """
    Small, deliberate penalty (not a hard drop) reflecting that Bitget's
    rToken liquidity works differently outside regular NASDAQ/NYSE hours -
    a different mechanism, not automatically a worse one.
      1.0  regular hours: 09:30-16:00 America/New_York (DST-aware), Mon-Fri
      0.8  weekday outside regular hours (pre/after-market, overnight)
      0.6  weekend window (Fri 20:00 ET -> Sun 20:00 ET) or market holiday
           (20:00 ET the evening before -> 20:00 ET on the holiday)
    Half-day early closes are not modelled.
    """
    ny = _to_new_york(timestamp)
    minutes = ny.hour * 60 + ny.minute
    if _is_native_window(ny):
        return 0.6
    if 9 * 60 + 30 <= minutes < 16 * 60:
        return 1.0
    return 0.8


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


def score_stock(data: dict, history: dict, now: datetime | None = None,
                 fresh_signal: bool = True) -> dict:
    """
    Convenience wrapper: scores one stock given its latest fetch
    entry and its own historical series.
    data: single entry from fetch_bitget_data() (one stock)
    history: {"price": [...], "depth": [...], "ts": [...]} -
    IMPORTANT: this must be the baseline BEFORE this run's values are
    added (the caller is responsible for scoring first, then updating
    history afterward - see main.py's run()). Passing a history that
    already includes the current reading silently biases depth toward
    ~1.0, since the current value would be part of its own baseline.
    """
    now = now or datetime.now(timezone.utc)
    # fresh_signal=False means the ticker snapshot is byte-identical to the
    # previous cycle (quiet or frozen feed - indistinguishable). The
    # history-based components then say "no signal" instead of "healthy".
    if fresh_signal:
        depth = score_depth(data.get("bid_size"), data.get("ask_size"),
                            history.get("depth", []), history.get("ts"), now)
        abnormal = score_abnormal_movement(history.get("price", []), data.get("price"),
                                           history.get("ts"), now)
    else:
        depth = abnormal = None
    components = {
        "spread": score_spread(data.get("spread_pct")),
        "depth": depth,
        "abnormal_movement": abnormal,
        "weekend": score_weekend_afterhours(now),
    }
    result = calculate_composite_score(components)
    # A partial score is useful for observability, but it must not be
    # presented as fully healthy when the action gate has not matured.
    if len(result.get("components_used", [])) < MIN_COMPONENTS:
        result["label"] = "warming_up" if fresh_signal else "no_fresh_signal"
    else:
        result["label"] = classify_score(result["score"])
    return result


if __name__ == "__main__":
    import json

    # Fabricated example: healthy token with enough PAST history.
    # Note: fake_history contains only PAST values - the current
    # reading lives in fake_data, kept separate (see score_stock's
    # docstring on why history must not include the current run).
    fake_data = {"price": 224.75, "spread_pct": 0.07, "bid_size": 65, "ask_size": 140}
    fake_history = {
        "price": [223.5, 224.1, 224.3, 224.5],
        "depth": [180, 190, 200],
    }
    print(json.dumps(score_stock(fake_data, fake_history), indent=2))

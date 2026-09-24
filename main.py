"""
Main orchestrator for Custos - single-issuer (Bitget) rToken
structural risk monitoring with active hold/sell/buy-back decisions.

Run on a schedule (every 4 hours via GitHub Actions; real spacing is
jittery, 3-8h). Each run:
  1. Pulls fresh data for all 10 rTokens from Bitget
  2. Scores each token (Health Score) against its PRE-update history
  3. Updates rolling, TIMESTAMPED history (price, depth, scores; volume is kept
     only to detect a frozen/unchanged snapshot - it is not scored)
  4. Logs a heartbeat entry (always)
  5. Among HELD tokens, finds the lowest-scoring one. Deterministic
     gates run first (score >= SELL_CEILING -> HOLD, no LLM call; score
     must stay below the ceiling for CONFIRM_RUNS consecutive runs;
     minimum hold time after entry). Only then is the LLM consulted.
  6. Among SOLD tokens (holding pooled USDT), mirrored logic for
     BUY BACK vs WAIT (floor, confirmation, cooldown after the sell).
  7. Persists state FIRST (atomically), then appends immutable logs.
  8. A stock is only actionable with enough components AND a mature
     baseline (>= MIN_HISTORY_POINTS points spanning >= MIN_HISTORY_HOURS).

State files (all under data/):
  history.json             - schema v2: per-stock price/depth/ts/score series (+ volume for
                             frozen-snapshot detection only)
  positions.json           - "USDT" pooled cash + per-stock held/sold state
  heartbeat_log.jsonl      - one line per run, all 10 scores + fetch status
  event_log.jsonl          - narrative decision log
  transaction_log.jsonl    - standard format required by the form
  performance_log.jsonl    - one entry per EXIT (sell) - see roundtrip_log
  roundtrip_log.jsonl      - one entry per completed sell -> buy-back round trip
  performance_summary.json - exits + round-trip metrics, drawdown
  raw_ticker_log.json      - last RAW_TICKER_LOG_RUNS runs of raw volume/size
                             fields, to verify what volume24h really means
  latest.json              - snapshot of the most recent run (dashboard)
"""

import copy
import json
import math
import os
import re
from datetime import datetime, timezone
from statistics import mean, stdev

from fetch_bitget import fetch_bitget_data
from health_score import score_stock, HISTORY_WINDOW
from llm_client import call_llm

DATA_DIR = "data"
HISTORY_PATH = f"{DATA_DIR}/history.json"
POSITIONS_PATH = f"{DATA_DIR}/positions.json"
HEARTBEAT_LOG_PATH = f"{DATA_DIR}/heartbeat_log.jsonl"
EVENT_LOG_PATH = f"{DATA_DIR}/event_log.jsonl"
TRANSACTION_LOG_PATH = f"{DATA_DIR}/transaction_log.jsonl"
PERFORMANCE_LOG_PATH = f"{DATA_DIR}/performance_log.jsonl"
ROUNDTRIP_LOG_PATH = f"{DATA_DIR}/roundtrip_log.jsonl"
PERFORMANCE_SUMMARY_PATH = f"{DATA_DIR}/performance_summary.json"
LATEST_PATH = f"{DATA_DIR}/latest.json"
RAW_TICKER_LOG_PATH = f"{DATA_DIR}/raw_ticker_log.json"
RECENT_EVENTS_PATH = os.path.join(DATA_DIR, "recent_events.json")
RECENT_TRANSACTIONS_PATH = os.path.join(DATA_DIR, "recent_transactions.json")

STOCKS = ["NVDA", "TSLA", "AAPL", "AMZN", "GOOGL", "SPY", "QQQ", "KO", "MCD", "PYPL"]

SCHEMA_VERSION = 2
INITIAL_HOLDING_USD = 300.0  # per stock, $3,000 total portfolio
TRADING_FEE_PCT = 0.0005  # 0.05% maker/taker, matching Bitget's disclosed rToken promo rate

# Hard rules (code-enforced, cannot be overridden by the LLM):
SELL_CEILING = 0.5       # never sell if score >= this
BUYBACK_FLOOR = 0.7      # never buy back if score < this
SELL_SEVERE = 0.25       # below this, the structural case for selling is strong
BUYBACK_STRONG = 0.9     # above this, the case for buying back is strong

# A token needs ALL 4 Health Score components available before it's eligible
# for an ACTUAL action (a missing component means "no reliable signal").
MIN_COMPONENTS_FOR_ACTION = 4
# AND its historical baseline must be mature: enough points AND enough
# elapsed time. Real run spacing is 3-8h (avg ~5.4h), so a point count
# alone says little about how much calendar time the baseline covers.
MIN_HISTORY_POINTS = 12
MIN_HISTORY_HOURS = 48.0
ALL_COMPONENTS = {"spread", "depth", "abnormal_movement", "weekend"}

# Anti-whipsaw (audit M-4): the score is noisy, so a single reading below
# the ceiling / above the floor is not enough to act on.
CONFIRM_RUNS = 2              # consecutive runs the condition must hold (incl. this one)
MIN_HOLD_HOURS = 12.0         # no SELL within this long of (re-)entering
BUYBACK_COOLDOWN_HOURS = 12.0 # no BUY_BACK within this long of the SELL
SCORE_HISTORY_LEN = 6

LLM_REASON_MAX_CHARS = 400
RAW_TICKER_LOG_RUNS = 30


# ── Small I/O helpers ──────────────────────────────────────────

class StateCorruptionError(RuntimeError):
    """Raised when an existing state file cannot be trusted."""


def load_json(path: str, default, *, required: bool = False):
    """Load JSON safely. Missing files may use a default; existing corrupt
    state files must fail closed instead of silently resetting portfolio state."""
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        message = f"failed to load existing state file {path}: {e}"
        if required:
            raise StateCorruptionError(message) from e
        print(f"[warn] {message} - using default")
        return default


def save_json(path: str, data):
    """Atomic write: temp file + os.replace, so a crash mid-write can
    never leave a half-written JSON file behind."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)


def append_jsonl(path: str, entry: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def read_jsonl(path: str) -> list[dict]:
    """Read JSONL robustly. Skip blank lines and any corrupt JSON lines
    so a single partial write can never crash the whole run."""
    if not os.path.exists(path):
        return []
    entries = []
    with open(path, "r") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[warn] skipping corrupt JSONL line {line_no} in {path}: {e}")
    return entries


# ── Startup reconciliation (ledger vs. positions.json) ─────────

def count_ledger_entries(transaction_path: str = TRANSACTION_LOG_PATH) -> tuple[dict, dict]:
    """Counts sells and buy-backs per stock from transaction_log.jsonl,
    the one ledger that records both directions with a consistent,
    purpose-built shape (direction: SELL/BUY). Pure read - separated
    from find_ledger_mismatches so both are unit-testable.

    M-3 fix: this used to count sells from performance_log.jsonl and
    buys from event_log.jsonl - two different files with independent
    corruption tolerance (read_jsonl silently skips bad lines in
    each) and no shared identity between them, so one corrupt or
    duplicated line in either file could desync the counts without
    the other file reflecting it, producing a false permanent freeze.
    transaction_log.jsonl already existed with exactly the fields
    needed for this and a dedicated "direction" field, but nothing
    read it until now. Entries are deduplicated by event_id, since a
    manual workflow re-run can reproduce the same event_id (event_id
    only has second-resolution - see the L-1 note on build_sell_event/
    build_buyback_event) and would otherwise be double-counted.
    """
    sells, buys = {}, {}
    seen_ids = set()
    for t in read_jsonl(transaction_path):
        event_id = t.get("event_id")
        if event_id is not None:
            if event_id in seen_ids:
                continue
            seen_ids.add(event_id)
        inst = t.get("instrument")
        direction = t.get("direction")
        if not inst or direction not in ("SELL", "BUY"):
            continue
        target = sells if direction == "SELL" else buys
        target[inst] = target.get(inst, 0) + 1
    return sells, buys


def find_ledger_mismatches(positions: dict, sells: dict, buys: dict) -> dict[str, str]:
    """Every SELL must leave the stock 'sold' until a BUY_BACK returns
    it to 'held'. If the ledgers and positions.json disagree, the state
    is inconsistent (e.g. a past crash between logging and saving) -
    return a per-stock explanation instead of acting on it."""
    problems = {}
    for stock in STOCKS:
        if stock not in positions:
            continue
        expected_sold = sells.get(stock, 0) - buys.get(stock, 0)
        actual_sold = 1 if positions[stock].get("status") == "sold" else 0
        if expected_sold != actual_sold:
            problems[stock] = (
                f"ledger mismatch: {sells.get(stock, 0)} sell(s) / "
                f"{buys.get(stock, 0)} buy-back(s) recorded, but "
                f"positions.json says status={positions[stock].get('status')}"
            )
    return problems


# ── Action eligibility ─────────────────────────────────────────

def _parse_iso(value) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def hours_since(value, now: datetime) -> float | None:
    parsed = _parse_iso(value)
    return None if parsed is None else (now - parsed).total_seconds() / 3600.0


def history_span_hours(series: dict) -> float | None:
    """Calendar time covered by the price baseline (first -> last valid
    timestamp). None when the series carries no usable timestamps."""
    stamps = [_parse_iso(t) for t in series.get("ts", [])]
    stamps = [t for t in stamps if t is not None]
    if len(stamps) < 2:
        return None
    return (max(stamps) - min(stamps)).total_seconds() / 3600.0


def check_actionable(stock: str, score_result: dict, history: dict) -> tuple[bool, str | None]:
    """A stock may only produce an ACTUAL trade if it has (a) enough
    score components AND (b) a mature baseline: enough points AND enough
    elapsed calendar time. Returns (eligible, reason_if_not)."""
    n_components = len(score_result.get("components_used", []))
    if n_components < MIN_COMPONENTS_FOR_ACTION:
        return False, (f"only {n_components}/{len(ALL_COMPONENTS)} components available, need "
                       f"{MIN_COMPONENTS_FOR_ACTION}")
    series = history.get(stock, {})
    hist_points = len(series.get("price", []))
    if hist_points < MIN_HISTORY_POINTS:
        return False, (f"history still warming up - {hist_points}/"
                       f"{MIN_HISTORY_POINTS} baseline points")
    span = history_span_hours(series)
    if span is None or span < MIN_HISTORY_HOURS:
        shown = "no timestamps" if span is None else f"{span:.0f}h"
        return False, (f"history still warming up - baseline spans {shown}, "
                       f"need {MIN_HISTORY_HOURS:.0f}h")
    return True, None


def check_confirmation(stock: str, score: float, history_before: dict,
                       predicate, description: str) -> tuple[bool, str | None]:
    """Anti-whipsaw gate: `predicate` (e.g. score < SELL_CEILING) must
    hold for this run AND the previous CONFIRM_RUNS-1 recorded runs."""
    if not predicate(score):
        return False, description
    needed = CONFIRM_RUNS - 1
    if needed <= 0:
        return True, None
    previous = history_before.get(stock, {}).get("score", [])[-needed:]
    if len(previous) < needed or any(s is None or not predicate(s) for s in previous):
        return False, (f"awaiting confirmation - {description} must hold for "
                       f"{CONFIRM_RUNS} consecutive runs")
    return True, None


# ── State bootstrapping ────────────────────────────────────────

def validate_history_state(history: dict) -> None:
    """Fail closed if history.json is syntactically valid but structurally unusable."""
    if not isinstance(history, dict):
        raise StateCorruptionError("history.json has invalid top-level schema")
    for stock, series in history.items():
        if stock not in STOCKS:
            continue
        if not isinstance(series, dict):
            raise StateCorruptionError(f"history.json has invalid series for {stock}")
        for key in ("volume", "price", "depth"):
            values = series.get(key, [])
            if not isinstance(values, list):
                raise StateCorruptionError(f"history.json {stock}.{key} must be a list")
            for value in values:
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(float(value)) or value < 0):
                    raise StateCorruptionError(f"history.json {stock}.{key} contains invalid numeric data")
        # schema v2 series (optional so legacy files still load)
        stamps = series.get("ts", [])
        if not isinstance(stamps, list) or any(t is not None and not isinstance(t, str) for t in stamps):
            raise StateCorruptionError(f"history.json {stock}.ts must be a list of ISO strings/null")
        score_series = series.get("score", [])
        if not isinstance(score_series, list) or any(
                s is not None and (isinstance(s, bool) or not isinstance(s, (int, float))
                                   or not math.isfinite(float(s)) or not 0 <= s <= 1)
                for s in score_series):
            raise StateCorruptionError(f"history.json {stock}.score must be a list of 0..1 numbers/null")


def validate_positions_state(positions: dict) -> None:
    """Fail closed if an existing portfolio state has the wrong shape."""
    if not isinstance(positions, dict) or not isinstance(positions.get("USDT"), dict):
        raise StateCorruptionError("positions.json has invalid top-level schema")
    balance = positions["USDT"].get("balance_usdt")
    if not isinstance(balance, (int, float)) or not math.isfinite(float(balance)) or balance < 0:
        raise StateCorruptionError("positions.json has invalid USDT balance")
    for stock in STOCKS:
        p = positions.get(stock)
        if not isinstance(p, dict) or p.get("status") not in {"held", "sold"}:
            raise StateCorruptionError(f"positions.json has invalid state for {stock}")
        if p.get("status") == "held":
            qty = p.get("quantity")
            cost = p.get("cost_basis_usd")
            if qty is None or not isinstance(qty, (int, float)) or not math.isfinite(float(qty)) or qty <= 0:
                raise StateCorruptionError(f"positions.json has invalid held quantity for {stock}")
            if cost is None or not isinstance(cost, (int, float)) or not math.isfinite(float(cost)) or cost <= 0:
                raise StateCorruptionError(f"positions.json has invalid cost basis for {stock}")


def bootstrap_is_complete(by_stock: dict) -> list[str]:
    """H-1 fix: returns the list of stocks that don't yet have a
    usable price (no 'ok' fetch, or no ask/price on the entry).
    Bootstrapping with any of these missing would write positions.json
    with quantity: None for a "held" stock - a shape that passes
    init_positions() silently but then fails validate_positions_state()
    on every subsequent run, since nothing ever revisits or repairs a
    "held" position after bootstrap. That's a permanent, unrecoverable
    crash loop from a single bad first run, not a one-cycle hiccup -
    so bootstrap must not proceed until this list is empty.
    """
    missing = []
    for stock in STOCKS:
        entry = by_stock.get(stock, {})
        if entry.get("status") != "ok" or not (entry.get("ask") or entry.get("price")):
            missing.append(stock)
    return missing


def init_positions(by_stock: dict) -> dict:
    """
    First-run bootstrap. USDT starts as its own explicit position at
    $0. Each stock enters HELD using the ASK price (the executable
    buy-side price, not lastPrice) minus the trading fee. Falls back
    to lastPrice if ask is unavailable, rather than failing the entry.

    Callers MUST check bootstrap_is_complete(by_stock) == [] first -
    this function assumes every stock already has a usable price and
    no longer tolerates a partial fetch (see H-1).
    """
    positions = {"USDT": {"balance_usdt": 0.0}}
    for stock in STOCKS:
        entry = by_stock.get(stock, {})
        price = entry.get("ask") or entry.get("price")
        quantity = (INITIAL_HOLDING_USD * (1 - TRADING_FEE_PCT) / price) if price else None
        positions[stock] = {
            "status": "held", "entry_price": price, "exit_price": None,
            "quantity": quantity, "cost_basis_usd": INITIAL_HOLDING_USD,
        }
    return positions


# ── History maintenance ─────────────────────────────────────────

def _empty_series() -> dict:
    return {"volume": [], "price": [], "depth": [], "ts": [], "score": []}


def migrate_history(history: dict, heartbeats: list[dict] | None = None) -> dict:
    """Bring a loaded history.json up to schema v2 (idempotent).
    - guarantees all series keys exist
    - aligns `ts` 1:1 with `price`; legacy points without timestamps are
      back-filled from heartbeat_log.jsonl when it can be matched
      exactly (a heartbeat with a non-null score for the stock == a run
      in which that stock's price was appended), otherwise left None.
    """
    heartbeats = heartbeats or []
    for stock in STOCKS:
        series = history.get(stock)
        if series is None:
            continue
        for key, default in _empty_series().items():
            series.setdefault(key, default)
        n = len(series["price"])
        ts = list(series["ts"])
        if len(ts) != n:
            if not ts:
                stamps = [h["timestamp"] for h in heartbeats
                          if isinstance(h.get("scores"), dict)
                          and h["scores"].get(stock) is not None
                          and h.get("timestamp")]
                ts = stamps[-n:] if n and len(stamps) >= n else [None] * n
            elif len(ts) < n:
                ts = [None] * (n - len(ts)) + ts
            else:
                ts = ts[-n:]
        series["ts"] = ts
    history["schema_version"] = SCHEMA_VERSION
    return history


def baseline_summary(history: dict) -> dict:
    """Per-stock dashboard summary: baseline maturity (how many price points
    exist and how many calendar hours they span, None when unknown) plus the
    last few composite scores (already capped at SCORE_HISTORY_LEN in
    history[stock]['score']) for a small trend indicator. A None entry in
    the trend means that run's score wasn't built from enough components."""
    out = {}
    for stock in STOCKS:
        series = history.get(stock) or {}
        span = history_span_hours(series)
        out[stock] = {"points": len(series.get("price", [])),
                      "span_hours": None if span is None else round(span, 1),
                      "trend": list(series.get("score", []))}
    return out


def is_unchanged_snapshot(series: dict, entry: dict) -> bool:
    """True when the ticker's price, volume and top-of-book size are all
    identical to the last stored reading: a quiet OR frozen feed (they
    cannot be told apart). Such a snapshot carries no fresh signal."""
    try:
        if entry.get("bid_size") is None or entry.get("ask_size") is None:
            return False
        if entry.get("volume_24h") is None or entry.get("price") is None:
            return False
        return (series["price"][-1] == entry["price"]
                and series["volume"][-1] == entry["volume_24h"]
                and series["depth"][-1] == entry["bid_size"] + entry["ask_size"])
    except (KeyError, IndexError, TypeError):
        return False


def update_history(history: dict, by_stock: dict, now: datetime | None = None,
                   skip: set | None = None) -> dict:
    """Rolling windows (HISTORY_WINDOW points) of price/depth (and volume,
    used only for frozen-snapshot detection) per stock, with a timestamp
    per price point. Stocks in `skip` (unchanged
    snapshots) are not appended, so duplicates never shrink the volatility
    estimate or masquerade as fresh baseline points."""
    now = now or datetime.now(timezone.utc)
    skip = skip or set()
    for stock, entry in by_stock.items():
        if entry.get("status") != "ok" or stock in skip:
            continue
        series = history.setdefault(stock, _empty_series())
        for key, default in _empty_series().items():
            series.setdefault(key, default)

        if entry.get("volume_24h") is not None:
            series["volume"].append(entry["volume_24h"])
            series["volume"] = series["volume"][-HISTORY_WINDOW:]

        if entry.get("price") is not None:
            series["price"].append(entry["price"])
            series["price"] = series["price"][-HISTORY_WINDOW:]
            series["ts"].append(entry.get("timestamp") or now.isoformat())
            series["ts"] = series["ts"][-HISTORY_WINDOW:]

        if entry.get("bid_size") is not None and entry.get("ask_size") is not None:
            series["depth"].append(entry["bid_size"] + entry["ask_size"])
            series["depth"] = series["depth"][-HISTORY_WINDOW:]

    return history


def record_scores(history: dict, scores: dict) -> dict:
    """Append this run's composite score per stock (None when the score
    was not built from >= MIN_COMPONENTS_FOR_ACTION components, so a
    partial spread+calendar score can never count as a confirmation)."""
    for stock in STOCKS:
        result = scores.get(stock, {})
        usable = (result.get("score") is not None
                  and len(result.get("components_used", [])) >= MIN_COMPONENTS_FOR_ACTION)
        series = history.setdefault(stock, _empty_series())
        series.setdefault("score", [])
        series["score"].append(result["score"] if usable else None)
        series["score"] = series["score"][-SCORE_HISTORY_LEN:]
    return history


def record_raw_tickers(by_stock: dict, now: datetime) -> None:
    """Keep the last RAW_TICKER_LOG_RUNS runs of raw bid/ask, size and
    volume fields for later analysis (Bitget offers no historical
    ticker/order-book API)."""
    log = load_json(RAW_TICKER_LOG_PATH, [])
    if not isinstance(log, list):
        log = []
    tickers = {s: e["raw_diagnostics"] for s, e in by_stock.items()
               if e.get("status") == "ok" and e.get("raw_diagnostics")}
    if tickers:
        log.append({"timestamp": now.isoformat(), "tickers": tickers})
    save_json(RAW_TICKER_LOG_PATH, log[-RAW_TICKER_LOG_RUNS:])


# ── LLM decision helpers ────────────────────────────────────────

_SCORE_SEMANTICS = (
    "IMPORTANT - how to read the numbers: EVERY component value and the composite "
    "are HEALTH scores from 0.0 to 1.0, where 1.0 = healthy and 0.0 = unhealthy. "
    "A HIGH value is GOOD for every component, never a warning: a high 'spread' "
    "value means a TIGHT spread, a high 'abnormal_movement' value means a calm, "
    "normal price move, a high 'weekend' value means regular market hours. A "
    "component that is missing had no reliable signal this cycle - do not treat "
    "it as either good or bad.\n"
)


def _raw_metrics_line(market: dict | None) -> str:
    if not market:
        return ""
    parts = []
    if market.get("spread_pct") is not None:
        parts.append(f"raw bid/ask spread {market['spread_pct']}% of mid")
    if market.get("bid_size") is not None and market.get("ask_size") is not None:
        parts.append(f"top-of-book size {market['bid_size'] + market['ask_size']:.4g}")
    return ("Raw context (for reference only): " + "; ".join(parts) + "\n") if parts else ""


def build_sell_hold_prompt(stock: str, score_result: dict, market: dict | None = None) -> str:
    score = score_result["score"]
    return (
        f"You are a structural risk monitor for a tokenized stock (rToken) on Bitget. "
        f"You do NOT predict price direction and you do NOT evaluate the underlying "
        f"company as an investment - your only job is judging the health of the "
        f"WRAPPER's liquidity mechanics (spread, order-book depth, abnormal "
        f"price-move magnitude relative to this token's own history).\n\n"
        f"{_SCORE_SEMANTICS}\n"
        f"Token: {stock}\n"
        f"Current composite health score: {score} (label: {score_result['label']})\n"
        f"Component health scores: {score_result['components_raw']}\n"
        f"{_raw_metrics_line(market)}\n"
        f"Hard rules (already enforced by the system, not your call):\n"
        f"- Score >= {SELL_CEILING}: SELL is never executed regardless of what you say.\n"
        f"- Score < {SELL_SEVERE}: structural stress is severe - lean strongly toward SELL "
        f"unless you have a specific, concrete reason to think the reading is a temporary "
        f"artifact (e.g. a known scheduled event) rather than genuine stress.\n"
        f"- Between {SELL_SEVERE} and {SELL_CEILING}: this is genuinely ambiguous. Use your "
        f"own judgment - does this pattern of components look like durable structural "
        f"degradation, or could it plausibly be an explainable short-term artifact? Your "
        f"reasoning here actually decides the outcome, not a formula.\n\n"
        f"Reply with ONLY a JSON object and nothing else (no markdown, no preamble, no "
        f"step-by-step thinking): "
        f'{{"decision": "SELL" or "HOLD", "reason": "at most 2 short sentences citing the '
        f'specific components"}}. Do not discuss whether {stock} is a good investment - '
        f"only the wrapper's structural condition."
    )


def build_buyback_wait_prompt(stock: str, score_result: dict, market: dict | None = None) -> str:
    score = score_result["score"]
    return (
        f"You are a structural risk monitor for a tokenized stock (rToken) on Bitget. "
        f"This token was previously sold due to structural risk and the proceeds are "
        f"currently held as USDT. You do NOT predict price direction or evaluate the "
        f"underlying company - only the wrapper's liquidity mechanics.\n\n"
        f"{_SCORE_SEMANTICS}\n"
        f"Token: {stock}\n"
        f"Current composite health score: {score} (label: {score_result['label']})\n"
        f"Component health scores: {score_result['components_raw']}\n"
        f"{_raw_metrics_line(market)}\n"
        f"Hard rules (already enforced by the system, not your call):\n"
        f"- Score < {BUYBACK_FLOOR}: BUY BACK is never executed regardless of what you say.\n"
        f"- Score >= {BUYBACK_STRONG}: recovery looks strong - lean toward BUY BACK unless "
        f"something in the components still looks fragile.\n"
        f"- Between {BUYBACK_FLOOR} and {BUYBACK_STRONG}: genuinely ambiguous - use your "
        f"judgment on whether the recovery looks durable or premature. Your reasoning "
        f"actually decides the outcome here.\n\n"
        f"Reply with ONLY a JSON object and nothing else (no markdown, no preamble, no "
        f"step-by-step thinking): "
        f'{{"decision": "BUY_BACK" or "WAIT", "reason": "at most 2 short sentences citing '
        f'the specific components"}}.'
    )


_DECISION_STRIP = "*_`\"'.:,;()[]{}—–- "


def parse_decision(llm_text, valid_words: set[str]) -> str | None:
    """
    Legacy/fallback parser: decision from the FIRST TOKEN of the first
    line, EXACT match against valid_words (never a substring search:
    "HOLD - SELL is not justified" must stay HOLD). Returns None
    (unparseable) for anything else - the caller treats None as "no
    action", the safer failure mode.

    Hardening (audit M-7): non-string input returns None instead of
    raising; trailing commas/brackets are stripped ("SELL, spread
    widened"); a leading "Decision:" label is skipped; "BUY BACK" is
    accepted as BUY_BACK.
    """
    if not isinstance(llm_text, str) or not llm_text.strip():
        return None
    first_line = llm_text.strip().splitlines()[0].strip()
    tokens = first_line.split()
    if not tokens:
        return None
    cleaned = [t.strip(_DECISION_STRIP).upper() for t in tokens[:3]]
    if cleaned[0] in {"DECISION", "ANSWER", "VERDICT"} and len(cleaned) > 1:
        cleaned = cleaned[1:]
    word = cleaned[0]
    if word == "BUY" and len(cleaned) > 1 and cleaned[1] == "BACK":
        word = "BUY_BACK"
    return word if word in valid_words else None


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_JSON_OBJECT = re.compile(r"\{.*?\}", re.DOTALL)


def _clip(text, limit: int = LLM_REASON_MAX_CHARS) -> str | None:
    if not isinstance(text, str):
        return None
    text = " ".join(text.split())
    if not text:
        return None
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def parse_llm_decision(llm_text, valid_words: set[str]) -> tuple[str | None, str | None]:
    """Returns (decision, reason). Preferred format is the JSON object the
    prompts ask for; falls back to parse_decision() on the first line.
    Never raises. `reason` is capped at LLM_REASON_MAX_CHARS - raw
    chain-of-thought is never stored/published (audit M-5)."""
    if not isinstance(llm_text, str) or not llm_text.strip():
        return None, None
    text = _THINK_BLOCK.sub("", llm_text).strip()
    if text.lower().startswith("<think>"):
        return None, None          # unterminated reasoning block: no decision reached
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()

    candidates = [text] + _JSON_OBJECT.findall(text)
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict) and isinstance(obj.get("decision"), str):
            word = obj["decision"].strip().upper().replace(" ", "_").replace("-", "_")
            if word in valid_words:
                return word, _clip(obj.get("reason"))

    decision = parse_decision(text, valid_words)
    if decision is None:
        return None, None
    lines = text.splitlines()
    rest = " ".join(lines[1:]) if len(lines) > 1 else lines[0]
    return decision, _clip(rest)


def make_validator(valid_words: set[str]):
    """call_llm() validator: a reply is usable only if it yields a decision."""
    return lambda content: parse_llm_decision(content, valid_words)[0] is not None


# ── Event builders (pure - logging happens in persist_logs) ─────
# These functions never touch disk. run() collects the events + ledger
# entries in memory, saves state first, and only then flushes them to
# disk via persist_logs().

def _event_id(prefix: str, stock: str, now: datetime) -> str:
    # Microsecond resolution: two events for one instrument can no longer
    # collide on the same second (audit L-6).
    return f"{prefix}_{stock}_{now.strftime('%Y%m%dT%H%M%S%f')}"


def build_sell_event(stock: str, entry_price: float | None, exit_price: float | None,
                     quantity: float | None, proceeds: float, realized_pnl: float,
                     score_result: dict, reasoning: str | None, llm_meta: dict, now: datetime,
                     cost_basis: float | None = None) -> tuple[dict, list]:
    event_id = _event_id("sell", stock, now)
    transaction_entry = {
        "event_id": event_id, "timestamp": now.isoformat(), "instrument": stock,
        "direction": "SELL", "price": exit_price, "quantity": quantity,
        "balance_change": round(proceeds, 2),
    }
    # Percentage against the actual cost basis (what realized_pnl is
    # measured against); fall back to entry_price * quantity.
    notional = cost_basis or ((entry_price * quantity) if (entry_price and quantity) else None)
    pnl_pct = round((realized_pnl / notional) * 100, 4) if notional else None
    decision_zone = "severe" if score_result["score"] < SELL_SEVERE else "grey_zone"
    performance_entry = {
        "event_id": event_id, "timestamp": now.isoformat(), "instrument": stock,
        "entry_price": entry_price, "exit_price": exit_price, "quantity": quantity,
        "realized_pnl_usd": realized_pnl, "realized_pnl_pct": pnl_pct,
        "score_at_decision": score_result["score"], "decision_zone": decision_zone,
    }
    event = {
        "event_id": event_id, "timestamp": now.isoformat(), "type": "sell",
        "instrument": stock, "entry_price": entry_price, "exit_price": exit_price,
        "proceeds_usd": round(proceeds, 2), "realized_pnl_usd": realized_pnl,
        "score_at_decision": score_result["score"], "decision_zone": decision_zone,
        "llm_reasoning": reasoning, "llm_provider_used": llm_meta.get("provider_used"),
        "llm_attempts": llm_meta.get("attempts"),
    }
    return event, [(TRANSACTION_LOG_PATH, transaction_entry),
                   (PERFORMANCE_LOG_PATH, performance_entry)]


def build_roundtrip_entry(stock: str, sold_position: dict, rebuy_price: float,
                          qty_after: float, now: datetime, event_id: str) -> dict | None:
    """A completed sell -> buy-back cycle, measured against the
    counterfactual of simply HOLDING through it (audit M-1).
    protection_gain_usd > 0  <=> selling and re-entering left us with
    more units (after both fees) than never selling would have."""
    qty_before = sold_position.get("quantity_sold")
    exit_price = sold_position.get("exit_price")
    if not qty_before or not exit_price or not rebuy_price:
        return None
    hold_value = qty_before * rebuy_price
    gain = (qty_after - qty_before) * rebuy_price
    return {
        "event_id": event_id, "timestamp": now.isoformat(), "instrument": stock,
        "sold_timestamp": sold_position.get("sold_timestamp"),
        "hours_out_of_position": (round(hours_since(sold_position.get("sold_timestamp"), now), 2)
                                  if hours_since(sold_position.get("sold_timestamp"), now) is not None else None),
        "exit_price": exit_price, "rebuy_price": rebuy_price,
        "price_change_pct_since_exit": round((rebuy_price - exit_price) / exit_price * 100, 4),
        "quantity_before_sell": qty_before, "quantity_after_rebuy": qty_after,
        "protection_gain_usd": round(gain, 4),
        "protection_gain_pct": round(gain / hold_value * 100, 4) if hold_value else None,
        "realized_pnl_at_exit_usd": sold_position.get("realized_pnl_usd"),
    }


def build_buyback_event(stock: str, price: float | None, quantity: float | None,
                        cost: float, score_result: dict, reasoning: str | None,
                        llm_meta: dict, now: datetime,
                        sold_position: dict | None = None) -> tuple[dict, list]:
    event_id = _event_id("buy", stock, now)
    transaction_entry = {
        "event_id": event_id, "timestamp": now.isoformat(), "instrument": stock,
        "direction": "BUY", "price": price, "quantity": quantity,
        "balance_change": round(-cost, 2),
    }
    decision_zone = "strong" if score_result["score"] >= BUYBACK_STRONG else "grey_zone"
    event = {
        "event_id": event_id, "timestamp": now.isoformat(), "type": "buy_back",
        "instrument": stock, "price": price, "cost_usd": round(cost, 2),
        "score_at_decision": score_result["score"], "decision_zone": decision_zone,
        "llm_reasoning": reasoning, "llm_provider_used": llm_meta.get("provider_used"),
        "llm_attempts": llm_meta.get("attempts"),
    }
    ledger = [(TRANSACTION_LOG_PATH, transaction_entry)]
    if sold_position is not None and price and quantity:
        trip = build_roundtrip_entry(stock, sold_position, price, quantity, now, event_id)
        if trip is not None:
            event["protection_gain_usd"] = trip["protection_gain_usd"]
            ledger.append((ROUNDTRIP_LOG_PATH, trip))
    return event, ledger


def build_eval_event(stock: str, decision_type: str, action_taken: str,
                     score_result: dict, reasoning: str | None, llm_meta: dict,
                     now: datetime) -> tuple[dict, list]:
    """Builds an event for an evaluation that did NOT result in a
    transaction (HOLD/WAIT/warming-up/skipped)."""
    event = {
        "event_id": _event_id("eval", stock, now),
        "timestamp": now.isoformat(), "type": decision_type, "instrument": stock,
        "action_taken": action_taken, "score": score_result.get("score"),
        "score_label": score_result.get("label"),
        "llm_reasoning": reasoning, "llm_provider_used": llm_meta.get("provider_used"),
        "llm_attempts": llm_meta.get("attempts"),
    }
    return event, []


def persist_logs(event: dict, ledger_entries: list) -> None:
    """Flushes ledger entries (transaction/performance/round-trip logs)
    BEFORE the event log, and only ever called after state is durable."""
    for path, entry in ledger_entries:
        append_jsonl(path, entry)
    append_jsonl(EVENT_LOG_PATH, event)


# ── Portfolio + performance summaries ───────────────────────────

def compute_portfolio_summary(positions: dict, by_stock: dict) -> dict:
    """
    Reports BOTH cost-basis totals (what's actually been transacted -
    the basis for realized P&L) and current mark-to-market totals.
    Mark-to-market is net of the exit fee that would actually be paid
    on liquidation, so the headline number is realizable, not gross.
    """
    usdt = positions["USDT"]["balance_usdt"]
    held_cost_basis, held_market_value = 0.0, 0.0
    held_stocks, sold_stocks = [], []
    valuation_is_stale = False

    for stock in STOCKS:
        p = positions[stock]
        if p["status"] == "held":
            held_stocks.append(stock)
            held_cost_basis += p["cost_basis_usd"]
            current_price = by_stock.get(stock, {}).get("price")
            if current_price and p.get("quantity"):
                # Net of the exit fee that selling would incur, so this
                # isn't a rosier number than the position could actually realize.
                held_market_value += p["quantity"] * current_price * (1 - TRADING_FEE_PCT)
            else:
                held_market_value += p["cost_basis_usd"]
                valuation_is_stale = True
        else:
            sold_stocks.append(stock)

    return {
        "total_usdt_cash": round(usdt, 2),
        "total_held_cost_basis_usd": round(held_cost_basis, 2),
        "total_held_market_value_usd": round(held_market_value, 2),
        "total_portfolio_value_cost_basis_usd": round(usdt + held_cost_basis, 2),
        "total_portfolio_value_market_usd": round(usdt + held_market_value, 2),
        "valuation_status": "stale" if valuation_is_stale else "live",
        "held_stocks": held_stocks,
        "sold_stocks_holding_cash": sold_stocks,
    }


def _dedupe_by_event_id(entries: list[dict]) -> list[dict]:
    seen, out = set(), []
    for e in entries:
        eid = e.get("event_id")
        if eid is not None:
            if eid in seen:
                continue
            seen.add(eid)
        out.append(e)
    return out


def compute_performance_metrics() -> dict:
    """
    Recomputed each run from the ledgers (deduplicated by event_id).

    Two DIFFERENT things are reported and must not be confused:
      * exit_* / trade_count: one row per SELL, P&L of the sell price
        against the ORIGINAL entry. Does NOT include what re-entering
        cost. (`trade_count` is kept as an alias of `exit_count`.)
      * round_trip_*: one row per completed sell -> buy-back cycle,
        measured against simply holding through it (protection gain).
    """
    exits = _dedupe_by_event_id(read_jsonl(PERFORMANCE_LOG_PATH))
    trips = _dedupe_by_event_id(read_jsonl(ROUNDTRIP_LOG_PATH))
    if not exits and not trips:
        return {"trade_count": 0, "exit_count": 0, "round_trips_completed": 0,
                "note": "no completed trades yet"}

    pnls = [t["realized_pnl_usd"] for t in exits if t.get("realized_pnl_usd") is not None]
    pcts = [t["realized_pnl_pct"] for t in exits if t.get("realized_pnl_pct") is not None]
    wins = [p for p in pnls if p > 0]

    running, peak, max_dd = 0.0, 0.0, 0.0
    for p in pnls:
        running += p
        peak = max(peak, running)
        max_dd = max(max_dd, peak - running)

    sharpe_like = None
    if len(pcts) >= 2:
        s = stdev(pcts)
        sharpe_like = round(mean(pcts) / s, 4) if s else None

    gains = [t["protection_gain_usd"] for t in trips if t.get("protection_gain_usd") is not None]
    gain_pcts = [t["protection_gain_pct"] for t in trips if t.get("protection_gain_pct") is not None]

    return {
        "exit_count": len(exits),
        "trade_count": len(exits),  # alias kept for older dashboards
        "open_exits": max(0, len(exits) - len(trips)),
        "total_realized_pnl_usd": round(sum(pnls), 2) if pnls else 0.0,
        "win_rate_pct": round(len(wins) / len(pnls) * 100, 2) if pnls else None,
        "avg_pnl_pct_per_trade": round(mean(pcts), 4) if pcts else None,
        "sharpe_like_ratio": sharpe_like,
        "max_drawdown_usd": round(max_dd, 2) if pnls else None,
        "round_trips_completed": len(trips),
        "round_trip_win_rate_pct": (round(len([g for g in gains if g > 0]) / len(gains) * 100, 2)
                                    if gains else None),
        "total_protection_gain_usd": round(sum(gains), 4) if gains else 0.0,
        "avg_protection_gain_pct": round(mean(gain_pcts), 4) if gain_pcts else None,
        "note": ("exit_* compares the sell price with the original entry and ignores re-entry cost; "
                 "round_trip_* compares selling + re-buying against simply holding. "
                 "Small sample - interpret with caution."),
    }


# ── Main run ──────────────────────────────────────────────────

def read_jsonl_tail(path: str, n: int) -> list[dict]:
    """Last n valid JSONL entries without parsing the whole file."""
    from collections import deque
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        tail = deque((ln for ln in f if ln.strip()), maxlen=n * 2)
    entries = []
    for line in tail:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries[-n:]


NO_LLM = {"provider_used": None, "attempts": []}


def run():
    now = datetime.now(timezone.utc)

    history = load_json(HISTORY_PATH, {}, required=True)
    validate_history_state(history)
    history = migrate_history(history, read_jsonl(HEARTBEAT_LOG_PATH))
    validate_history_state(history)

    raw_entries = fetch_bitget_data()
    by_stock = {e["underlying"]: e for e in raw_entries}

    positions = load_json(POSITIONS_PATH, None, required=True)
    if positions is None:
        # Never bootstrap on a partial fetch (a "held" stock with
        # quantity None would crash-loop validate_positions_state()).
        missing = bootstrap_is_complete(by_stock)
        if missing:
            append_jsonl(HEARTBEAT_LOG_PATH, {
                "timestamp": now.isoformat(),
                "status": "bootstrap_incomplete",
                "reason": f"waiting for a full fetch before first-run bootstrap - "
                          f"missing usable price for: {', '.join(missing)}",
            })
            print(f"[{now.isoformat()}] Bootstrap deferred - missing price for: "
                  f"{', '.join(missing)}. No state written; will retry next cycle.")
            return
        positions = init_positions(by_stock)
        save_json(POSITIONS_PATH, positions)
    else:
        validate_positions_state(positions)

    # Safety net: ledgers and positions.json must agree. Freeze any stock
    # where they don't instead of trading on inconsistent state.
    sells, buys = count_ledger_entries()
    mismatches = find_ledger_mismatches(positions, sells, buys)
    frozen = set(mismatches)
    for stock, reason in mismatches.items():
        append_jsonl(EVENT_LOG_PATH, {
            "event_id": _event_id("reconcile", stock, now),
            "timestamp": now.isoformat(), "type": "state_reconciliation_warning",
            "instrument": stock, "action_taken": "FROZEN (manual review required)",
            "warning": reason, "llm_reasoning": None,
            "llm_provider_used": None, "llm_attempts": [],
        })
        print(f"  !! {stock}: {reason} - frozen for this run")

    scores = {}
    heartbeat_scores, heartbeat_labels, fetch_status, unchanged = {}, {}, {}, []
    for stock in STOCKS:
        entry = by_stock.get(stock, {"status": "error"})
        fetch_status[stock] = entry.get("status", "missing")
        # Score against the PRE-update history (this run's values are not
        # yet included).
        stock_history = history.get(stock, _empty_series())
        is_ok = entry.get("status") == "ok"
        stale_snapshot = is_ok and is_unchanged_snapshot(stock_history, entry)
        if stale_snapshot:
            unchanged.append(stock)
        result = (score_stock(entry, stock_history, now, fresh_signal=not stale_snapshot)
                  if is_ok else
                  {"score": None, "label": "unknown", "components_raw": {}, "components_used": []})
        scores[stock] = result
        heartbeat_scores[stock] = result["score"]
        heartbeat_labels[stock] = result["label"]

    # Snapshot history BEFORE folding in this run's values: the maturity
    # gate and the confirmation gate must not count this run's own reading.
    history_before_update = copy.deepcopy(history)

    history = update_history(history, by_stock, now, skip=set(unchanged))
    history = record_scores(history, scores)

    append_jsonl(HEARTBEAT_LOG_PATH, {
        "timestamp": now.isoformat(), "scores": heartbeat_scores, "labels": heartbeat_labels,
        "fetch_status": fetch_status, "unchanged_snapshot": unchanged,
    })

    events_this_run = []   # flushed after state is saved
    logs_this_run = []

    def emit(event, ledger_entries):
        events_this_run.append(event)
        logs_this_run.append((event, ledger_entries))

    # Observability: a held/sold stock that could not be evaluated because
    # its market data fetch failed is logged, never silently skipped.
    for stock in STOCKS:
        if by_stock.get(stock, {}).get("status") != "ok" and stock in positions:
            entry_status = by_stock.get(stock, {}).get("status", "missing")
            if positions[stock].get("status") in ("held", "sold"):
                event, ledger = build_eval_event(
                    stock, "evaluation_skipped",
                    f"SKIPPED (market data unavailable - fetch status: {entry_status})",
                    {"score": None, "label": "unknown"}, None, NO_LLM, now,
                )
                logs_this_run.append((event, ledger))

    # Candidate lists are fixed BEFORE any trade this run mutates
    # positions, so a token sold now is not immediately re-evaluated for
    # buy-back in the same run.
    held_candidates = [s for s in STOCKS
                       if positions[s]["status"] == "held"
                       and scores[s]["score"] is not None and s not in frozen]
    sold_candidates = [s for s in STOCKS
                       if positions[s]["status"] == "sold"
                       and scores[s]["score"] is not None and s not in frozen]

    # --- Evaluate the lowest-scoring HELD stock ---
    if held_candidates:
        worst = min(held_candidates, key=lambda s: scores[s]["score"])
        score_result = scores[worst]
        score = score_result["score"]
        eligible, reason = check_actionable(worst, score_result, history_before_update)
        confirmed, confirm_reason = (check_confirmation(
            worst, score, history_before_update, lambda s: s < SELL_CEILING,
            f"score below SELL_CEILING {SELL_CEILING}") if eligible else (False, None))
        held_hours = hours_since(positions[worst].get("entered_timestamp"), now)

        gate_action = None
        if not eligible:
            gate_action = f"HOLD (warming up - {reason})"
        elif score >= SELL_CEILING:
            gate_action = (f"HOLD (blocked by hard rule - score {score} >= SELL_CEILING "
                           f"{SELL_CEILING}; LLM not consulted)")
        elif not confirmed:
            gate_action = f"HOLD ({confirm_reason})"
        elif held_hours is not None and held_hours < MIN_HOLD_HOURS:
            gate_action = (f"HOLD (minimum hold - re-entered {held_hours:.1f}h ago, "
                           f"need {MIN_HOLD_HOURS:.0f}h)")

        if gate_action is not None:
            event, ledger = build_eval_event(worst, "sell_hold_evaluation", gate_action,
                                             score_result, None, NO_LLM, now)
            emit(event, ledger)
        else:
            valid = {"SELL", "HOLD"}
            llm_result = call_llm(build_sell_hold_prompt(worst, score_result, by_stock[worst]),
                                  validator=make_validator(valid))
            decision, reasoning = parse_llm_decision(llm_result.get("content"), valid)
            decided_sell = decision == "SELL" and score < SELL_CEILING

            if decided_sell:
                price = by_stock[worst].get("bid") or by_stock[worst].get("price")
                qty = positions[worst].get("quantity")
                entry_price = positions[worst].get("entry_price")
                cost_basis = positions[worst].get("cost_basis_usd")

                # Never mutate state with missing executable data.
                if not qty or not price or cost_basis is None:
                    event, ledger = build_eval_event(
                        worst, "sell_hold_evaluation",
                        "HOLD (missing quantity/price/cost_basis - cannot execute safely)",
                        score_result, reasoning, llm_result, now)
                    emit(event, ledger)
                else:
                    gross_proceeds = qty * price
                    proceeds = gross_proceeds - gross_proceeds * TRADING_FEE_PCT
                    realized_pnl = round(proceeds - cost_basis, 2)

                    positions["USDT"]["balance_usdt"] = round(
                        positions["USDT"]["balance_usdt"] + proceeds, 2)
                    positions[worst] = {
                        "status": "sold", "entry_price": entry_price, "exit_price": price,
                        "quantity_sold": qty, "proceeds_usd": round(proceeds, 2),
                        "realized_pnl_usd": realized_pnl, "sold_timestamp": now.isoformat(),
                    }
                    event, ledger = build_sell_event(
                        worst, entry_price, price, qty, proceeds, realized_pnl,
                        score_result, reasoning, llm_result, now, cost_basis=cost_basis)
                    emit(event, ledger)
            else:
                if not llm_result.get("success"):
                    action = "HOLD (no usable LLM reply - fail-safe default, not an evaluated decision)"
                elif decision is None:
                    action = "HOLD (response unparseable - fail-safe default, not a confirmed decision)"
                else:
                    action = "HOLD"
                event, ledger = build_eval_event(worst, "sell_hold_evaluation", action,
                                                 score_result, reasoning, llm_result, now)
                emit(event, ledger)

    # --- Evaluate the highest-scoring SOLD stock ---
    if sold_candidates:
        best = max(sold_candidates, key=lambda s: scores[s]["score"])
        score_result = scores[best]
        score = score_result["score"]
        eligible, reason = check_actionable(best, score_result, history_before_update)
        confirmed, confirm_reason = (check_confirmation(
            best, score, history_before_update, lambda s: s >= BUYBACK_FLOOR,
            f"score at/above BUYBACK_FLOOR {BUYBACK_FLOOR}") if eligible else (False, None))
        out_hours = hours_since(positions[best].get("sold_timestamp"), now)

        gate_action = None
        if not eligible:
            gate_action = f"WAIT (warming up - {reason})"
        elif score < BUYBACK_FLOOR:
            gate_action = (f"WAIT (blocked by hard rule - score {score} < BUYBACK_FLOOR "
                           f"{BUYBACK_FLOOR}; LLM not consulted)")
        elif not confirmed:
            gate_action = f"WAIT ({confirm_reason})"
        elif out_hours is not None and out_hours < BUYBACK_COOLDOWN_HOURS:
            gate_action = (f"WAIT (cooldown - sold {out_hours:.1f}h ago, "
                           f"need {BUYBACK_COOLDOWN_HOURS:.0f}h)")

        if gate_action is not None:
            event, ledger = build_eval_event(best, "buyback_wait_evaluation", gate_action,
                                             score_result, None, NO_LLM, now)
            emit(event, ledger)
        else:
            valid = {"BUY_BACK", "WAIT"}
            llm_result = call_llm(build_buyback_wait_prompt(best, score_result, by_stock[best]),
                                  validator=make_validator(valid))
            decision, reasoning = parse_llm_decision(llm_result.get("content"), valid)
            decided_buy = decision == "BUY_BACK" and score >= BUYBACK_FLOOR

            if decided_buy:
                price = by_stock[best].get("ask") or by_stock[best].get("price")
                sold_position = positions[best]
                available_cash = positions["USDT"].get("balance_usdt") or 0.0
                # Re-buy with THIS token's own sale proceeds (not a flat
                # $300 from the shared pool) - no idle cash dust, and no
                # token silently funded by another token's proceeds.
                own_proceeds = sold_position.get("proceeds_usd") or INITIAL_HOLDING_USD
                target_notional = min(own_proceeds, available_cash)

                if target_notional <= 0 or not price or price <= 0:
                    why = ("insufficient USDT cash" if target_notional <= 0
                           else "missing or invalid ask/price")
                    event, ledger = build_eval_event(
                        best, "buyback_wait_evaluation", f"WAIT ({why})", score_result,
                        reasoning, llm_result, now)
                    emit(event, ledger)
                else:
                    qty = (target_notional * (1 - TRADING_FEE_PCT)) / price
                    positions["USDT"]["balance_usdt"] = round(available_cash - target_notional, 2)
                    positions[best] = {
                        "status": "held", "entry_price": price, "exit_price": None,
                        "quantity": qty, "cost_basis_usd": round(target_notional, 2),
                        "entered_timestamp": now.isoformat(),
                    }
                    event, ledger = build_buyback_event(
                        best, price, qty, target_notional,
                        score_result, reasoning, llm_result, now,
                        sold_position=sold_position)
                    emit(event, ledger)
            else:
                if not llm_result.get("success"):
                    action = "WAIT (no usable LLM reply - fail-safe default, not an evaluated decision)"
                elif decision is None:
                    action = "WAIT (response unparseable - fail-safe default, not a confirmed decision)"
                else:
                    action = "WAIT"
                event, ledger = build_eval_event(best, "buyback_wait_evaluation", action,
                                                 score_result, reasoning, llm_result, now)
                emit(event, ledger)

    # ── Persistence ordering ──
    # In CI the durable boundary is the git commit at the end of the
    # workflow (it only runs if this script exits 0); within a run,
    # state is saved first, atomically, then the immutable ledgers.
    save_json(HISTORY_PATH, history)
    save_json(POSITIONS_PATH, positions)
    for event, ledger_entries in logs_this_run:
        persist_logs(event, ledger_entries)
    try:  # diagnostics must never be able to fail a run
        record_raw_tickers(by_stock, now)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[warn] raw ticker diagnostics not saved: {type(exc).__name__}")
    performance_summary = compute_performance_metrics()
    save_json(PERFORMANCE_SUMMARY_PATH, performance_summary)
    save_json(LATEST_PATH, {
        "timestamp": now.isoformat(), "schema_version": SCHEMA_VERSION,
        "scores": scores, "positions": positions, "baseline": baseline_summary(history),
        "portfolio_summary": compute_portfolio_summary(positions, by_stock),
        "performance_summary": performance_summary,
        "events_this_run": events_this_run,
    })
    # Small dashboard feeds so the frontend never downloads the full logs.
    save_json(RECENT_EVENTS_PATH, list(reversed(read_jsonl_tail(EVENT_LOG_PATH, 20))))
    save_json(RECENT_TRANSACTIONS_PATH, list(reversed(read_jsonl_tail(TRANSACTION_LOG_PATH, 20))))

    print(f"[{now.isoformat()}] Run complete. {len(events_this_run)} evaluation(s).")
    for event in events_this_run:
        print(f"  - {event.get('instrument')}: {event.get('type')} -> "
              f"{event.get('action_taken', event.get('type'))}")


if __name__ == "__main__":
    run()

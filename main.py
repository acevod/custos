"""
Main orchestrator for Custos - single-issuer (Bitget) rToken
structural risk monitoring with active hold/sell/buy-back decisions.

Run on a schedule (every 4 hours via GitHub Actions). Each run:
  1. Pulls fresh data for all 10 rTokens from Bitget
  2. Updates rolling history (volume, price, order-book depth)
  3. Computes a composite Health Score per token
  4. Logs a heartbeat entry (always)
  5. Among HELD tokens, finds the lowest-scoring one and ALWAYS asks
     the LLM to evaluate SELL vs HOLD. Score >= SELL_CEILING is a
     hard rule against selling (code-enforced, can't be overridden).
  6. Among SOLD tokens (holding pooled USDT), finds the
     highest-scoring one and ALWAYS asks the LLM to evaluate
     BUY BACK vs WAIT, with the mirrored grey-zone logic.
  7. Persists state FIRST (atomically), then appends immutable logs.
     (H1 fix: state is durable before any ledger entry exists, so a
     crash can never leave a phantom trade in the performance log
     that positions.json doesn't know about. A startup reconciliation
     pass catches any pre-existing divergence and freezes the
     affected stock instead of acting on inconsistent state.)
  8. A stock is only actionable once it has BOTH enough score
     components AND enough mature history (H2 fix) - 4/5 components
     AND >= MIN_HISTORY_POINTS baseline points. Early baselines of
     3 samples are too noisy to trade on.

State files (all under data/):
  history.json           - rolling volume/price/depth history per stock
  positions.json          - "USDT" pooled cash + per-stock held/sold state
  heartbeat_log.jsonl      - one line per run, all 10 scores
  event_log.jsonl          - narrative decision log (every LLM call)
  transaction_log.jsonl    - standard format required by the form
  performance_log.jsonl    - one entry per completed round-trip trade
  performance_summary.json - win-rate, realized P&L, Sharpe-like, drawdown
  latest.json              - snapshot of the most recent run, for the
                             dashboard to read
"""

import copy
import json
import math
import os
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
PERFORMANCE_SUMMARY_PATH = f"{DATA_DIR}/performance_summary.json"
LATEST_PATH = f"{DATA_DIR}/latest.json"
RECENT_EVENTS_PATH = os.path.join(DATA_DIR, "recent_events.json")

STOCKS = ["NVDA", "TSLA", "AAPL", "AMZN", "GOOGL", "SPY", "QQQ", "KO", "MCD", "PYPL"]

INITIAL_HOLDING_USD = 300.0  # per stock, $3,000 total portfolio
TRADING_FEE_PCT = 0.0005  # 0.05% maker/taker, matching Bitget's disclosed rToken promo rate

# Hard rules (code-enforced, cannot be overridden by the LLM):
SELL_CEILING = 0.5       # never sell if score >= this
BUYBACK_FLOOR = 0.7      # never buy back if score < this
SELL_SEVERE = 0.25       # below this, the structural case for selling is strong
BUYBACK_STRONG = 0.9     # above this, the case for buying back is strong

# A token needs at least this many of the 5 Health Score components
# available before it's eligible for an ACTUAL action.
MIN_COMPONENTS_FOR_ACTION = 4
# H2 fix: AND its historical baseline must be mature. Each historical
# component activates at just 3 samples, which is far too noisy to
# trade on. Require a baseline spanning at least ~2 days (12 points at
# the 4-hour cadence). Raise toward HISTORY_WINDOW (42) for stricter
# maturity.
MIN_HISTORY_POINTS = 12
ALL_COMPONENTS = {"spread", "depth", "volume_trend", "abnormal_movement", "weekend"}


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
    never leave a half-written JSON file behind (H1 fix)."""
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


# ── Startup reconciliation (H1 safety net) ─────────────────────

def count_ledger_entries(performance_path: str = PERFORMANCE_LOG_PATH,
                          event_path: str = EVENT_LOG_PATH) -> tuple[dict, dict]:
    """Counts sells per stock (performance log) and buy-backs per stock
    (event log). Pure read - separated from find_ledger_mismatches so
    both are unit-testable."""
    sells, buys = {}, {}
    for t in read_jsonl(performance_path):
        inst = t.get("instrument")
        if inst:
            sells[inst] = sells.get(inst, 0) + 1
    for e in read_jsonl(event_path):
        if e.get("type") == "buy_back" and e.get("instrument"):
            inst = e["instrument"]
            buys[inst] = buys.get(inst, 0) + 1
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


# ── Action eligibility (H2 fix) ────────────────────────────────

def check_actionable(stock: str, score_result: dict, history: dict) -> tuple[bool, str | None]:
    """A stock may only produce an ACTUAL trade if it has (a) enough
    score components AND (b) a mature enough historical baseline.
    Returns (eligible, reason_if_not)."""
    n_components = len(score_result.get("components_used", []))
    if n_components < MIN_COMPONENTS_FOR_ACTION:
        return False, (f"only {n_components}/5 components available, need "
                       f"{MIN_COMPONENTS_FOR_ACTION}")
    hist_points = len(history.get(stock, {}).get("price", []))
    if hist_points < MIN_HISTORY_POINTS:
        return False, (f"history still warming up - {hist_points}/"
                       f"{MIN_HISTORY_POINTS} baseline points")
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
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
                    raise StateCorruptionError(f"history.json {stock}.{key} contains invalid numeric data")


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


def init_positions(by_stock: dict) -> dict:
    """
    First-run bootstrap. USDT starts as its own explicit position at
    $0. Each stock enters HELD using the ASK price (the executable
    buy-side price, not lastPrice) minus the trading fee. Falls back
    to lastPrice if ask is unavailable, rather than failing the entry.
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

def update_history(history: dict, by_stock: dict) -> dict:
    """Rolling windows (HISTORY_WINDOW points) of volume/price/depth per stock."""
    for stock, entry in by_stock.items():
        if entry.get("status") != "ok":
            continue
        series = history.setdefault(stock, {"volume": [], "price": [], "depth": []})

        if entry.get("volume_24h") is not None:
            series["volume"].append(entry["volume_24h"])
            series["volume"] = series["volume"][-HISTORY_WINDOW:]

        if entry.get("price") is not None:
            series["price"].append(entry["price"])
            series["price"] = series["price"][-HISTORY_WINDOW:]

        if entry.get("bid_size") is not None and entry.get("ask_size") is not None:
            series["depth"].append(entry["bid_size"] + entry["ask_size"])
            series["depth"] = series["depth"][-HISTORY_WINDOW:]

    return history


# ── LLM decision helpers ────────────────────────────────────────

def build_sell_hold_prompt(stock: str, score_result: dict) -> str:
    score = score_result["score"]
    return (
        f"You are a structural risk monitor for a tokenized stock (rToken) on Bitget. "
        f"You do NOT predict price direction and you do NOT evaluate the underlying "
        f"company as an investment - your only job is judging the health of the "
        f"WRAPPER's liquidity mechanics (spread, order-book depth, volume, abnormal "
        f"price-move magnitude relative to this token's own history).\n\n"
        f"Token: {stock}\n"
        f"Current composite health score: {score} (label: {score_result['label']})\n"
        f"Component breakdown: {score_result['components_raw']}\n\n"
        f"Hard rules (already enforced by the system, not your call):\n"
        f"- Score >= {SELL_CEILING}: SELL is never executed regardless of what you say.\n"
        f"- Score < {SELL_SEVERE}: structural stress is severe - lean strongly toward SELL "
        f"unless you have a specific, concrete reason to think the reading is a temporary "
        f"artifact (e.g. a known scheduled event) rather than genuine stress.\n"
        f"- Between {SELL_SEVERE} and {SELL_CEILING}: this is genuinely ambiguous. Use your "
        f"own judgment - does this pattern of components look like durable structural "
        f"degradation, or could it plausibly be an explainable short-term artifact? Your "
        f"reasoning here actually decides the outcome, not a formula.\n\n"
        f"Respond with your decision (SELL or HOLD) on the first line, then a 2-3 sentence "
        f"justification referencing the specific components and, if relevant, any context "
        f"about why this reading might be genuine or an artifact. Do not discuss whether "
        f"{stock} is a good investment - only the wrapper's structural condition."
    )


def build_buyback_wait_prompt(stock: str, score_result: dict) -> str:
    score = score_result["score"]
    return (
        f"You are a structural risk monitor for a tokenized stock (rToken) on Bitget. "
        f"This token was previously sold due to structural risk and the proceeds are "
        f"currently held as USDT. You do NOT predict price direction or evaluate the "
        f"underlying company - only the wrapper's liquidity mechanics.\n\n"
        f"Token: {stock}\n"
        f"Current composite health score: {score} (label: {score_result['label']})\n"
        f"Component breakdown: {score_result['components_raw']}\n\n"
        f"Hard rules (already enforced by the system, not your call):\n"
        f"- Score < {BUYBACK_FLOOR}: BUY BACK is never executed regardless of what you say.\n"
        f"- Score >= {BUYBACK_STRONG}: recovery looks strong - lean toward BUY BACK unless "
        f"something in the components still looks fragile.\n"
        f"- Between {BUYBACK_FLOOR} and {BUYBACK_STRONG}: genuinely ambiguous - use your "
        f"judgment on whether the recovery looks durable or premature. Your reasoning "
        f"actually decides the outcome here.\n\n"
        f"Respond with your decision (BUY_BACK or WAIT) on the first line, then a 2-3 "
        f"sentence justification referencing the specific components."
    )


def parse_decision(llm_text: str | None, valid_words: set[str]) -> str | None:
    """
    Strictly parses the LLM's decision from the first line of its
    reply. Requires an EXACT match (after stripping whitespace and
    common markdown/punctuation wrapping) against one of valid_words.
    Deliberately NOT a substring search: a naive `"SELL" in first_line`
    check would incorrectly match "HOLD - SELL is not justified",
    silently turning a HOLD into a SELL. Returns None (unparseable)
    for anything that isn't an exact match - the caller treats None
    the same as "no action", the safer failure mode.
    """
    # H-1 fix: guard on the STRIPPED text, not the raw text. A
    # whitespace-only response (e.g. "   " or "\n\n") is truthy and
    # would previously slip past `if not llm_text`, then crash with
    # IndexError on splitlines()[0] once it's stripped to "".
    if not llm_text or not llm_text.strip():
        return None
    first_line = llm_text.strip().splitlines()[0]
    cleaned = first_line.strip().strip("*_`\"'.:—- ").upper()
    return cleaned if cleaned in valid_words else None


# ── Event builders (pure - logging happens in persist_logs) ─────
# H1 fix: these functions no longer touch disk. run() collects the
# events + ledger entries in memory, saves state first, and only then
# flushes them to disk via persist_logs(). A crash therefore can never
# produce a ledger entry whose state was never saved.

def build_sell_event(stock: str, entry_price: float | None, exit_price: float | None,
                     quantity: float | None, proceeds: float, realized_pnl: float,
                     score_result: dict, reasoning: str, llm_meta: dict, now: datetime) -> tuple[dict, list]:
    event_id = f"sell_{stock}_{now.strftime('%Y%m%dT%H%M%S')}"
    transaction_entry = {
        "event_id": event_id, "timestamp": now.isoformat(), "instrument": stock,
        "direction": "SELL", "price": exit_price, "quantity": quantity,
        "balance_change": round(proceeds, 2),
    }
    # Prefer notional from entry_price * quantity; guard zero/None to
    # avoid ZeroDivisionError or nonsense percentages.
    notional = (entry_price * quantity) if (entry_price and quantity) else None
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


def build_buyback_event(stock: str, price: float | None, quantity: float | None,
                        cost: float, score_result: dict, reasoning: str,
                        llm_meta: dict, now: datetime) -> tuple[dict, list]:
    event_id = f"buy_{stock}_{now.strftime('%Y%m%dT%H%M%S')}"
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
    return event, [(TRANSACTION_LOG_PATH, transaction_entry)]


def build_eval_event(stock: str, decision_type: str, action_taken: str,
                     score_result: dict, reasoning: str | None, llm_meta: dict,
                     now: datetime) -> tuple[dict, list]:
    """Builds an event for an LLM evaluation that did NOT result in a
    transaction (HOLD/WAIT/warming-up/skipped)."""
    event = {
        "event_id": f"eval_{stock}_{now.strftime('%Y%m%dT%H%M%S')}",
        "timestamp": now.isoformat(), "type": decision_type, "instrument": stock,
        "action_taken": action_taken, "score": score_result.get("score"),
        "score_label": score_result.get("label"),
        "llm_reasoning": reasoning, "llm_provider_used": llm_meta.get("provider_used"),
        "llm_attempts": llm_meta.get("attempts"),
    }
    return event, []


def persist_logs(event: dict, ledger_entries: list) -> None:
    """H1 fix: flushes ledger entries (transaction/performance logs)
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
                # L4 fix: net of the exit fee that selling would incur
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


def compute_performance_metrics() -> dict:
    """
    Recomputed each run from performance_log.jsonl (every completed
    round-trip trade). Small sample sizes are flagged explicitly
    rather than presented as statistically robust.
    """
    trades = read_jsonl(PERFORMANCE_LOG_PATH)
    if not trades:
        return {"trade_count": 0, "note": "no completed trades yet"}

    pnls = [t["realized_pnl_usd"] for t in trades if t.get("realized_pnl_usd") is not None]
    pcts = [t["realized_pnl_pct"] for t in trades if t.get("realized_pnl_pct") is not None]
    wins = [p for p in pnls if p > 0]

    cum, running, peak, max_dd = [], 0.0, 0.0, 0.0
    for p in pnls:
        running += p
        peak = max(peak, running)
        max_dd = max(max_dd, peak - running)
        cum.append(running)

    sharpe_like = None
    if len(pcts) >= 2:
        s = stdev(pcts)
        sharpe_like = round(mean(pcts) / s, 4) if s else None

    return {
        "trade_count": len(trades),
        "total_realized_pnl_usd": round(sum(pnls), 2) if pnls else 0.0,
        "win_rate_pct": round(len(wins) / len(pnls) * 100, 2) if pnls else None,
        "avg_pnl_pct_per_trade": round(mean(pcts), 4) if pcts else None,
        "sharpe_like_ratio": sharpe_like,
        "max_drawdown_usd": round(max_dd, 2) if pnls else None,
        "note": "small sample size - interpret with caution" if len(trades) < 10 else None,
    }


# ── Main run ──────────────────────────────────────────────────

def run():
    now = datetime.now(timezone.utc)

    history = load_json(HISTORY_PATH, {}, required=True)
    validate_history_state(history)

    raw_entries = fetch_bitget_data()
    by_stock = {e["underlying"]: e for e in raw_entries}

    positions = load_json(POSITIONS_PATH, None, required=True)
    if positions is None:
        positions = init_positions(by_stock)
        # Bootstrap is allowed only when the file truly does not exist.
        save_json(POSITIONS_PATH, positions)
    else:
        validate_positions_state(positions)

    # H1 safety net: if a previous run died between writing ledger
    # entries and saving state, the ledgers and positions.json will
    # disagree. Freeze the affected stocks and surface a loud warning
    # instead of trading on inconsistent state.
    sells, buys = count_ledger_entries()
    mismatches = find_ledger_mismatches(positions, sells, buys)
    frozen = set(mismatches)
    for stock, reason in mismatches.items():
        append_jsonl(EVENT_LOG_PATH, {
            "event_id": f"reconcile_{stock}_{now.strftime('%Y%m%dT%H%M%S')}",
            "timestamp": now.isoformat(), "type": "state_reconciliation_warning",
            "instrument": stock, "action_taken": "FROZEN (manual review required)",
            "warning": reason, "llm_reasoning": None,
            "llm_provider_used": None, "llm_attempts": [],
        })
        print(f"  !! {stock}: {reason} - frozen for this run")

    scores = {}
    heartbeat_scores, heartbeat_labels = {}, {}
    for stock in STOCKS:
        entry = by_stock.get(stock, {"status": "error"})
        # Score against the PRE-update history (this run's values are
        # not yet included).
        stock_history = history.get(stock, {"volume": [], "price": [], "depth": []})
        result = score_stock(entry, stock_history) if entry.get("status") == "ok" else \
            {"score": None, "label": "unknown", "components_raw": {}, "components_used": []}
        scores[stock] = result
        heartbeat_scores[stock] = result["score"]
        heartbeat_labels[stock] = result["label"]

    # M-1 fix: snapshot history BEFORE folding in this run's values,
    # and use this snapshot for check_actionable()'s maturity gate
    # below. Without this, check_actionable would count this run's
    # own just-appended reading as one of its MIN_HISTORY_POINTS,
    # opening the action gate one cycle earlier than the H2 fix
    # intended (mirrors the same "don't let a component see its own
    # current reading" principle score_stock already follows).
    history_before_update = copy.deepcopy(history)

    # Only now fold this run's values into history, so the NEXT run's
    # baseline includes them - never this run's own scoring.
    history = update_history(history, by_stock)

    append_jsonl(HEARTBEAT_LOG_PATH, {
        "timestamp": now.isoformat(), "scores": heartbeat_scores, "labels": heartbeat_labels,
    })

    events_this_run = []   # (event, ledger_entries) - flushed after state is saved
    logs_this_run = []

    def emit(event, ledger_entries):
        events_this_run.append(event)
        logs_this_run.append((event, ledger_entries))

    # M3 observability: explicitly log when a held/sold stock could
    # not be evaluated because its market data fetch failed - a
    # skipped evaluation is no longer silent.
    for stock in STOCKS:
        if by_stock.get(stock, {}).get("status") != "ok" and stock in positions:
            entry_status = by_stock.get(stock, {}).get("status", "missing")
            if positions[stock].get("status") in ("held", "sold"):
                event, ledger = build_eval_event(
                    stock, "evaluation_skipped",
                    f"SKIPPED (market data unavailable - fetch status: {entry_status})",
                    {"score": None, "label": "unknown"}, None,
                    {"provider_used": None, "attempts": []}, now,
                )
                logs_this_run.append((event, ledger))

    # --- Evaluate the lowest-scoring HELD stock ---
    held = [s for s in STOCKS
            if positions[s]["status"] == "held"
            and scores[s]["score"] is not None
            and s not in frozen]
    if held:
        worst_stock = min(held, key=lambda s: scores[s]["score"])
        score_result = scores[worst_stock]
        eligible, reason = check_actionable(worst_stock, score_result, history_before_update)

        if not eligible:
            # Not enough signal/maturity yet - skip the LLM call
            # entirely rather than acting on thin evidence.
            event, ledger = build_eval_event(
                worst_stock, "sell_hold_evaluation",
                f"HOLD (warming up - {reason})",
                score_result, None, {"provider_used": None, "attempts": []}, now,
            )
            emit(event, ledger)
        else:
            prompt = build_sell_hold_prompt(worst_stock, score_result)
            llm_result = call_llm(prompt)
            decision = parse_decision(llm_result.get("content"), {"SELL", "HOLD"})
            decided_sell = decision == "SELL" and score_result["score"] < SELL_CEILING

            if decided_sell:
                price = by_stock[worst_stock].get("bid") or by_stock[worst_stock].get("price")
                qty = positions[worst_stock].get("quantity")
                entry_price = positions[worst_stock].get("entry_price")
                cost_basis = positions[worst_stock].get("cost_basis_usd")

                # Guard: never mutate state with missing executable data.
                # A None quantity or price can occur after a partial
                # bootstrap (fetch failure on first run). Treat as
                # non-actionable rather than writing corrupt "sold" state.
                if not qty or not price or cost_basis is None:
                    event, ledger = build_eval_event(
                        worst_stock, "sell_hold_evaluation",
                        "HOLD (missing quantity/price/cost_basis - cannot execute safely)",
                        score_result, llm_result.get("content"), llm_result, now)
                    emit(event, ledger)
                else:
                    gross_proceeds = qty * price
                    fee = gross_proceeds * TRADING_FEE_PCT
                    proceeds = gross_proceeds - fee
                    realized_pnl = round(proceeds - cost_basis, 2)

                    positions["USDT"]["balance_usdt"] = round(
                        positions["USDT"]["balance_usdt"] + proceeds, 2)
                    positions[worst_stock] = {
                        "status": "sold", "entry_price": entry_price, "exit_price": price,
                        "quantity_sold": qty, "proceeds_usd": round(proceeds, 2),
                        "realized_pnl_usd": realized_pnl, "sold_timestamp": now.isoformat(),
                    }
                    event, ledger = build_sell_event(
                        worst_stock, entry_price, price, qty, proceeds, realized_pnl,
                        score_result, llm_result.get("content"), llm_result, now)
                    emit(event, ledger)
            else:
                # M-2 fix: distinguish a genuine LLM HOLD from a SELL
                # that the LLM recommended but the hard rule blocked -
                # these previously looked identical in the log
                # ("HOLD"), hiding a meaningful disagreement between
                # the model's judgment and the code-enforced ceiling.
                if not llm_result.get("success"):
                    action = "HOLD (LLM unavailable - fail-safe default, not an evaluated decision)"
                elif decision is None:
                    action = "HOLD (response unparseable - fail-safe default, not a confirmed decision)"
                elif decision == "SELL":
                    action = (f"HOLD (LLM recommended SELL, blocked by hard rule - "
                              f"score {score_result['score']} >= SELL_CEILING {SELL_CEILING})")
                else:
                    action = "HOLD"
                event, ledger = build_eval_event(worst_stock, "sell_hold_evaluation", action,
                                                 score_result, llm_result.get("content"),
                                                 llm_result, now)
                emit(event, ledger)

    # --- Evaluate the highest-scoring SOLD stock ---
    sold = [s for s in STOCKS
            if positions[s]["status"] == "sold"
            and scores[s]["score"] is not None
            and s not in frozen]
    if sold:
        best_stock = max(sold, key=lambda s: scores[s]["score"])
        score_result = scores[best_stock]
        eligible, reason = check_actionable(best_stock, score_result, history_before_update)

        if not eligible:
            event, ledger = build_eval_event(
                best_stock, "buyback_wait_evaluation",
                f"WAIT (warming up - {reason})",
                score_result, None, {"provider_used": None, "attempts": []}, now,
            )
            emit(event, ledger)
        else:
            prompt = build_buyback_wait_prompt(best_stock, score_result)
            llm_result = call_llm(prompt)
            decision = parse_decision(llm_result.get("content"), {"BUY_BACK", "WAIT"})
            decided_buy = decision == "BUY_BACK" and score_result["score"] >= BUYBACK_FLOOR

            if decided_buy:
                price = by_stock[best_stock].get("ask") or by_stock[best_stock].get("price")
                available_cash = positions["USDT"].get("balance_usdt") or 0.0
                target_notional = min(INITIAL_HOLDING_USD, available_cash)

                # Guard: require positive executable price and enough cash.
                if target_notional <= 0 or not price or price <= 0:
                    reason = ("insufficient USDT cash" if target_notional <= 0
                              else "missing or invalid ask/price")
                    event, ledger = build_eval_event(
                        best_stock, "buyback_wait_evaluation",
                        f"WAIT ({reason})", score_result,
                        llm_result.get("content"), llm_result, now)
                    emit(event, ledger)
                else:
                    qty = (target_notional * (1 - TRADING_FEE_PCT)) / price
                    positions["USDT"]["balance_usdt"] = round(available_cash - target_notional, 2)
                    positions[best_stock] = {
                        "status": "held", "entry_price": price, "exit_price": None,
                        "quantity": qty, "cost_basis_usd": round(target_notional, 2),
                    }
                    event, ledger = build_buyback_event(
                        best_stock, price, qty, target_notional,
                        score_result, llm_result.get("content"), llm_result, now)
                    emit(event, ledger)
            else:
                # M-2 fix: same distinction as the sell/hold branch -
                # a BUY_BACK recommendation blocked by the hard floor
                # is not the same signal as a genuine WAIT.
                if not llm_result.get("success"):
                    action = "WAIT (LLM unavailable - fail-safe default, not an evaluated decision)"
                elif decision is None:
                    action = "WAIT (response unparseable - fail-safe default, not a confirmed decision)"
                elif decision == "BUY_BACK":
                    action = (f"WAIT (LLM recommended BUY_BACK, blocked by hard rule - "
                              f"score {score_result['score']} < BUYBACK_FLOOR {BUYBACK_FLOOR})")
                else:
                    action = "WAIT"
                event, ledger = build_eval_event(best_stock, "buyback_wait_evaluation", action,
                                                 score_result, llm_result.get("content"),
                                                 llm_result, now)
                emit(event, ledger)

    # ── Persistence ordering (H1 fix) ──
    # 1) State first, atomically: a crash from here on can at worst
    #    lose a log entry, never create a phantom trade.
    save_json(HISTORY_PATH, history)
    save_json(POSITIONS_PATH, positions)
    # 2) Immutable ledger entries, only after state is durable.
    for event, ledger_entries in logs_this_run:
        persist_logs(event, ledger_entries)
    # 3) Summary reflects this run's freshly-flushed ledger entries.
    performance_summary = compute_performance_metrics()
    save_json(PERFORMANCE_SUMMARY_PATH, performance_summary)
    # 4) Dashboard snapshot last - it embeds events + summary.
    save_json(LATEST_PATH, {
        "timestamp": now.isoformat(), "scores": scores, "positions": positions,
        "portfolio_summary": compute_portfolio_summary(positions, by_stock),
        "performance_summary": performance_summary,
        "events_this_run": events_this_run,
    })
    # Small dashboard feed so the frontend never needs to download the
    # entire append-only event log just to show the latest activity.
    recent_events = read_jsonl(EVENT_LOG_PATH)[-20:]
    save_json(RECENT_EVENTS_PATH, list(reversed(recent_events)))

    print(f"[{now.isoformat()}] Run complete. {len(events_this_run)} LLM evaluation(s).")
    for event in events_this_run:
        print(f"  - {event.get('instrument')}: {event.get('type')} -> "
              f"{event.get('action_taken', event.get('type'))}")


if __name__ == "__main__":
    run()

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
     Below SELL_SEVERE the structural case for selling is strong;
     between SELL_SEVERE and SELL_CEILING is a genuine grey zone
     where the LLM's own judgment about whether the anomaly looks
     like durable structural stress vs a temporary/explainable
     artifact actually determines the outcome.
  6. Among SOLD tokens (holding pooled USDT), finds the
     highest-scoring one and ALWAYS asks the LLM to evaluate
     BUY BACK vs WAIT, with the mirrored grey-zone logic.
  7. On an actual SELL: proceeds credit the pooled USDT balance;
     the stock's record keeps entry_price and adds exit_price, so
     the full round-trip is visible. A matching entry is appended to
     performance_log.jsonl for realized P&L / win-rate / drawdown.
  8. On an actual BUY BACK: draws up to $300 from the pooled USDT
     balance (capped at whatever's actually available).
  9. Persists state + a recomputed performance_summary.json.

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

import json
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

STOCKS = ["NVDA", "TSLA", "AAPL", "AMZN", "GOOGL", "SPY", "QQQ", "KO", "MCD", "PYPL"]

INITIAL_HOLDING_USD = 300.0  # per stock, $3,000 total portfolio

# Hard rules (code-enforced, cannot be overridden by the LLM):
SELL_CEILING = 0.5       # never sell if score >= this
BUYBACK_FLOOR = 0.7      # never buy back if score < this
# Inside these bounds is the genuine "grey zone" - severity markers
# used only to give the LLM context in the prompt, not extra code
# branching:
SELL_SEVERE = 0.25       # below this, the structural case for selling is strong
BUYBACK_STRONG = 0.9     # above this, the case for buying back is strong


# ── Small I/O helpers ──────────────────────────────────────────

def load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def append_jsonl(path: str, entry: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


# ── State bootstrapping ────────────────────────────────────────

def init_positions(by_stock: dict) -> dict:
    """
    First-run bootstrap. USDT starts as its own explicit position at
    $0 (all capital is deployed into stocks at the start - nothing
    is held as cash yet). Each stock enters HELD at its REAL price
    from this run's fetch; quantity is fixed at this point and not
    recalculated later, which is what makes sell proceeds genuinely
    diverge from $300 depending on how price moved since entry.
    """
    positions = {"USDT": {"balance_usdt": 0.0}}
    for stock in STOCKS:
        price = by_stock.get(stock, {}).get("price")
        quantity = (INITIAL_HOLDING_USD / price) if price else None
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


def parse_decision(llm_text: str | None, positive_word: str) -> bool:
    """
    Looks for the expected keyword on the first line of the LLM's reply.
    Deliberate fail-safe: if the LLM produced no text at all (e.g. all
    three providers in the fallback chain failed), this returns False
    unconditionally - meaning a total LLM outage defaults to HOLD/WAIT
    (no action) rather than SELL/BUY_BACK. Taking no action on missing
    reasoning is the safer failure mode for a risk-monitoring system.
    The caller distinguishes this fail-safe default from a genuine
    LLM-evaluated HOLD/WAIT when logging (see action_taken in run()).
    """
    if not llm_text:
        return False
    first_line = llm_text.strip().splitlines()[0].upper()
    return positive_word in first_line


# ── Transaction + event + performance logging ───────────────────

def log_sell(stock: str, entry_price: float | None, exit_price: float | None,
             quantity: float | None, proceeds: float, realized_pnl: float,
             score_result: dict, reasoning: str, llm_meta: dict, now: datetime) -> dict:
    event_id = f"sell_{stock}_{now.strftime('%Y%m%dT%H%M%S')}"
    append_jsonl(TRANSACTION_LOG_PATH, {
        "event_id": event_id, "timestamp": now.isoformat(), "instrument": stock,
        "direction": "SELL", "price": exit_price, "quantity": quantity,
        "balance_change": round(proceeds, 2),
    })
    pnl_pct = round((realized_pnl / (entry_price * quantity)) * 100, 4) \
        if (entry_price and quantity) else None
    # Classify which zone triggered this - severe (score < SELL_SEVERE, an
    # easy/obvious call) vs grey zone (SELL_SEVERE <= score < SELL_CEILING,
    # where the LLM's own judgment actually decided the outcome). This is
    # what lets Layer 2 (decision quality) be assessed separately from
    # Layer 1 (the realized dollar P&L below).
    decision_zone = "severe" if score_result["score"] < SELL_SEVERE else "grey_zone"
    append_jsonl(PERFORMANCE_LOG_PATH, {
        "event_id": event_id, "timestamp": now.isoformat(), "instrument": stock,
        "entry_price": entry_price, "exit_price": exit_price, "quantity": quantity,
        "realized_pnl_usd": realized_pnl, "realized_pnl_pct": pnl_pct,
        "score_at_decision": score_result["score"], "decision_zone": decision_zone,
    })
    event = {
        "event_id": event_id, "timestamp": now.isoformat(), "type": "sell",
        "instrument": stock, "entry_price": entry_price, "exit_price": exit_price,
        "proceeds_usd": round(proceeds, 2), "realized_pnl_usd": realized_pnl,
        "score_at_decision": score_result["score"], "decision_zone": decision_zone,
        "llm_reasoning": reasoning, "llm_provider_used": llm_meta.get("provider_used"),
        "llm_attempts": llm_meta.get("attempts"),
    }
    append_jsonl(EVENT_LOG_PATH, event)
    return event


def log_buyback(stock: str, price: float | None, quantity: float | None,
                 cost: float, score_result: dict, reasoning: str, llm_meta: dict, now: datetime) -> dict:
    event_id = f"buy_{stock}_{now.strftime('%Y%m%dT%H%M%S')}"
    append_jsonl(TRANSACTION_LOG_PATH, {
        "event_id": event_id, "timestamp": now.isoformat(), "instrument": stock,
        "direction": "BUY", "price": price, "quantity": quantity,
        "balance_change": round(-cost, 2),
    })
    decision_zone = "strong" if score_result["score"] >= BUYBACK_STRONG else "grey_zone"
    event = {
        "event_id": event_id, "timestamp": now.isoformat(), "type": "buy_back",
        "instrument": stock, "price": price, "cost_usd": round(cost, 2),
        "score_at_decision": score_result["score"], "decision_zone": decision_zone,
        "llm_reasoning": reasoning, "llm_provider_used": llm_meta.get("provider_used"),
        "llm_attempts": llm_meta.get("attempts"),
    }
    append_jsonl(EVENT_LOG_PATH, event)
    return event


def log_evaluation_only(stock: str, decision_type: str, action_taken: str,
                         score_result: dict, reasoning: str, llm_meta: dict, now: datetime) -> dict:
    """Logs an LLM evaluation that did NOT result in a transaction (HOLD/WAIT)."""
    event = {
        "event_id": f"eval_{stock}_{now.strftime('%Y%m%dT%H%M%S')}",
        "timestamp": now.isoformat(), "type": decision_type, "instrument": stock,
        "action_taken": action_taken, "score": score_result["score"],
        "score_label": score_result["label"],
        "llm_reasoning": reasoning, "llm_provider_used": llm_meta.get("provider_used"),
        "llm_attempts": llm_meta.get("attempts"),
    }
    append_jsonl(EVENT_LOG_PATH, event)
    return event


# ── Portfolio + performance summaries ───────────────────────────

def compute_portfolio_summary(positions: dict, by_stock: dict) -> dict:
    """
    Reports BOTH cost-basis totals (what's actually been transacted -
    the basis for realized P&L) and current mark-to-market totals
    (what the portfolio is worth right now at live prices). Neither
    figure feeds back into the SELL/BUY decision logic, which stays
    purely structural - these are reporting-only numbers.
    """
    usdt = positions["USDT"]["balance_usdt"]
    held_cost_basis, held_market_value = 0.0, 0.0
    held_stocks, sold_stocks = [], []

    for stock in STOCKS:
        p = positions[stock]
        if p["status"] == "held":
            held_stocks.append(stock)
            held_cost_basis += p["cost_basis_usd"]
            current_price = by_stock.get(stock, {}).get("price")
            if current_price and p.get("quantity"):
                held_market_value += p["quantity"] * current_price
            else:
                held_market_value += p["cost_basis_usd"]
        else:
            sold_stocks.append(stock)

    return {
        "total_usdt_cash": round(usdt, 2),
        "total_held_cost_basis_usd": round(held_cost_basis, 2),
        "total_held_market_value_usd": round(held_market_value, 2),
        "total_portfolio_value_cost_basis_usd": round(usdt + held_cost_basis, 2),
        "total_portfolio_value_market_usd": round(usdt + held_market_value, 2),
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

    history = load_json(HISTORY_PATH, {})

    raw_entries = fetch_bitget_data()
    by_stock = {e["underlying"]: e for e in raw_entries}

    positions = load_json(POSITIONS_PATH, None)
    if positions is None:
        positions = init_positions(by_stock)

    scores = {}
    heartbeat_scores, heartbeat_labels = {}, {}
    for stock in STOCKS:
        entry = by_stock.get(stock, {"status": "error"})
        # Score against the PRE-update history (this run's values are
        # not yet included) - see health_score.py's score_stock
        # docstring for why this ordering matters.
        stock_history = history.get(stock, {"volume": [], "price": [], "depth": []})
        result = score_stock(entry, stock_history) if entry.get("status") == "ok" else \
            {"score": None, "label": "unknown", "components_raw": {}, "components_used": []}
        scores[stock] = result
        heartbeat_scores[stock] = result["score"]
        heartbeat_labels[stock] = result["label"]

    # Only now fold this run's values into history, so the NEXT run's
    # baseline includes them - never this run's own scoring.
    history = update_history(history, by_stock)

    append_jsonl(HEARTBEAT_LOG_PATH, {
        "timestamp": now.isoformat(), "scores": heartbeat_scores, "labels": heartbeat_labels,
    })

    events_this_run = []

    # --- Evaluate the lowest-scoring HELD stock ---
    held = [s for s in STOCKS if positions[s]["status"] == "held" and scores[s]["score"] is not None]
    if held:
        worst_stock = min(held, key=lambda s: scores[s]["score"])
        score_result = scores[worst_stock]
        prompt = build_sell_hold_prompt(worst_stock, score_result)
        llm_result = call_llm(prompt)
        decided_sell = parse_decision(llm_result.get("content"), "SELL") and score_result["score"] < SELL_CEILING

        if decided_sell:
            price = by_stock[worst_stock].get("price")
            qty = positions[worst_stock]["quantity"]
            entry_price = positions[worst_stock]["entry_price"]
            cost_basis = positions[worst_stock]["cost_basis_usd"]
            proceeds = qty * price if (qty and price) else cost_basis
            realized_pnl = round(proceeds - cost_basis, 2)

            positions["USDT"]["balance_usdt"] = round(positions["USDT"]["balance_usdt"] + proceeds, 2)
            positions[worst_stock] = {
                "status": "sold", "entry_price": entry_price, "exit_price": price,
                "quantity_sold": qty, "proceeds_usd": round(proceeds, 2),
                "realized_pnl_usd": realized_pnl, "sold_timestamp": now.isoformat(),
            }
            event = log_sell(worst_stock, entry_price, price, qty, proceeds, realized_pnl,
                              score_result, llm_result.get("content"), llm_result, now)
        else:
            action = "HOLD" if llm_result.get("success") else "HOLD (LLM unavailable - fail-safe default, not an evaluated decision)"
            event = log_evaluation_only(worst_stock, "sell_hold_evaluation", action,
                                         score_result, llm_result.get("content"), llm_result, now)
        events_this_run.append(event)

    # --- Evaluate the highest-scoring SOLD stock ---
    sold = [s for s in STOCKS if positions[s]["status"] == "sold" and scores[s]["score"] is not None]
    if sold:
        best_stock = max(sold, key=lambda s: scores[s]["score"])
        score_result = scores[best_stock]
        prompt = build_buyback_wait_prompt(best_stock, score_result)
        llm_result = call_llm(prompt)
        decided_buy = parse_decision(llm_result.get("content"), "BUY_BACK") and \
            score_result["score"] >= BUYBACK_FLOOR

        if decided_buy:
            price = by_stock[best_stock].get("price")
            available_cash = positions["USDT"]["balance_usdt"]
            target_notional = min(INITIAL_HOLDING_USD, available_cash)

            if target_notional <= 0 or not price:
                event = log_evaluation_only(best_stock, "buyback_wait_evaluation",
                                             "WAIT (insufficient USDT cash)", score_result,
                                             llm_result.get("content"), llm_result, now)
            else:
                qty = target_notional / price
                positions["USDT"]["balance_usdt"] = round(available_cash - target_notional, 2)
                positions[best_stock] = {
                    "status": "held", "entry_price": price, "exit_price": None,
                    "quantity": qty, "cost_basis_usd": round(target_notional, 2),
                }
                event = log_buyback(best_stock, price, qty, target_notional,
                                     score_result, llm_result.get("content"), llm_result, now)
        else:
            action = "WAIT" if llm_result.get("success") else "WAIT (LLM unavailable - fail-safe default, not an evaluated decision)"
            event = log_evaluation_only(best_stock, "buyback_wait_evaluation", action,
                                         score_result, llm_result.get("content"), llm_result, now)
        events_this_run.append(event)

    performance_summary = compute_performance_metrics()

    save_json(HISTORY_PATH, history)
    save_json(POSITIONS_PATH, positions)
    save_json(PERFORMANCE_SUMMARY_PATH, performance_summary)
    save_json(LATEST_PATH, {
        "timestamp": now.isoformat(), "scores": scores, "positions": positions,
        "portfolio_summary": compute_portfolio_summary(positions, by_stock),
        "performance_summary": performance_summary,
        "events_this_run": events_this_run,
    })

    print(f"[{now.isoformat()}] Run complete. {len(events_this_run)} LLM evaluation(s).")
    for event in events_this_run:
        print(f"  - {event.get('instrument')}: {event.get('type')} -> "
              f"{event.get('action_taken', event.get('type'))}")


if __name__ == "__main__":
    run()

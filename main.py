"""
Main orchestrator for Custos - single-issuer (Bitget) rToken
structural risk monitoring with active hold/sell/buy-back decisions.

Run on a schedule (every 4 hours via GitHub Actions). Each run:
  1. Pulls fresh data for all 10 rTokens from Bitget
  2. Updates rolling history (volume, price, order-book depth) used
     as each token's own baseline
  3. Computes a composite Health Score per token
  4. Logs a heartbeat entry (always) - proof the system stayed alive
     and monitoring throughout the competition
  5. Among HELD tokens, finds the lowest-scoring one and ALWAYS asks
     the LLM to evaluate SELL vs HOLD (with the actual score and an
     explicit threshold in the prompt, so a healthy token reliably
     gets HOLD rather than being sold just for ranking last)
  6. Among SOLD tokens (holding cash), finds the highest-scoring one
     and ALWAYS asks the LLM to evaluate BUY BACK vs WAIT
  7. On an actual SELL or BUY BACK: logs a standard transaction
     record (as required by the submission form) plus a narrative
     event log entry with the LLM's reasoning
  8. Persists state back to disk - the GitHub Actions workflow
     commits data/ back to the repo after this script runs

State files (all under data/):
  history.json          - rolling volume/price/depth history per stock
  positions.json         - status per stock: held (entry price/qty)
                            or sold (cash held, sale details)
  heartbeat_log.jsonl     - one line per run, all 10 scores
  event_log.jsonl         - narrative decision log (every LLM call,
                            not just ones that acted)
  transaction_log.jsonl   - standard format: timestamp, instrument,
                            direction, price, quantity, balance_change
  latest.json             - snapshot of the most recent run, for the
                            dashboard to read
"""

import json
import os
from datetime import datetime, timezone

from fetch_bitget import fetch_bitget_data
from health_score import score_stock, HISTORY_WINDOW
from llm_client import call_llm

DATA_DIR = "data"
HISTORY_PATH = f"{DATA_DIR}/history.json"
POSITIONS_PATH = f"{DATA_DIR}/positions.json"
HEARTBEAT_LOG_PATH = f"{DATA_DIR}/heartbeat_log.jsonl"
EVENT_LOG_PATH = f"{DATA_DIR}/event_log.jsonl"
TRANSACTION_LOG_PATH = f"{DATA_DIR}/transaction_log.jsonl"
LATEST_PATH = f"{DATA_DIR}/latest.json"

STOCKS = ["NVDA", "TSLA", "AAPL", "AMZN", "GOOGL", "SPY", "QQQ", "KO", "MCD", "PYPL"]

INITIAL_HOLDING_USD = 300.0  # per stock, $3,000 total portfolio

SELL_THRESHOLD = 0.5    # only sell if score is genuinely in red-flag territory
BUYBACK_THRESHOLD = 0.7  # only buy back once score has genuinely recovered


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


# ── State bootstrapping ────────────────────────────────────────

def init_positions() -> dict:
    """First-run bootstrap: every stock starts HELD with $300 exposure."""
    return {stock: {"status": "held", "entry_price": None, "quantity": None,
                     "cost_basis_usd": INITIAL_HOLDING_USD} for stock in STOCKS}


# ── History maintenance ─────────────────────────────────────────

def update_history(history: dict, by_stock: dict) -> dict:
    """
    Maintains rolling windows (HISTORY_WINDOW points) of volume,
    price, and order-book depth per stock - these are each token's
    own baseline, used by health_score.py's self-referential scoring.
    """
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
    return (
        f"You are a structural risk monitor for a tokenized stock (rToken) on Bitget, "
        f"NOT a directional stock picker - you do not predict price direction.\n\n"
        f"Token: {stock}\n"
        f"Current composite health score: {score_result['score']} "
        f"(label: {score_result['label']})\n"
        f"Component breakdown: {score_result['components_raw']}\n\n"
        f"Rule: only recommend SELL if the score genuinely reflects structural "
        f"risk (below {SELL_THRESHOLD}) - e.g. spread widening, thin order-book "
        f"depth, abnormal volatility, or unusual volume. If the score is healthy "
        f"or borderline, recommend HOLD. Do not react to price direction itself.\n\n"
        f"Respond with your decision (SELL or HOLD) on the first line, then a "
        f"2-3 sentence justification referencing the specific components above."
    )


def build_buyback_wait_prompt(stock: str, score_result: dict) -> str:
    return (
        f"You are a structural risk monitor for a tokenized stock (rToken) on Bitget. "
        f"This token was previously sold due to structural risk and is currently held "
        f"as cash (USDT).\n\n"
        f"Token: {stock}\n"
        f"Current composite health score: {score_result['score']} "
        f"(label: {score_result['label']})\n"
        f"Component breakdown: {score_result['components_raw']}\n\n"
        f"Rule: only recommend BUY BACK if the score has genuinely recovered "
        f"(at or above {BUYBACK_THRESHOLD}) - e.g. spread normalized, depth "
        f"restored, volatility settled. Otherwise recommend WAIT.\n\n"
        f"Respond with your decision (BUY_BACK or WAIT) on the first line, then a "
        f"2-3 sentence justification referencing the specific components above."
    )


def parse_decision(llm_text: str | None, positive_word: str) -> bool:
    """Looks for the expected keyword on the first line of the LLM's reply."""
    if not llm_text:
        return False
    first_line = llm_text.strip().splitlines()[0].upper()
    return positive_word in first_line


# ── Transaction + event logging ─────────────────────────────────

def log_sell(stock: str, price: float | None, quantity: float | None,
             proceeds: float, reasoning: str, llm_meta: dict, now: datetime) -> dict:
    event_id = f"sell_{stock}_{now.strftime('%Y%m%dT%H%M%S')}"
    append_jsonl(TRANSACTION_LOG_PATH, {
        "event_id": event_id, "timestamp": now.isoformat(), "instrument": stock,
        "direction": "SELL", "price": price, "quantity": quantity,
        "balance_change": round(proceeds, 2),
    })
    event = {
        "event_id": event_id, "timestamp": now.isoformat(), "type": "sell",
        "instrument": stock, "price": price, "proceeds_usd": round(proceeds, 2),
        "llm_reasoning": reasoning, "llm_provider_used": llm_meta.get("provider_used"),
        "llm_attempts": llm_meta.get("attempts"),
    }
    append_jsonl(EVENT_LOG_PATH, event)
    return event


def log_buyback(stock: str, price: float | None, quantity: float | None,
                 cost: float, reasoning: str, llm_meta: dict, now: datetime) -> dict:
    event_id = f"buy_{stock}_{now.strftime('%Y%m%dT%H%M%S')}"
    append_jsonl(TRANSACTION_LOG_PATH, {
        "event_id": event_id, "timestamp": now.isoformat(), "instrument": stock,
        "direction": "BUY", "price": price, "quantity": quantity,
        "balance_change": round(-cost, 2),
    })
    event = {
        "event_id": event_id, "timestamp": now.isoformat(), "type": "buy_back",
        "instrument": stock, "price": price, "cost_usd": round(cost, 2),
        "llm_reasoning": reasoning, "llm_provider_used": llm_meta.get("provider_used"),
        "llm_attempts": llm_meta.get("attempts"),
    }
    append_jsonl(EVENT_LOG_PATH, event)
    return event


def log_evaluation_only(stock: str, decision_type: str, action_taken: str,
                         score_result: dict, reasoning: str, llm_meta: dict, now: datetime) -> dict:
    """
    Logs an LLM evaluation that did NOT result in a transaction (e.g.
    HOLD or WAIT) - this is what keeps the event log populated every
    cycle even when the portfolio doesn't change, per the "always
    evaluate" design.
    """
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


# ── Main run ──────────────────────────────────────────────────

def run():
    now = datetime.now(timezone.utc)

    history = load_json(HISTORY_PATH, {})
    positions = load_json(POSITIONS_PATH, None)
    if positions is None:
        positions = init_positions()

    raw_entries = fetch_bitget_data()
    by_stock = {e["underlying"]: e for e in raw_entries}
    history = update_history(history, by_stock)

    scores = {}
    heartbeat_scores, heartbeat_labels = {}, {}
    for stock in STOCKS:
        entry = by_stock.get(stock, {"status": "error"})
        stock_history = history.get(stock, {"volume": [], "price": [], "depth": []})
        result = score_stock(entry, stock_history) if entry.get("status") == "ok" else \
            {"score": None, "label": "unknown", "components_raw": {}, "components_used": []}
        scores[stock] = result
        heartbeat_scores[stock] = result["score"]
        heartbeat_labels[stock] = result["label"]

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
        decided_sell = parse_decision(llm_result.get("content"), "SELL") and score_result["score"] < SELL_THRESHOLD

        if decided_sell:
            price = by_stock[worst_stock].get("price")
            qty = positions[worst_stock]["cost_basis_usd"] / price if price else None
            proceeds = qty * price if (qty and price) else positions[worst_stock]["cost_basis_usd"]
            event = log_sell(worst_stock, price, qty, proceeds,
                              llm_result.get("content"), llm_result, now)
            positions[worst_stock] = {"status": "sold", "cash_usdt": round(proceeds, 2),
                                       "sold_price": price, "sold_timestamp": now.isoformat()}
        else:
            event = log_evaluation_only(worst_stock, "sell_hold_evaluation", "HOLD",
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
            score_result["score"] >= BUYBACK_THRESHOLD

        if decided_buy:
            price = by_stock[best_stock].get("price")
            cash = positions[best_stock]["cash_usdt"]
            qty = cash / price if price else None
            event = log_buyback(best_stock, price, qty, cash,
                                 llm_result.get("content"), llm_result, now)
            positions[best_stock] = {"status": "held", "entry_price": price,
                                      "quantity": qty, "cost_basis_usd": cash}
        else:
            event = log_evaluation_only(best_stock, "buyback_wait_evaluation", "WAIT",
                                         score_result, llm_result.get("content"), llm_result, now)
        events_this_run.append(event)

    save_json(HISTORY_PATH, history)
    save_json(POSITIONS_PATH, positions)
    save_json(LATEST_PATH, {
        "timestamp": now.isoformat(), "scores": scores, "positions": positions,
        "events_this_run": events_this_run,
    })

    print(f"[{now.isoformat()}] Run complete. {len(events_this_run)} LLM evaluation(s).")
    for event in events_this_run:
        print(f"  - {event.get('instrument')}: {event.get('type')} -> "
              f"{event.get('action_taken', event.get('type'))}")


if __name__ == "__main__":
    run()

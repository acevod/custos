"""
Main orchestrator for the Custos issuer health monitoring system.

Run this on a schedule (every 15 min via GitHub Actions). Each run:
  1. Pulls fresh data from all 3 issuers (Bitget, Binance, OKX-xStocks)
  2. Updates rolling volume history (used as the volume_trend baseline)
  3. Computes a composite Health Score per issuer per stock
  4. Appends to the heartbeat log (always - this is what proves the
     system was "alive" and monitoring throughout the competition)
  5. Checks held positions - if the issuer currently holding exposure
     for a stock looks unhealthy, evaluates whether to rotate to a
     healthier issuer
  6. On rotation: logs a standard transaction record (SELL + BUY legs,
     as required by the submission form) plus a narrative event log
     entry with LLM-generated reasoning
  7. Persists updated state (positions + history) back to disk - the
     GitHub Actions workflow is responsible for committing these
     files back to the repo after this script runs

State files (all under data/):
  history.json          - rolling volume history per issuer+stock
  positions.json         - current simulated exposure per stock
  heartbeat_log.jsonl     - one line per run, all issuer/stock scores
  event_log.jsonl         - narrative decision log (rotations + flags)
  transaction_log.jsonl   - standard format: timestamp, instrument,
                            direction, price, quantity, balance_change
  latest.json             - snapshot of the most recent run, for the
                            dashboard to read
"""

import json
import os
from datetime import datetime, timezone

from fetch_bitget import fetch_bitget_data
from fetch_binance import fetch_binance_data
from fetch_okx import fetch_okx_data
from health_score import score_all_issuers, HISTORY_WINDOW
from llm_client import call_llm

DATA_DIR = "data"
HISTORY_PATH = f"{DATA_DIR}/history.json"
POSITIONS_PATH = f"{DATA_DIR}/positions.json"
HEARTBEAT_LOG_PATH = f"{DATA_DIR}/heartbeat_log.jsonl"
EVENT_LOG_PATH = f"{DATA_DIR}/event_log.jsonl"
TRANSACTION_LOG_PATH = f"{DATA_DIR}/transaction_log.jsonl"
LATEST_PATH = f"{DATA_DIR}/latest.json"

STOCKS = ["NVDA", "TSLA", "AAPL", "AMZN", "GOOGL"]
ISSUERS = ["bitget", "binance", "okx"]

INITIAL_EXPOSURE_PER_ISSUER = 300.0  # clean number, per issuer per stock
# -> $900 per stock (3 issuers), $4,500 total portfolio (5 stocks) -
#    stays within the $500-5000 retail trader persona used in the
#    project description.

# A position is only flagged for rotation if its current issuer's
# score drops into "red_flag" territory (see classify_score in
# health_score.py) - i.e. below 0.5.
ROTATION_TRIGGER_LABEL = "red_flag"


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
    """
    First-run bootstrap: $300 exposure per issuer per stock (clean
    number, $900/stock, $4,500 total portfolio). Only used if
    positions.json doesn't exist yet.
    """
    positions = {}
    for stock in STOCKS:
        for issuer in ISSUERS:
            positions.setdefault(stock, {})[issuer] = INITIAL_EXPOSURE_PER_ISSUER
    return positions


# ── Data collection ─────────────────────────────────────────────

def collect_all_issuer_data() -> dict:
    """
    Returns: { "NVDA": {"bitget": {...}, "binance": {...}, "okx": {...}}, ... }
    Each fetcher already handles its own per-symbol errors, so a
    single failed issuer/stock shows up as status="error" rather
    than crashing the whole run.
    """
    raw = {
        "bitget": fetch_bitget_data(),
        "binance": fetch_binance_data(),
        "okx": fetch_okx_data(),
    }

    by_stock = {stock: {} for stock in STOCKS}
    for issuer, entries in raw.items():
        for entry in entries:
            stock = entry["underlying"]
            if stock in by_stock:
                by_stock[stock][issuer] = entry
    return by_stock


def update_volume_history(history: dict, by_stock: dict) -> dict:
    """
    Maintains a rolling window (HISTORY_WINDOW points) of volume per
    issuer+stock. Key format: "ISSUER:STOCK".
    """
    for stock, issuer_entries in by_stock.items():
        for issuer, entry in issuer_entries.items():
            if entry.get("status") != "ok" or entry.get("volume_24h") is None:
                continue
            key = f"{issuer}:{stock}"
            history.setdefault(key, [])
            history[key].append(entry["volume_24h"])
            history[key] = history[key][-HISTORY_WINDOW:]
    return history


# ── Rotation logic ──────────────────────────────────────────────

def pick_best_alternative(current_issuer: str, scores: dict) -> tuple[str, float] | None:
    """
    Among issuers other than current_issuer, returns the one with the
    highest health score, as (issuer_name, score). Returns None if no
    other issuer has a valid score.
    """
    candidates = {
        name: result["score"]
        for name, result in scores.items()
        if name != current_issuer and result.get("score") is not None
    }
    if not candidates:
        return None
    best = max(candidates, key=candidates.get)
    return best, candidates[best]


def build_llm_prompt(stock: str, from_issuer: str, to_issuer: str, scores: dict) -> str:
    return (
        f"You are a risk-monitoring assistant for tokenized stock issuers. "
        f"For {stock}, the health score of {from_issuer} has degraded to "
        f"{scores[from_issuer]['score']} (components: {scores[from_issuer]['components_raw']}). "
        f"The healthiest alternative is {to_issuer} at {scores[to_issuer]['score']}. "
        f"In 2-3 sentences, explain the likely reason for {from_issuer}'s degradation "
        f"based on the component scores, and state whether rotating to {to_issuer} "
        f"is a reasonable risk-mitigation action right now."
    )


def execute_rotation(
    stock: str,
    from_issuer: str,
    to_issuer: str,
    exposure_usd: float,
    issuer_data: dict,
    scores: dict,
    now: datetime,
) -> dict:
    """
    Logs a rotation as two transaction legs (SELL the old issuer's
    position, BUY the new one) in the standard format required by the
    submission form, plus a narrative event log entry with LLM
    reasoning. Returns the event summary (also used for console output).
    """
    from_price = issuer_data[stock][from_issuer].get("price")
    to_price = issuer_data[stock][to_issuer].get("price")
    event_id = f"rot_{stock}_{now.strftime('%Y%m%dT%H%M%S')}"

    if from_price and to_price:
        quantity_from = round(exposure_usd / from_price, 6)
        quantity_to = round(exposure_usd / to_price, 6)
    else:
        quantity_from = quantity_to = None

    # Leg 1: SELL from the degraded issuer
    append_jsonl(TRANSACTION_LOG_PATH, {
        "event_id": event_id,
        "timestamp": now.isoformat(),
        "instrument": stock,
        "direction": f"SELL ({from_issuer})",
        "price": from_price,
        "quantity": quantity_from,
        "balance_change": round(-exposure_usd, 2) if exposure_usd else None,
    })
    # Leg 2: BUY into the healthier issuer
    append_jsonl(TRANSACTION_LOG_PATH, {
        "event_id": event_id,
        "timestamp": now.isoformat(),
        "instrument": stock,
        "direction": f"BUY ({to_issuer})",
        "price": to_price,
        "quantity": quantity_to,
        "balance_change": round(exposure_usd, 2) if exposure_usd else None,
    })

    # LLM reasoning (goes through the Qwen -> Groq -> OpenRouter fallback chain)
    prompt = build_llm_prompt(stock, from_issuer, to_issuer, scores)
    llm_result = call_llm(prompt)

    event = {
        "event_id": event_id,
        "timestamp": now.isoformat(),
        "type": "rotation",
        "instrument": stock,
        "from_issuer": from_issuer,
        "to_issuer": to_issuer,
        "exposure_usd": exposure_usd,
        "from_score": scores[from_issuer]["score"],
        "to_score": scores[to_issuer]["score"],
        "llm_reasoning": llm_result.get("content"),
        "llm_provider_used": llm_result.get("provider_used"),
        "llm_attempts": llm_result.get("attempts"),
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

    by_stock = collect_all_issuer_data()
    history = update_volume_history(history, by_stock)

    all_scores = {}
    heartbeat_entries = []
    events_this_run = []

    for stock in STOCKS:
        issuer_entries = by_stock.get(stock, {})

        volume_histories = {
            issuer: history.get(f"{issuer}:{stock}", []) for issuer in ISSUERS
        }
        scores = score_all_issuers(issuer_entries, volume_histories, now)
        all_scores[stock] = scores

        # Surface any fetch errors so failures are visible in the repo
        # instead of silently disappearing - this is what we check
        # when an issuer's score looks suspiciously flat/empty.
        errors = {
            issuer: entry.get("error")
            for issuer, entry in issuer_entries.items()
            if entry.get("status") == "error"
        }

        heartbeat_entries.append({
            "timestamp": now.isoformat(),
            "instrument": stock,
            "scores": {name: r["score"] for name, r in scores.items()},
            "labels": {name: r["label"] for name, r in scores.items()},
            "errors": errors if errors else None,
        })

        # Check whether the issuer currently holding exposure needs rotating
        for issuer, exposure in positions.get(stock, {}).items():
            if exposure <= 0:
                continue  # nothing held here - no action regardless of score
            issuer_score = scores.get(issuer, {})
            if issuer_score.get("label") != ROTATION_TRIGGER_LABEL:
                continue

            alternative = pick_best_alternative(issuer, scores)
            if alternative is None:
                continue
            to_issuer, to_score = alternative
            if to_score <= issuer_score["score"]:
                continue  # no better option available right now

            event = execute_rotation(
                stock, issuer, to_issuer, exposure, by_stock, scores, now
            )
            events_this_run.append(event)

            # Update position bookkeeping: move exposure between issuers
            positions[stock][issuer] = 0.0
            positions[stock][to_issuer] = positions[stock].get(to_issuer, 0.0) + exposure

    for entry in heartbeat_entries:
        append_jsonl(HEARTBEAT_LOG_PATH, entry)

    save_json(HISTORY_PATH, history)
    save_json(POSITIONS_PATH, positions)
    save_json(LATEST_PATH, {
        "timestamp": now.isoformat(),
        "scores": all_scores,
        "positions": positions,
        "events_this_run": events_this_run,
    })

    print(f"[{now.isoformat()}] Run complete. "
          f"{len(events_this_run)} rotation(s) triggered.")
    for event in events_this_run:
        print(f"  - {event['instrument']}: {event['from_issuer']} -> {event['to_issuer']}")


if __name__ == "__main__":
    run()

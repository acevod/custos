# Custos

[![Data pull status](https://github.com/acevod/custos/actions/workflows/data-pull.yml/badge.svg)](https://github.com/acevod/custos/actions/workflows/data-pull.yml)
![Track](https://img.shields.io/badge/track-Agentic%20Trading-5fa88f)
![Hackathon](https://img.shields.io/badge/Bitget%20AI%20Hackathon-S2-c9a24b)
![License](https://img.shields.io/badge/license-MIT-8890a4)
![Python](https://img.shields.io/badge/python-3.11-8890a4)

**[Live dashboard →](https://acevod.github.io/custos/)**

Structural risk monitor for Bitget stock rTokens. Custos does not predict where a
stock's price is going — it watches whether the *rToken wrapper itself* (spread,
top-of-book liquidity and price-move abnormality relative to its own history) is
showing signs of liquidity stress, independent of the underlying company's performance.
When a held token looks structurally unhealthy, an LLM evaluates whether to sell into
USDT; when a sold token's own condition recovers, the same evaluation runs in reverse.

Built for the **Agentic Trading** track (Open Theme) of Bitget AI Hackathon S2.

---

## Why this exists

Tokenized stocks (rTokens) trade 24/7, but the underlying shares they represent don't —
NASDAQ and NYSE still close nights and weekends. Outside those hours, Bitget supplies
liquidity for rTokens internally rather than routing to the real exchange. That's a
different liquidity *mechanism*, not automatically a worse one, but it's a risk that has
nothing to do with whether the underlying company is a good investment. A holder has no
easy way to tell "the stock dropped because of bad earnings" apart from "the wrapper's
liquidity is thinning" just by watching the price.

Custos is a narrow, honest attempt at that second problem: a background monitor that
reacts to the *wrapper's* condition, never to price direction or the company's
fundamentals.

## How it works

```
Every ~4 hours (GitHub Actions cron; real spacing observed: 3-8h):
  1. Pull live ticker data for 10 rTokens from Bitget's public API
  2. Score each token's Health Score from 4 components (2 historical, 2 fixed - see below)
  3. Log the full snapshot (heartbeat) — every cycle, regardless of outcome
  4. Find the lowest-scoring HELD token and highest-scoring SOLD token
  5. Deterministic gates first: enough components, a mature baseline (>= 12 points
     AND >= 48h), the hard score ceiling/floor, the score confirmed over consecutive
     runs, a minimum hold / buy-back cooldown. Only a candidate that passes ALL of
     them is sent to the LLM (SELL/HOLD or BUY_BACK/WAIT, as strict JSON)
  6. A reply only counts if it contains a parseable decision - otherwise the next
     LLM provider is tried
  7. Execute the paper trade
  8. Persist state before immutable logs, then recompute performance metrics
  9. Commit updated data back to the repo (rebase + retry if main moved)
```

### Health Score — four components (two historical, two fixed)

Two of four components are scored against **each token's own historical
baseline** (top-of-book size, abnormal movement) — never against
another token or against price direction. Spread uses a fixed absolute
threshold rather than a historical comparison, and weekend/after-hours is a
fixed calendar-based penalty rather than data-driven. This mix is what keeps
the system a liquidity/structural monitor rather than a stock picker — a
token can get flagged while its price rises, and stay untouched during a
price drop if its own liquidity mechanics look normal.

| Component | Weight | Basis | What it measures |
|---|---|---|---|
| Spread | 30% | Fixed threshold | Real-time bid/ask spread |
| Top-of-book size | 25% | Own history, same market regime | Best-bid + best-ask size vs this token's own recent MEDIAN, compared only with readings from the same regime (market open / weekday night / weekend) and lightly smoothed — a thin book is fragile even when the spread looks tight. This is top-of-book only, not full multi-level order-book depth |
| Abnormal movement | 25% | Own history | How large the latest price move is relative to this token's own recent volatility (normalised for elapsed time, skipped across gaps > 12h) — direction-agnostic, a sharp move up scores the same as a sharp move down. Note it does react to any large price move, including a genuine earnings gap; its 20% weight is what keeps that from triggering a sale on its own |
| Weekend / after-hours | 20% | Fixed calendar | A deliberate small penalty reflecting that Bitget's rToken liquidity works differently outside regular US hours: 1.0 in regular hours, 0.8 on weekdays outside them, 0.6 in the weekend window (Fri 20:00 ET → Sun 20:00 ET) and on market holidays (from 20:00 ET the evening before to 20:00 ET on the holiday - verified on Labor Day 2026). DST-aware (America/New_York); the NYSE holiday table covers 2026 only - extend it yearly. |

Weights are heuristic, manually tuned — not the result of backtesting, and that's
disclosed rather than dressed up. Missing components (e.g. not enough history yet) are
excluded and the remaining weights are renormalized, rather than treating "no data" as
either healthy or unhealthy. A token needs all 4 components available
before it's eligible for an actual SELL/BUY_BACK decision — with fewer, the system
logs a "warming up" state and skips the LLM call rather than acting on thin evidence.

Eligibility for an actual action is gated on two independent things, not just the
component count above: (1) all 4 components, **and** (2) a mature baseline:
at least `MIN_HISTORY_POINTS` (12) points **spanning at least `MIN_HISTORY_HOURS`
(48h)** of real calendar time (real run spacing is 3-8h, so a point count alone says
little). History points carry timestamps.

Because the historical components need only a few same-regime samples to compute, a token can show
4/4 components and a "healthy"/"watch"/"red flag" label on the dashboard while still
failing the second gate. The decision log then reads e.g. "HOLD (warming up - history
still warming up - baseline spans 30h, need 48h)" even though the Holdings row does not
say "warming up". That is two separate safety checks, not a bug, but worth knowing so
the two sections of the dashboard don't look contradictory. Conversely, a token whose
same-regime baseline is still too thin (e.g. the first weekend) shows "warming up" in
Holdings because those components report "no signal".

### Decision logic — hard rules plus a genuine grey zone

The LLM (Qwen, with Groq and OpenRouter as automatic fallbacks) is only consulted for
candidates that already passed every deterministic gate (see step 5 above): a score at
or above the ceiling never reaches the LLM at all, a single bad reading needs
`CONFIRM_RUNS` (2) consecutive runs, a token can't be sold within 12h of re-entering
and can't be bought back within 12h of its sale. The prompt states that *all values are
health scores where high = good*, and asks for a strict JSON reply
(`{"decision": ..., "reason": ...}`). Only the short `reason` is stored - never raw
chain-of-thought.

```
Score >= 0.5   → SELL is never executed, and the LLM is not even consulted (hard rule)
Score < 0.25   → structural stress is severe — LLM leans toward SELL unless it has a
                 concrete reason to think the reading is a temporary artifact
0.25 – 0.5     → genuine grey zone — the LLM's own reasoning about whether the pattern
                 looks like durable degradation or an explainable blip decides the
                 outcome, not a formula
```
(mirrored for buy-back: hard floor at 0.7, strong zone above 0.9, grey zone between)

Even with a passing score, a reply only counts if it contains a parseable decision;
otherwise the next provider is tried. If **no** provider returns a usable reply the
system holds - the fail-safe default is "do nothing", including in the severe zone.

The LLM is explicitly scoped to reasoning about the **wrapper's** condition — it is told
not to discuss whether the underlying company is a good investment.

## Repo structure

```
custos/
├── index.html              # dashboard (GitHub Pages)
├── og-image.png             # social preview image
├── main.py                  # orchestrator — run every cycle
├── fetch_bitget.py           # Bitget public market data
├── health_score.py           # composite scoring logic
├── llm_client.py              # Qwen → Groq → OpenRouter fallback chain
├── requirements.txt
├── CHANGES.md               # audit round-2 fixes and what is still open
├── LICENSE
├── .gitignore
├── .github/
│   ├── dependabot.yml         # weekly updates for Actions + pip
│   └── workflows/
│       └── data-pull.yml      # runs main.py every ~4 hours (20 min timeout)
├── data/                       # state and runtime-generated artifacts
│   ├── history.json             # schema v2: per-token price/depth + timestamps + recent scores (volume kept only to detect frozen snapshots)
│   ├── positions.json           # USDT balance + per-token held/sold state
│   ├── heartbeat_log.jsonl      # every score, every cycle
│   ├── event_log.jsonl          # decision/evaluation events
│   ├── recent_events.json       # bounded feed (last 20) the dashboard reads,
│   │                             so it never has to download the full
│   │                             append-only event_log.jsonl
│   ├── transaction_log.jsonl    # generated trade records (timestamp, instrument,
│   │                             direction, price, quantity, balance change)
│   ├── recent_transactions.json # bounded feed (last 20) the dashboard's Run
│   │                             Records section reads, same reasoning as
│   │                             recent_events.json above
│   ├── performance_log.jsonl    # one record per EXIT (sell) vs original entry
│   ├── roundtrip_log.jsonl      # one record per completed sell -> buy-back, vs holding
│   ├── raw_ticker_log.json      # last 30 runs of raw bid/ask, size and volume fields (diagnostics)
│   ├── performance_summary.json # exit + round-trip metrics, drawdown
│   └── latest.json              # snapshot the dashboard reads
└── tests/
    ├── test_custos.py
    ├── test_fixes.py            # regression tests for the audit round-2 fixes
    └── _fixtures.py             # fixed clock / same-regime history fixtures
```

## Running it yourself

No local setup is required to review the project — the dashboard and `data/` logs are
the live artifact. To run the orchestrator manually:

```bash
pip install -r requirements.txt
export GROQ_API_KEY=...        # free tier, no card required
export OPENROUTER_API_KEY=...   # free tier, no card required
python main.py
```
Run the tests with `python -m unittest discover -s tests -v` (they use a temporary
directory and a fixed clock, so they never touch `data/`).

Two practical notes for local runs:

- `python main.py` **executes against the real `data/` directory**: it appends a history
  point, may open or close paper positions, and rewrites the state files. Work in a
  separate clone (or throw the changes away) if the scheduled workflow owns the state,
  otherwise your commit will conflict with the next automated data commit.
- The market-hours calendar uses Python's `zoneinfo`. On Windows install the timezone
  database with `pip install tzdata`; without it the code falls back to an approximate
  US daylight-saving rule.

**Resetting the paper portfolio:** delete the *whole* `data/` directory, not individual
files. If `positions.json` is removed but `transaction_log.jsonl` is not (or vice versa),
the ledger check sees a mismatch and freezes that token. After a reset the warm-up
(>= 48h and 12 points, plus one weekend for weekend baselines) starts over.

`QWEN_BITGET_API_KEY` / `QWEN_BITGET_BASE_URL` are optional — without them, the LLM
chain falls back to Groq then OpenRouter automatically.

## Known limitations

- **Small sample size.** Trades accumulate slowly by design (the system is meant to hold,
  not churn) — `performance_summary.json` flags this explicitly rather than presenting
  early numbers as statistically robust.
- **Two performance views, only one answers the real question.** `exit_*` metrics compare
  each sell price with the original entry and ignore what re-entering cost; `round_trip_*`
  compares sell-then-rebuy against simply holding. Judge the strategy by the round-trip
  view. Neither is a backtest, and paper fills at top-of-book flatter real execution.
- **Scheduling isn't exact.** GitHub Actions doesn't guarantee precise cron timing on
  the free tier: observed gaps between runs were 3-8h (mean ~5.4h) for a nominal 4h
  schedule. History is timestamped so the scoring tolerates this, and the dashboard only
  shows a "no update" warning after 10h.
- **Third-party model availability can change without warning.** During development,
  Groq deprecated the free model this project originally used (`qwen3-32b`, announced
  17 Jun 2026), then deprecated its replacement `qwen3.6-27b` in favour of
  `qwen3.8-27b`; a separate free Qwen model on OpenRouter was also pulled. A deprecated
  Groq model id shows up as HTTP 404 (`http_404` in the event log). The chain now uses
  the same model family everywhere - Groq `qwen/qwen3.8-27b` (instruct mode,
  `reasoning_effort="none"`) and OpenRouter `qwen/qwen3.8-27b:free` - so the JSON
  decision contract behaves alike across providers. The trade-off is that OpenRouter is
  pinned to one free model again (the earlier `openrouter/free` auto-router survived
  models being pulled), so if that id is retired the OpenRouter link fails with a 404
  until the id is updated. The fallback chain absorbs the failure (it logs the outage
  rather than crashing) but recovery still needs a manual code change.
- **Volume is deliberately not a score component.** Bitget's `volume24h` is a counter that
  resets at 16:00 UTC (00:00 UTC+8) and, from Sunday 20:00 ET to Friday 20:00 ET, it mirrors the
  trading volume of the underlying US stock (about $45B/day for NVDA) rather than trading on
  Bitget; only in the weekend window does it show Bitget's own tiny volume. This was checked against
  Bitget's own hourly candles. It says nothing about the wrapper's liquidity, so it is recorded
  (for frozen-snapshot detection and diagnostics) but never scored.
- **Weekend baselines need a weekend of data.** Readings are compared only with the same
  market regime, so the first weekend after deployment has no weekend baseline and those
  components report "no signal" (the token shows "warming up").
- **A quiet feed and a frozen feed look identical.** Byte-identical snapshots are flagged
  "no fresh signal" and excluded from baselines instead of being scored as healthy.
- **The frozen-ledger state has no automatic recovery.** If `positions.json` and the
  transaction ledger ever disagree, that token is frozen with a "manual review required"
  event until someone fixes the files (or resets `data/`). In CI this is unlikely because
  state is only committed after a successful run.
- **Calendar model is simple.** US market hours are DST-aware, but the NYSE holiday table
  covers 2026 only (extend it yearly) and half-day early closes are not modelled.
- **Heuristic weights.** The four component weights were set manually based on reasoning
  about what each signal means, not fit to historical data.
- **Paper execution model.** Execution uses the available top-of-book price and a
  configured fee. It does not yet model full market impact or partial fills.
- **Partial data on first runs.** If a ticker fails to return usable execution data,
  the action path refuses to mutate state and logs an explicit HOLD/WAIT reason.
  JSONL readers also tolerate isolated corrupt lines.

## Current scope

Custos is a Bitget-only, scheduled, paper/simulation system. It is designed to
demonstrate an agentic decision loop with observable state, logs, safeguards, and
performance accounting rather than to act as a production live-trading bot.

## License

[MIT](LICENSE)

---

Built for Bitget AI Hackathon S2 · Agentic Trading track, Open Theme ·
Backed by Bitget, Alibaba Qwen, Alibaba Cloud

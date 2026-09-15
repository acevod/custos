# Custos

[![Data pull status](https://github.com/acevod/custos/actions/workflows/data-pull.yml/badge.svg)](https://github.com/acevod/custos/actions/workflows/data-pull.yml)
![Track](https://img.shields.io/badge/track-Agentic%20Trading-5fa88f)
![Hackathon](https://img.shields.io/badge/Bitget%20AI%20Hackathon-S2-c9a24b)
![License](https://img.shields.io/badge/license-MIT-8890a4)
![Python](https://img.shields.io/badge/python-3.11-8890a4)

**[Live dashboard →](https://acevod.github.io/custos/)**

Structural risk monitor for Bitget stock rTokens. Custos does not predict where a
stock's price is going — it watches whether the *rToken wrapper itself* (spread,
top-of-book liquidity, volume, and price-move abnormality relative to its own history) is
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
Every 4 hours (GitHub Actions):
  1. Pull live ticker data for 10 rTokens from Bitget's public API
  2. Score each token's Health Score from 5 components (3 historical, 2 fixed - see below)
  3. Log the full snapshot (heartbeat) — every cycle, regardless of outcome
  4. Find the lowest-scoring HELD token and highest-scoring SOLD token
  5. Require enough score components and mature history before an actual action
  6. If actionable, send the candidate to the LLM for SELL/HOLD or BUY_BACK/WAIT
  7. Apply hard code-enforced score rules and fail-safe execution guards
  8. Persist state before immutable logs, then recompute performance metrics
  9. Commit updated data back to the repo
```

### Health Score — five components (three historical, two fixed)

Three of five components are scored against **each token's own historical
baseline** (top-of-book size, volume trend, abnormal movement) — never against
another token or against price direction. Spread uses a fixed absolute
threshold rather than a historical comparison, and weekend/after-hours is a
fixed calendar-based penalty rather than data-driven. This mix is what keeps
the system a liquidity/structural monitor rather than a stock picker — a
token can get flagged while its price rises, and stay untouched during a
price drop if its own liquidity mechanics look normal.

| Component | Weight | Basis | What it measures |
|---|---|---|---|
| Spread | 25% | Fixed threshold | Real-time bid/ask spread |
| Top-of-book size | 20% | Own history | Best-bid + best-ask size vs this token's own recent average — a thin book is fragile even when the spread looks tight. This is top-of-book only, not full multi-level order-book depth |
| Volume trend | 20% | Own history | Current volume vs this token's own baseline |
| Abnormal movement | 20% | Own history | How large the latest price move is relative to this token's own recent volatility — direction-agnostic, a sharp move up scores the same as a sharp move down |
| Weekend / after-hours | 15% | Fixed calendar | A deliberate small penalty reflecting Bitget's internally-supplied liquidity outside NASDAQ/NYSE hours. Hardcoded UTC hours, doesn't account for DST or US market holidays |

Weights are heuristic, manually tuned — not the result of backtesting, and that's
disclosed rather than dressed up. Missing components (e.g. not enough history yet) are
excluded and the remaining weights are renormalized, rather than treating "no data" as
either healthy or unhealthy. A token needs at least 4 of the 5 components available
before it's eligible for an actual SELL/BUY_BACK decision — with fewer, the system
logs a "warming up" state and skips the LLM call rather than acting on thin evidence.

### Decision logic — hard rules plus a genuine grey zone

The LLM (Qwen, with Groq and OpenRouter as automatic fallbacks) is used for actionable
candidates. During the warm-up period, the system deliberately skips the LLM call and
logs the reason instead of making a decision from insufficient evidence.

```
Score >= 0.5   → SELL is never executed, regardless of what the LLM says (hard rule)
Score < 0.25   → structural stress is severe — LLM leans toward SELL unless it has a
                 concrete reason to think the reading is a temporary artifact
0.25 – 0.5     → genuine grey zone — the LLM's own reasoning about whether the pattern
                 looks like durable degradation or an explainable blip decides the
                 outcome, not a formula
```
(mirrored for buy-back: hard floor at 0.7, strong zone above 0.9, grey zone between)

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
├── LICENSE
├── .github/workflows/
│   └── data-pull.yml          # runs main.py every 4 hours
├── data/                       # state and runtime-generated artifacts
│   ├── history.json             # rolling per-token volume/price/depth history
│   ├── positions.json           # USDT balance + per-token held/sold state
│   ├── heartbeat_log.jsonl      # every score, every cycle
│   ├── event_log.jsonl          # decision/evaluation events
│   ├── transaction_log.jsonl    # generated trade records
│   ├── performance_log.jsonl    # generated completed round-trip records
│   ├── performance_summary.json # win rate, realized P&L, Sharpe-like, drawdown
│   └── latest.json              # snapshot the dashboard reads
└── tests/
    └── test_custos.py
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
`QWEN_BITGET_API_KEY` / `QWEN_BITGET_BASE_URL` are optional — without them, the LLM
chain falls back to Groq then OpenRouter automatically.

## Known limitations

- **Small sample size.** Realized trades accumulate slowly by design (the system is
  meant to hold, not churn) — `performance_summary.json` flags this explicitly rather
  than presenting early numbers as statistically robust.
- **Scheduling isn't exact.** GitHub Actions doesn't guarantee precise cron timing on
  the free tier, especially at common intervals; the 4-hour cadence tolerates the delay
  this can introduce.
- **Third-party model availability can change without warning.** During development,
  Groq deprecated the free model this project originally used (`qwen3-32b`, announced
  17 Jun 2026) and a separate free Qwen model on OpenRouter was also pulled — both
  fallback providers failed in the same run, alongside an unrelated timeout on the
  primary. The fix: Groq moved to a current model, and OpenRouter now uses its
  `openrouter/free` auto-router instead of a pinned model ID, so it no longer depends
  on one specific free model staying available. The fallback chain absorbed the
  failure (it logged the outage rather than crashing) but still needed a manual code
  fix to fully recover — a real limitation, not just a theoretical one.
- **Heuristic weights.** The five component weights were set manually based on reasoning
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

"""
Unit tests for Custos core logic. Run from the repo root with:

    python -m unittest discover -s tests -v

These lock in the security/correctness-critical behavior called out in
the audit: exact-match decision parsing (no substring attacks), the
action-eligibility gates (components + history maturity), ledger
reconciliation, and the calendar/scoring helpers.
"""

import os
import sys
import unittest
import unittest.mock
from datetime import datetime, timezone

# Make the repo root importable when discovered from tests/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main
from health_score import score_spread, score_weekend_afterhours, calculate_composite_score
from fetch_bitget import validate_ticker


class TestParseDecision(unittest.TestCase):
    """parse_decision is the security boundary between LLM output and
    trade execution - these tests must never regress."""

    def test_exact_match(self):
        self.assertEqual(main.parse_decision("SELL\nBecause the spread widened.", {"SELL", "HOLD"}), "SELL")

    def test_markdown_wrapped(self):
        self.assertEqual(main.parse_decision("**HOLD**", {"SELL", "HOLD"}), "HOLD")

    def test_trailing_punctuation(self):
        self.assertEqual(main.parse_decision("BUY_BACK.", {"BUY_BACK", "WAIT"}), "BUY_BACK")

    def test_substring_attack_rejected(self):
        # "HOLD - SELL is not justified" must NOT parse as SELL.
        self.assertIsNone(main.parse_decision("HOLD - SELL is not justified", {"SELL", "HOLD"}))

    def test_none_and_empty(self):
        self.assertIsNone(main.parse_decision(None, {"SELL", "HOLD"}))
        self.assertIsNone(main.parse_decision("", {"SELL", "HOLD"}))

    def test_unknown_word(self):
        self.assertIsNone(main.parse_decision("UNCLEAR", {"SELL", "HOLD"}))

    def test_whitespace_only_does_not_crash(self):
        # H-1 regression: a truthy-but-blank LLM response ("   ",
        # "\n\n") used to slip past `if not llm_text` and then crash
        # with IndexError on splitlines()[0]. Must return None
        # (treated as "unparseable", the safe fail-closed outcome),
        # never raise.
        self.assertIsNone(main.parse_decision("   ", {"SELL", "HOLD"}))
        self.assertIsNone(main.parse_decision("\n\n\t", {"SELL", "HOLD"}))


class TestCheckActionable(unittest.TestCase):
    """H2 fix: actionability requires enough components AND a mature
    baseline, not just a component count."""

    def _history(self, n):
        return {"price": [100.0] * n, "volume": [1.0] * n, "depth": [1.0] * n}

    def test_blocks_insufficient_components(self):
        result = {"score": 0.3, "components_used": ["spread", "weekend"], "label": "red_flag"}
        history = {"NVDA": self._history(50)}
        eligible, reason = main.check_actionable("NVDA", result, history)
        self.assertFalse(eligible)
        self.assertIn("components", reason)

    def test_blocks_immature_history(self):
        result = {"score": 0.3, "components_used": ["spread", "depth", "volume_trend", "weekend"],
                  "label": "red_flag"}
        history = {"NVDA": self._history(3)}
        eligible, reason = main.check_actionable("NVDA", result, history)
        self.assertFalse(eligible)
        self.assertIn("warming up", reason)

    def test_allows_mature_stock(self):
        result = {"score": 0.3, "components_used": ["spread", "depth", "volume_trend", "weekend"],
                  "label": "red_flag"}
        history = {"NVDA": self._history(main.MIN_HISTORY_POINTS)}
        eligible, _ = main.check_actionable("NVDA", result, history)
        self.assertTrue(eligible)


class TestLedgerReconciliation(unittest.TestCase):
    """H1 safety net: ledger entries and positions.json must agree on
    how many sells are currently open per stock."""

    def _positions(self, status):
        return {"USDT": {"balance_usdt": 0.0},
                **{s: {"status": status} for s in main.STOCKS}}

    def test_healthy_empty_state(self):
        sells, buys = {}, {}
        self.assertEqual(main.find_ledger_mismatches(self._positions("held"), sells, buys), {})

    def test_one_open_sell_matches_sold_status(self):
        sells = {"NVDA": 1}
        positions = self._positions("held")
        positions["NVDA"]["status"] = "sold"
        self.assertEqual(main.find_ledger_mismatches(positions, sells, {}), {})

    def test_phantom_sell_detected(self):
        # Ledger says NVDA was sold once and never bought back, but
        # positions.json still says held - this is exactly the
        # divergence a crash mid-run used to produce silently.
        sells = {"NVDA": 1}
        positions = self._positions("held")
        positions["NVDA"]["status"] = "held"
        problems = main.find_ledger_mismatches(positions, sells, {})
        self.assertIn("NVDA", problems)

    def test_buyback_resolves_sell(self):
        sells = {"NVDA": 1}
        buys = {"NVDA": 1}
        self.assertEqual(main.find_ledger_mismatches(self._positions("held"), sells, buys), {})

    def test_state_sold_but_no_ledger_entry(self):
        positions = self._positions("held")
        positions["NVDA"]["status"] = "sold"
        problems = main.find_ledger_mismatches(positions, {}, {})
        self.assertIn("NVDA", problems)


class TestInitPositions(unittest.TestCase):
    def test_ask_preferred(self):
        by_stock = {"NVDA": {"status": "ok", "ask": 210.0, "price": 209.5, "bid": 209.0}}
        positions = main.init_positions(by_stock)
        self.assertEqual(positions["NVDA"]["entry_price"], 210.0)
        self.assertAlmostEqual(positions["NVDA"]["quantity"] * 210.0, 300.0 * (1 - main.TRADING_FEE_PCT))

    def test_fallback_to_price(self):
        by_stock = {"NVDA": {"status": "ok", "ask": None, "price": 209.5}}
        positions = main.init_positions(by_stock)
        self.assertEqual(positions["NVDA"]["entry_price"], 209.5)

    def test_missing_price_leaves_none(self):
        by_stock = {"NVDA": {"status": "error"}}
        positions = main.init_positions(by_stock)
        self.assertIsNone(positions["NVDA"]["entry_price"])
        self.assertIsNone(positions["NVDA"]["quantity"])


class TestScoringHelpers(unittest.TestCase):
    def test_spread_none(self):
        self.assertIsNone(score_spread(None))

    def test_spread_curve(self):
        self.assertEqual(score_spread(0.0), 1.0)
        self.assertEqual(score_spread(1.0), 0.0)
        self.assertAlmostEqual(score_spread(0.5), 0.5)

    def test_weekend(self):
        saturday = datetime(2026, 9, 12, 15, 0, tzinfo=timezone.utc)  # a Saturday
        self.assertEqual(score_weekend_afterhours(saturday), 0.6)

    def test_weekday_after_hours(self):
        tuesday_morning = datetime(2026, 9, 8, 10, 0, tzinfo=timezone.utc)
        self.assertEqual(score_weekend_afterhours(tuesday_morning), 0.8)

    def test_weekday_market_hours(self):
        tuesday_midday = datetime(2026, 9, 8, 15, 0, tzinfo=timezone.utc)
        self.assertEqual(score_weekend_afterhours(tuesday_midday), 1.0)

    def test_composite_renormalizes_missing(self):
        result = calculate_composite_score({"spread": 0.8, "weekend": 1.0})
        self.assertIsNotNone(result["score"])
        self.assertEqual(set(result["components_used"]), {"spread", "weekend"})

    def test_composite_all_missing(self):
        result = calculate_composite_score({"spread": None, "depth": None})
        self.assertIsNone(result["score"])


class TestRobustIO(unittest.TestCase):
    """Regression tests for the hardened JSON/JSONL loaders that
    prevent a single corrupt line or file from crashing the agent."""

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.TemporaryDirectory()
        self.dir = self._tmpdir.name

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_read_jsonl_skips_corrupt_lines(self):
        path = os.path.join(self.dir, "test.jsonl")
        with open(path, "w") as f:
            f.write('{"ok": 1}\n')
            f.write('this is not json\n')
            f.write('{"ok": 2}\n')
            f.write('\n')
            f.write('{broken\n')
        entries = main.read_jsonl(path)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["ok"], 1)
        self.assertEqual(entries[1]["ok"], 2)

    def test_read_jsonl_missing_file(self):
        self.assertEqual(main.read_jsonl(os.path.join(self.dir, "nope.jsonl")), [])

    def test_load_json_corrupt_fails_closed_when_required(self):
        path = os.path.join(self.dir, "bad.json")
        with open(path, "w") as f:
            f.write("{not valid json")
        with self.assertRaises(main.StateCorruptionError):
            main.load_json(path, {"fallback": True}, required=True)

    def test_load_json_corrupt_can_still_use_default_for_non_state_data(self):
        path = os.path.join(self.dir, "bad.json")
        with open(path, "w") as f:
            f.write("{not valid json")
        result = main.load_json(path, {"fallback": True})
        self.assertEqual(result, {"fallback": True})

    def test_load_json_missing_returns_default(self):
        result = main.load_json(os.path.join(self.dir, "missing.json"), [])
        self.assertEqual(result, [])


class TestMarketDataValidation(unittest.TestCase):
    def _ticker(self, **overrides):
        ticker = {
            "ts": str(int(datetime.now(timezone.utc).timestamp() * 1000)),
            "lastPrice": "100", "bid1Price": "99.9", "ask1Price": "100.1",
            "bid1Size": "10", "ask1Size": "12", "volume24h": "1000",
        }
        ticker.update(overrides)
        return ticker

    def test_valid_ticker(self):
        market, _ = validate_ticker(self._ticker())
        self.assertEqual(market["price"], 100.0)
        self.assertIn("source_timestamp", market)

    def test_stale_ticker_rejected(self):
        old_ts = int((datetime.now(timezone.utc).timestamp() - 3600) * 1000)
        with self.assertRaises(ValueError):
            validate_ticker(self._ticker(ts=str(old_ts)))

    def test_non_finite_price_rejected(self):
        with self.assertRaises(ValueError):
            validate_ticker(self._ticker(lastPrice="NaN"))

    def test_invalid_order_book_rejected(self):
        with self.assertRaises(ValueError):
            validate_ticker(self._ticker(bid1Price="101", ask1Price="100"))


class TestStateValidation(unittest.TestCase):
    def _positions(self):
        return {
            "USDT": {"balance_usdt": 0.0},
            **{s: {"status": "held", "quantity": 1.0, "cost_basis_usd": 300.0}
               for s in main.STOCKS},
        }

    def test_valid_positions(self):
        main.validate_positions_state(self._positions())

    def test_invalid_held_quantity_rejected(self):
        positions = self._positions()
        positions["NVDA"]["quantity"] = 0
        with self.assertRaises(main.StateCorruptionError):
            main.validate_positions_state(positions)

    def test_invalid_history_rejected(self):
        history = {"NVDA": {"price": [100.0, float("nan")], "volume": [], "depth": []}}
        with self.assertRaises(main.StateCorruptionError):
            main.validate_history_state(history)


class TestRunIntegration(unittest.TestCase):
    """M-3: end-to-end tests for run() itself, with fetch_bitget_data
    and call_llm mocked out. These exercise the orchestration logic
    that the smaller unit tests above can't reach on their own -
    especially the hard-rule enforcement, which is the single most
    security/financially-critical path in the whole system."""

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.TemporaryDirectory()
        self._cwd = os.getcwd()
        os.chdir(self._tmpdir.name)
        os.makedirs("data", exist_ok=True)

    def tearDown(self):
        os.chdir(self._cwd)
        self._tmpdir.cleanup()

    def _base_positions(self, nvda_status="held"):
        positions = {"USDT": {"balance_usdt": 100.0}}
        for stock in main.STOCKS:
            if stock == "NVDA":
                if nvda_status == "held":
                    positions["NVDA"] = {
                        "status": "held", "entry_price": 200.0, "exit_price": None,
                        "quantity": 1.5, "cost_basis_usd": 300.0,
                    }
                else:
                    positions["NVDA"] = {
                        "status": "sold", "entry_price": 200.0, "exit_price": 200.0,
                        "quantity_sold": 1.5, "proceeds_usd": 300.0,
                        "realized_pnl_usd": 0.0, "sold_timestamp": "2026-01-01T00:00:00+00:00",
                    }
            else:
                # Other 9 stocks: valid but irrelevant - fetch will
                # return "error" for them so their score is None and
                # they're excluded from the held/sold candidate lists.
                positions[stock] = {
                    "status": "held", "entry_price": 100.0, "exit_price": None,
                    "quantity": 3.0, "cost_basis_usd": 300.0,
                }
        return positions

    def _mature_history(self, n=main.MIN_HISTORY_POINTS):
        return {"NVDA": {
            "volume": [50_000_000.0] * n,
            "price": [200.0] * n,
            "depth": [200.0] * n,
        }}

    def _healthy_nvda_fetch_entry(self):
        # Tight spread, in-line depth/volume, no abnormal move -
        # every component scores high, so the composite lands well
        # above SELL_CEILING (0.5) / at BUYBACK_STRONG territory.
        return {
            "underlying": "NVDA", "symbol": "rNVDAUSDT", "status": "ok",
            "price": 200.0, "bid": 199.9, "ask": 200.1,
            "bid_size": 100.0, "ask_size": 100.0,
            "volume_24h": 50_000_000.0, "spread_pct": 0.1,
        }

    def _write_state(self, history, positions):
        main.save_json(main.HISTORY_PATH, history)
        main.save_json(main.POSITIONS_PATH, positions)

    def _last_event(self):
        events = main.read_jsonl(main.EVENT_LOG_PATH)
        self.assertTrue(events, "expected at least one event to be logged")
        return events[-1]

    def test_hard_ceiling_blocks_sell_even_when_llm_says_sell(self):
        """SELL_CEILING must be enforced regardless of what the LLM
        says - a healthy score (>= 0.5) must never actually execute
        a sell, even if the LLM's first line is literally 'SELL'."""
        self._write_state(self._mature_history(), self._base_positions("held"))

        with unittest.mock.patch("main.fetch_bitget_data",
                                  return_value=[self._healthy_nvda_fetch_entry()]), \
             unittest.mock.patch("main.call_llm",
                                  return_value={"success": True,
                                                "content": "SELL\nSpread looks a bit wide.",
                                                "provider_used": "test", "attempts": []}):
            main.run()

        positions = main.load_json(main.POSITIONS_PATH, None)
        self.assertEqual(positions["NVDA"]["status"], "held",
                          "hard SELL_CEILING rule was bypassed - NVDA got sold")
        event = self._last_event()
        self.assertIn("blocked by hard rule", event.get("action_taken", ""))

    def test_hard_floor_blocks_buyback_even_when_llm_says_buy_back(self):
        """BUYBACK_FLOOR must be enforced regardless of what the LLM
        says - too low a score must never actually execute a
        buy-back, even if the LLM's first line is literally
        'BUY_BACK'."""
        history = self._mature_history()
        positions = self._base_positions("sold")
        # The startup reconciliation check requires the ledger to
        # already show NVDA as sold (one performance_log entry, no
        # matching buy-back) - otherwise main.py correctly treats
        # positions.json's "sold" status as an unexplained mismatch
        # and freezes the stock instead of evaluating it, which would
        # make this test about reconciliation, not about the hard
        # floor. Seed a matching ledger entry so NVDA is eligible.
        main.append_jsonl(main.PERFORMANCE_LOG_PATH, {
            "event_id": "sell_NVDA_seed", "timestamp": "2026-01-01T00:00:00+00:00",
            "instrument": "NVDA", "entry_price": 200.0, "exit_price": 200.0,
            "quantity": 1.5, "realized_pnl_usd": 0.0, "realized_pnl_pct": 0.0,
            "score_at_decision": 0.3, "decision_zone": "grey_zone",
        })

        with unittest.mock.patch("main.fetch_bitget_data",
                                  return_value=[self._healthy_nvda_fetch_entry()]) as fetch_mock:
            # Force a low score by making the fetch entry look
            # structurally stressed (very wide spread, thin depth,
            # low volume) while keeping >=4 components available.
            fetch_mock.return_value = [{
                "underlying": "NVDA", "symbol": "rNVDAUSDT", "status": "ok",
                "price": 200.0, "bid": 198.0, "ask": 202.0,
                "bid_size": 2.0, "ask_size": 2.0,
                "volume_24h": 500_000.0, "spread_pct": 2.0,
            }]
            self._write_state(history, positions)
            with unittest.mock.patch("main.call_llm",
                                      return_value={"success": True,
                                                    "content": "BUY_BACK\nLooks recovered.",
                                                    "provider_used": "test", "attempts": []}):
                main.run()

        result_positions = main.load_json(main.POSITIONS_PATH, None)
        self.assertEqual(result_positions["NVDA"]["status"], "sold",
                          "hard BUYBACK_FLOOR rule was bypassed - NVDA got bought back")
        event = self._last_event()
        self.assertIn("blocked by hard rule", event.get("action_taken", ""))

    def test_immature_history_skips_llm_even_after_this_runs_own_update(self):
        """M-1 regression: check_actionable() must use the PRE-update
        history snapshot. With exactly MIN_HISTORY_POINTS - 1 prior
        points, the stock must still be 'warming up' THIS cycle -
        the fresh reading this run just fetched must not count toward
        its own maturity gate. If the bug regresses, this stock would
        incorrectly become eligible and call_llm WOULD be invoked."""
        history = self._mature_history(n=main.MIN_HISTORY_POINTS - 1)
        self._write_state(history, self._base_positions("held"))

        with unittest.mock.patch("main.fetch_bitget_data",
                                  return_value=[self._healthy_nvda_fetch_entry()]), \
             unittest.mock.patch("main.call_llm") as llm_mock:
            main.run()
            llm_mock.assert_not_called()

        event = self._last_event()
        self.assertIn("warming up", event.get("action_taken", ""))

    def test_whitespace_llm_response_does_not_crash_run(self):
        """H-1 regression at the integration level: a whitespace-only
        completion from the LLM must not raise inside run() - it
        should fail closed to HOLD, and state must still be
        persisted for this cycle."""
        self._write_state(self._mature_history(), self._base_positions("held"))

        with unittest.mock.patch("main.fetch_bitget_data",
                                  return_value=[self._healthy_nvda_fetch_entry()]), \
             unittest.mock.patch("main.call_llm",
                                  return_value={"success": True, "content": "   \n  ",
                                                "provider_used": "test", "attempts": []}):
            main.run()  # must not raise

        positions = main.load_json(main.POSITIONS_PATH, None)
        self.assertEqual(positions["NVDA"]["status"], "held")


if __name__ == "__main__":
    unittest.main()

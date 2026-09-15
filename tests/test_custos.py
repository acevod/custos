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


if __name__ == "__main__":
    unittest.main()

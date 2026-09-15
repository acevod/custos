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


if __name__ == "__main__":
    unittest.main()

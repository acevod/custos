"""
Regression tests for the audit-round-2 fixes (see CHANGES.md). Run with:

    python -m unittest discover -s tests -v
"""

import json
import os
import sys
import tempfile
import unittest
import unittest.mock as mock
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fetch_bitget
import health_score as hs
import llm_client
import main
from _fixtures import FIXED_NOW, FixedDatetime, open_market_stamps


def utc(*args):
    return datetime(*args, tzinfo=timezone.utc)


# ── LLM reply parsing ─────────────────────────────────────────

class TestParseLlmDecision(unittest.TestCase):
    SELL = {"SELL", "HOLD"}

    def test_json_reply(self):
        d, r = main.parse_llm_decision('{"decision": "SELL", "reason": "spread wide"}', self.SELL)
        self.assertEqual((d, r), ("SELL", "spread wide"))

    def test_json_in_code_fence(self):
        d, _ = main.parse_llm_decision('```json\n{"decision":"hold","reason":"ok"}\n```', self.SELL)
        self.assertEqual(d, "HOLD")

    def test_think_block_is_stripped(self):
        d, _ = main.parse_llm_decision(
            '<think>lots of reasoning about SELL</think>{"decision":"HOLD","reason":"fine"}', self.SELL)
        self.assertEqual(d, "HOLD")

    def test_unterminated_think_block_is_no_decision(self):
        self.assertEqual(main.parse_llm_decision("<think>Here's a thinking process: 1. Analyze", self.SELL),
                         (None, None))

    def test_buy_back_with_space_in_json(self):
        d, _ = main.parse_llm_decision('{"decision": "BUY BACK", "reason": "x"}', {"BUY_BACK", "WAIT"})
        self.assertEqual(d, "BUY_BACK")

    def test_reason_is_length_capped(self):
        d, r = main.parse_llm_decision(json.dumps({"decision": "SELL", "reason": "x" * 5000}), self.SELL)
        self.assertEqual(d, "SELL")
        self.assertLessEqual(len(r), main.LLM_REASON_MAX_CHARS)

    def test_legacy_first_line_fallback(self):
        d, r = main.parse_llm_decision("SELL\nSpread widened a lot.", self.SELL)
        self.assertEqual((d, r), ("SELL", "Spread widened a lot."))

    def test_json_without_valid_decision_is_none(self):
        self.assertEqual(main.parse_llm_decision('{"decision": "MAYBE"}', self.SELL), (None, None))

    def test_non_string_never_crashes(self):
        for bad in (123, ["SELL"], {"a": 1}, None, b"SELL"):
            self.assertEqual(main.parse_llm_decision(bad, self.SELL), (None, None))
            self.assertIsNone(main.parse_decision(bad, self.SELL))

    def test_legacy_parser_accepts_common_variants(self):
        self.assertEqual(main.parse_decision("SELL, spread widened", self.SELL), "SELL")
        self.assertEqual(main.parse_decision("Decision: SELL", self.SELL), "SELL")
        self.assertEqual(main.parse_decision("BUY BACK - recovered", {"BUY_BACK", "WAIT"}), "BUY_BACK")
        self.assertEqual(main.parse_decision("HOLD - SELL is not justified", self.SELL), "HOLD")


class TestPromptContract(unittest.TestCase):
    RESULT = {"score": 0.42, "label": "red_flag",
              "components_raw": {"spread": 0.78, "abnormal_movement": 0.99}}

    def test_sell_prompt_states_score_direction(self):
        prompt = main.build_sell_hold_prompt("GOOGL", self.RESULT, {"spread_pct": 0.22,
                                                                     "bid_size": 5.0, "ask_size": 6.0})
        self.assertIn("HIGH value is GOOD", prompt)
        self.assertIn("1.0 = healthy", prompt)
        self.assertIn("raw bid/ask spread 0.22%", prompt)
        self.assertIn('"decision"', prompt)          # JSON contract

    def test_buyback_prompt_states_score_direction(self):
        prompt = main.build_buyback_wait_prompt("GOOGL", self.RESULT)
        self.assertIn("HIGH value is GOOD", prompt)
        self.assertIn("BUY_BACK", prompt)


# ── LLM client fallback / error handling ──────────────────────

class _Resp:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code, self._payload, self.text = status, payload, text

    def json(self):
        return self._payload


def _chat(content, finish="stop"):
    return _Resp(200, {"choices": [{"message": {"content": content}, "finish_reason": finish}]})


class TestLlmClient(unittest.TestCase):
    def _providers(self):
        return [
            {"name": "a", "base_url": "https://secret-a.example/v1", "api_key": "k", "model": "m"},
            {"name": "b", "base_url": "https://b.example/v1", "api_key": "k", "model": "m"},
        ]

    def test_truncated_reply_falls_through_to_next_provider(self):
        replies = [_chat("Here's a thinking process: 1.", finish="length"),
                   _chat('{"decision": "HOLD", "reason": "ok"}')]
        with mock.patch.object(llm_client, "PROVIDERS", self._providers()), \
             mock.patch.object(llm_client.requests, "post", side_effect=replies):
            out = llm_client.call_llm("p", validator=main.make_validator({"SELL", "HOLD"}))
        self.assertTrue(out["success"])
        self.assertEqual(out["provider_used"], "b")
        self.assertEqual(out["attempts"][0]["status"], "unusable_response")
        self.assertEqual(out["attempts"][0]["error"], "truncated")

    def test_empty_and_null_content_are_unusable(self):
        replies = [_chat(None), _chat("   ")]
        with mock.patch.object(llm_client, "PROVIDERS", self._providers()), \
             mock.patch.object(llm_client.requests, "post", side_effect=replies):
            out = llm_client.call_llm("p")
        self.assertFalse(out["success"])
        self.assertEqual([a["error"] for a in out["attempts"]], ["empty_response", "empty_response"])

    def test_list_content_is_normalised(self):
        reply = _chat([{"type": "text", "text": '{"decision":"SELL","reason":"r"}'}])
        with mock.patch.object(llm_client, "PROVIDERS", self._providers()[:1]), \
             mock.patch.object(llm_client.requests, "post", return_value=reply):
            out = llm_client.call_llm("p", validator=main.make_validator({"SELL", "HOLD"}))
        self.assertTrue(out["success"])

    def test_error_strings_never_contain_urls(self):
        body = "model not found at https://secret-a.example/v1/chat/completions"
        with mock.patch.object(llm_client, "PROVIDERS", self._providers()), \
             mock.patch.object(llm_client.requests, "post", return_value=_Resp(404, None, body)):
            out = llm_client.call_llm("p")
        dumped = json.dumps(out)
        self.assertNotIn("secret-a", dumped)
        self.assertEqual(out["attempts"][0]["error"], "http_404")

    def test_unexpected_exception_reports_category_only(self):
        boom = RuntimeError("failed for url https://secret-a.example/v1")
        with mock.patch.object(llm_client, "PROVIDERS", self._providers()[:1]), \
             mock.patch.object(llm_client.requests, "post", side_effect=boom):
            out = llm_client.call_llm("p")
        self.assertNotIn("secret-a", json.dumps(out))
        self.assertEqual(out["attempts"][0]["error"], "error_RuntimeError")

    def test_validator_exception_counts_as_unusable(self):
        with mock.patch.object(llm_client, "PROVIDERS", self._providers()[:1]), \
             mock.patch.object(llm_client.requests, "post", return_value=_chat("x")):
            out = llm_client.call_llm("p", validator=lambda c: 1 / 0)
        self.assertFalse(out["success"])


class TestProviderConfig(unittest.TestCase):
    """The Groq 404 in production was a deprecated model id. Pin the chain."""

    def test_model_ids(self):
        models = {p["name"]: p["model"] for p in llm_client.PROVIDERS}
        self.assertEqual(models["groq"], "qwen/qwen3.8-27b")
        self.assertEqual(models["openrouter"], "qwen/qwen3.8-27b:free")
        for deprecated in ("qwen/qwen3.6-27b", "qwen/qwen3-32b", "openrouter/free"):
            self.assertNotIn(deprecated, models.values())

    def test_only_groq_gets_reasoning_effort_and_it_reaches_the_request(self):
        providers = [{**p, "api_key": "k"} for p in llm_client.PROVIDERS]
        sent = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            sent[json["model"]] = json
            return _chat('{"decision": "HOLD", "reason": "r"}')

        with mock.patch.object(llm_client, "PROVIDERS", providers), \
             mock.patch.object(llm_client.requests, "post", side_effect=fake_post):
            for provider in providers:
                llm_client._call_provider(provider, "p")
        self.assertEqual(sent["qwen/qwen3.8-27b"]["reasoning_effort"], "none")
        self.assertNotIn("reasoning_effort", sent["qwen/qwen3.8-27b:free"])
        self.assertNotIn("reasoning_effort", sent["qwen3.8-max"])
        for body in sent.values():
            self.assertEqual(body["max_tokens"], llm_client.MAX_TOKENS)


# ── Health-score components ───────────────────────────────────

class TestRobustComponents(unittest.TestCase):
    def test_depth_baseline_is_median_not_mean(self):
        # One 4103-size outlier used to drag the mean to ~300 and push
        # every normal reading (~60) to ~0.2. With the median it is healthy.
        history = [55.0, 60.0, 4103.0, 58.0, 62.0, 57.0]
        self.assertEqual(hs.score_depth(30.0, 30.0, history), 1.0)

    def test_depth_single_thin_snapshot_is_smoothed(self):
        history = [200.0] * 10
        self.assertEqual(hs.score_depth(1.0, 0.5, history), 1.0)   # one thin reading != stress

    def test_depth_sustained_thin_book_scores_low(self):
        history = [200.0] * 8 + [3.0, 3.0]
        self.assertLess(hs.score_depth(1.0, 1.0, history), 0.1)

    def test_frozen_snapshot_is_not_reported_as_healthy(self):
        data = {"price": 100.0, "spread_pct": 0.05, "bid_size": 10, "ask_size": 10, "volume_24h": 1000}
        history = {"price": [100.0] * 12, "depth": [20.0] * 12, "volume": [1000.0] * 12}
        result = hs.score_stock(data, history, now=utc(2026, 9, 16, 15), fresh_signal=False)
        self.assertEqual(result["components_used"].count("depth"), 0)
        self.assertEqual(result["label"], "no_fresh_signal")

    def test_abnormal_movement_skips_returns_measured_across_long_gaps(self):
        base = utc(2026, 9, 14, 0)
        prices = [100.0, 100.1, 100.0, 100.1]
        ts = [(base + timedelta(hours=6 * i)).isoformat() for i in range(4)]
        # last reading is 30h after the previous point -> not comparable
        self.assertIsNone(hs.score_abnormal_movement(prices, 105.0, ts, base + timedelta(hours=48)))
        # same move, normal spacing -> scored as abnormal
        score = hs.score_abnormal_movement(prices, 105.0, ts, base + timedelta(hours=24))
        self.assertIsNotNone(score)
        self.assertLess(score, 0.2)

    def test_abnormal_movement_vol_floor_prevents_blowup(self):
        base = utc(2026, 9, 14, 0)
        prices = [100.0] * 5                         # perfectly flat baseline (stdev 0)
        ts = [(base + timedelta(hours=6 * i)).isoformat() for i in range(5)]
        score = hs.score_abnormal_movement(prices, 100.02, ts, base + timedelta(hours=30))
        self.assertIsNotNone(score)                  # legacy code returned None here
        self.assertGreater(score, 0.8)               # 0.02% is not abnormal

    def test_abnormal_movement_legacy_without_timestamps_still_works(self):
        score = hs.score_abnormal_movement([100, 101, 100, 101, 100], 100.5)
        self.assertIsNotNone(score)


class TestRegimeBaselines(unittest.TestCase):
    def _series(self, stamps_values):
        return [v for _, v in stamps_values], [t.isoformat() for t, _ in stamps_values]

    def test_weekend_reading_is_compared_with_weekend_history_only(self):
        weekday = [(utc(2026, 9, 10 + i, 15), 500.0) for i in range(4) if (10 + i) != 12 and (10 + i) != 13]
        weekend = [(utc(2026, 9, 12, h), 10.0) for h in (8, 12, 14)]
        values, stamps = self._series(weekday + weekend)
        now = utc(2026, 9, 19, 8)                                    # a Saturday
        self.assertEqual(hs.score_depth(5.0, 5.0, values, stamps, now), 1.0)   # 10 vs weekend median 10

    def test_too_few_same_regime_points_means_no_signal(self):
        weekday = [(utc(2026, 9, 14 + i, 15), 500.0) for i in range(4)]
        values, stamps = self._series(weekday)
        self.assertIsNone(hs.score_depth(1.0, 1.0, values, stamps, utc(2026, 9, 19, 8)))

    def test_night_and_open_are_separate_regimes(self):
        self.assertEqual(hs._regime(utc(2026, 9, 15, 15)), "open")
        self.assertEqual(hs._regime(utc(2026, 9, 15, 6)), "night")
        self.assertEqual(hs._regime(utc(2026, 9, 19, 8)), "weekend")
        self.assertEqual(hs._regime(utc(2026, 9, 7, 15)), "weekend")          # holiday

    def test_missing_timestamps_fall_back_to_whole_history(self):
        self.assertEqual(hs.score_depth(100.0, 100.0, [200.0] * 5, [None] * 5, utc(2026, 9, 19, 8)), 1.0)


class TestVolumeRemoved(unittest.TestCase):
    """volume24h is a daily counter that mirrors the US stock on weekdays - not a score input."""

    def test_weights_have_no_volume_and_sum_to_one(self):
        self.assertNotIn("volume_trend", hs.WEIGHTS)
        self.assertAlmostEqual(sum(hs.WEIGHTS.values()), 1.0)
        self.assertEqual(hs.MIN_COMPONENTS, len(hs.WEIGHTS))
        self.assertFalse(hasattr(hs, "score_volume_trend"))

    def test_score_ignores_volume_completely(self):
        history = {"price": [100.0, 100.1, 100.0, 100.1], "depth": [50.0] * 6,
                   "ts": [(utc(2026, 9, 15, 14) + timedelta(hours=2 * i)).isoformat() for i in range(4)]}
        base = {"price": 100.05, "spread_pct": 0.05, "bid_size": 25.0, "ask_size": 25.0}
        now = utc(2026, 9, 15, 15)
        with_vol = hs.score_stock({**base, "volume_24h": 5e7}, history, now)
        crazy_vol = hs.score_stock({**base, "volume_24h": 14.47}, history, now)
        no_vol = hs.score_stock(base, history, now)
        self.assertEqual(with_vol["score"], crazy_vol["score"])
        self.assertEqual(with_vol["score"], no_vol["score"])
        self.assertNotIn("volume_trend", with_vol["components_raw"])


class TestMarketCalendar(unittest.TestCase):
    def test_winter_20_utc_is_still_market_hours(self):
        # 2026-01-13 20:00 UTC = 15:00 EST -> open. Old fixed-UTC logic said "after hours".
        self.assertEqual(hs.score_weekend_afterhours(utc(2026, 1, 13, 20)), 1.0)

    def test_summer_20_utc_is_after_close(self):
        # 2026-07-14 20:00 UTC = 16:00 EDT -> just closed.
        self.assertEqual(hs.score_weekend_afterhours(utc(2026, 7, 14, 20)), 0.8)

    def test_open_at_930_et(self):
        self.assertEqual(hs.score_weekend_afterhours(utc(2026, 9, 15, 13, 29)), 0.8)
        self.assertEqual(hs.score_weekend_afterhours(utc(2026, 9, 15, 13, 30)), 1.0)

    def test_weekend_window_is_fri_2000_et_to_sun_2000_et(self):
        # September = EDT (UTC-4). Fri 20:00 ET = Sat 00:00 UTC; Sun 20:00 ET = Mon 00:00 UTC.
        self.assertEqual(hs.score_weekend_afterhours(utc(2026, 9, 18, 23, 59)), 0.8)   # Fri 19:59 ET
        self.assertEqual(hs.score_weekend_afterhours(utc(2026, 9, 19, 0, 0)), 0.6)     # Fri 20:00 ET
        self.assertEqual(hs.score_weekend_afterhours(utc(2026, 9, 21, 0, 0) - timedelta(minutes=1)), 0.6)  # Sun 19:59 ET
        self.assertEqual(hs.score_weekend_afterhours(utc(2026, 9, 21, 0, 0)), 0.8)     # Sun 20:00 ET

    def test_regime_names_follow_the_same_window(self):
        self.assertEqual(hs._regime(utc(2026, 9, 19, 0, 0)), "weekend")   # Fri 20:00 ET
        self.assertEqual(hs._regime(utc(2026, 9, 21, 0, 0)), "night")     # Sun 20:00 ET
        self.assertEqual(hs._regime(utc(2026, 9, 21, 15, 0)), "open")     # Mon 11:00 ET

    def test_holiday_window_matches_labor_day_2026_candles(self):
        # Observed: native from Fri 04 Sep 20:00 ET until Mon 07 Sep 20:00 ET (Tue 08 Sep 00:00 UTC).
        self.assertEqual(hs.score_weekend_afterhours(utc(2026, 9, 7, 0, 0)), 0.6)     # Sun 20:00 ET, evening before
        self.assertEqual(hs.score_weekend_afterhours(utc(2026, 9, 7, 23, 59)), 0.6)   # Mon 19:59 ET
        self.assertEqual(hs.score_weekend_afterhours(utc(2026, 9, 8, 0, 0)), 0.8)     # Mon 20:00 ET: overnight session is back
        self.assertEqual(hs._regime(utc(2026, 9, 7, 0, 0)), "weekend")
        self.assertEqual(hs._regime(utc(2026, 9, 8, 0, 0)), "night")

    def test_holiday_counts_as_closed_day(self):
        self.assertEqual(hs.score_weekend_afterhours(utc(2026, 9, 7, 15)), 0.6)   # Labor Day


# ── History maintenance ───────────────────────────────────────

class TestHistoryHelpers(unittest.TestCase):
    def test_migrate_backfills_timestamps_from_heartbeats(self):
        history = {"NVDA": {"volume": [1.0, 2.0, 3.0], "price": [10.0, 11.0, 12.0], "depth": [1.0] * 3}}
        beats = [{"timestamp": f"2026-09-1{i}T00:00:00+00:00", "scores": {"NVDA": 0.9}} for i in range(5)]
        beats[2]["scores"]["NVDA"] = None            # a run where NVDA's fetch failed
        out = main.migrate_history(history, beats)
        self.assertEqual(len(out["NVDA"]["ts"]), 3)
        self.assertEqual(out["NVDA"]["ts"], ["2026-09-11T00:00:00+00:00", "2026-09-13T00:00:00+00:00",
                                              "2026-09-14T00:00:00+00:00"])
        self.assertEqual(out["schema_version"], main.SCHEMA_VERSION)
        self.assertEqual(main.migrate_history(out, beats)["NVDA"]["ts"], out["NVDA"]["ts"])  # idempotent

    def test_migrate_without_matching_heartbeats_leaves_unknown(self):
        history = {"NVDA": {"volume": [1.0] * 5, "price": [10.0] * 5, "depth": [1.0] * 5}}
        out = main.migrate_history(history, [])
        self.assertEqual(out["NVDA"]["ts"], [None] * 5)
        self.assertIsNone(main.history_span_hours(out["NVDA"]))

    def test_unchanged_snapshot_detection(self):
        series = {"price": [100.0], "volume": [5.0], "depth": [20.0]}
        same = {"price": 100.0, "volume_24h": 5.0, "bid_size": 12.0, "ask_size": 8.0}
        self.assertTrue(main.is_unchanged_snapshot(series, same))
        self.assertFalse(main.is_unchanged_snapshot(series, {**same, "volume_24h": 6.0}))
        self.assertFalse(main.is_unchanged_snapshot({}, same))
        self.assertFalse(main.is_unchanged_snapshot(series, {**same, "volume_24h": None}))

    def test_update_history_records_ts_and_skips_unchanged(self):
        now = utc(2026, 9, 19, 8)
        entry = {"status": "ok", "price": 100.0, "volume_24h": 5.0, "bid_size": 1.0, "ask_size": 2.0}
        history = main.update_history({}, {"NVDA": entry, "TSLA": entry}, now, skip={"TSLA"})
        self.assertEqual(history["NVDA"]["ts"], [now.isoformat()])
        self.assertNotIn("TSLA", history)

    def test_record_scores_ignores_partial_component_scores(self):
        scores = {s: {"score": 0.95, "components_used": ["spread", "weekend"]} for s in main.STOCKS}
        scores["NVDA"] = {"score": 0.4, "components_used": ["spread", "depth", "abnormal_movement", "weekend"]}
        history = main.record_scores({}, scores)
        self.assertEqual(history["NVDA"]["score"], [0.4])
        self.assertEqual(history["TSLA"]["score"], [None])

    def test_check_confirmation(self):
        below = lambda s: s < 0.5
        hist = {"NVDA": {"score": [0.9, 0.4]}}
        self.assertTrue(main.check_confirmation("NVDA", 0.3, hist, below, "x")[0])
        hist = {"NVDA": {"score": [0.4, 0.9]}}
        ok, why = main.check_confirmation("NVDA", 0.3, hist, below, "x")
        self.assertFalse(ok)
        self.assertIn("awaiting confirmation", why)
        self.assertFalse(main.check_confirmation("NVDA", 0.3, {"NVDA": {"score": [None]}}, below, "x")[0])
        self.assertFalse(main.check_confirmation("NVDA", 0.3, {}, below, "x")[0])

    def test_validate_history_accepts_v2_and_rejects_bad_scores(self):
        good = {"NVDA": {"volume": [], "price": [1.0], "depth": [], "ts": [None], "score": [None, 0.5]}}
        main.validate_history_state(good)
        for bad in ({"score": [1.5]}, {"score": ["x"]}, {"ts": [123]}):
            with self.assertRaises(main.StateCorruptionError):
                main.validate_history_state({"NVDA": {"volume": [], "price": [], "depth": [], **bad}})


# ── Round-trip accounting ─────────────────────────────────────

class TestRoundTripMetrics(unittest.TestCase):
    def test_protection_gain_math(self):
        sold = {"quantity_sold": 1.5, "exit_price": 200.0, "sold_timestamp": "2026-09-18T00:00:00+00:00",
                "realized_pnl_usd": 5.0}
        # sold at 200, re-bought at 210 with 297 USDT -> fewer units than we started with
        qty_after = 297 * (1 - 0.0005) / 210
        trip = main.build_roundtrip_entry("NVDA", sold, 210.0, qty_after, utc(2026, 9, 19), "buy_x")
        self.assertLess(trip["protection_gain_usd"], 0)
        self.assertAlmostEqual(trip["price_change_pct_since_exit"], 5.0)
        self.assertAlmostEqual(trip["hours_out_of_position"], 24.0)
        # re-bought cheaper -> positive protection
        cheap = main.build_roundtrip_entry("NVDA", sold, 190.0, 1.6, utc(2026, 9, 19), "buy_y")
        self.assertGreater(cheap["protection_gain_usd"], 0)

    def test_roundtrip_missing_data_returns_none(self):
        self.assertIsNone(main.build_roundtrip_entry("NVDA", {}, 210.0, 1.0, utc(2026, 9, 19), "id"))


class TestPerformanceMetrics(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._cwd = os.getcwd()
        os.chdir(self._tmp.name)
        os.makedirs("data")

    def tearDown(self):
        os.chdir(self._cwd)
        self._tmp.cleanup()

    def test_sell_alone_is_an_open_exit_not_a_round_trip(self):
        main.append_jsonl(main.PERFORMANCE_LOG_PATH, {"event_id": "s1", "realized_pnl_usd": 5.0,
                                                        "realized_pnl_pct": 1.6})
        m = main.compute_performance_metrics()
        self.assertEqual(m["exit_count"], 1)
        self.assertEqual(m["open_exits"], 1)
        self.assertEqual(m["round_trips_completed"], 0)
        self.assertIsNone(m["round_trip_win_rate_pct"])

    def test_round_trip_reports_protection_gain_and_dedupes(self):
        main.append_jsonl(main.PERFORMANCE_LOG_PATH, {"event_id": "s1", "realized_pnl_usd": 5.0,
                                                        "realized_pnl_pct": 1.6})
        for _ in range(2):   # duplicated line must not double count
            main.append_jsonl(main.ROUNDTRIP_LOG_PATH, {"event_id": "b1", "protection_gain_usd": -18.0,
                                                          "protection_gain_pct": -5.7})
        m = main.compute_performance_metrics()
        self.assertEqual(m["round_trips_completed"], 1)
        self.assertEqual(m["open_exits"], 0)
        self.assertEqual(m["win_rate_pct"], 100.0)              # per-exit view looks like a win...
        self.assertEqual(m["round_trip_win_rate_pct"], 0.0)     # ...the round trip shows it was not
        self.assertEqual(m["total_protection_gain_usd"], -18.0)


# ── Fetcher retry semantics ───────────────────────────────────

class TestFetchRetries(unittest.TestCase):
    def _resp(self, status, payload=None):
        r = mock.MagicMock()
        r.status_code = status
        r.json.return_value = payload if payload is not None else {"data": [{"ts": "1"}]}
        if status >= 400:
            import requests
            r.raise_for_status.side_effect = requests.exceptions.HTTPError(f"HTTP {status}")
        return r

    def test_client_error_is_not_retried(self):
        import requests
        with mock.patch.object(fetch_bitget.SESSION, "get", return_value=self._resp(404)) as get, \
             mock.patch.object(fetch_bitget.time, "sleep") as sleep:
            with self.assertRaises(requests.exceptions.HTTPError):
                fetch_bitget.fetch_ticker("rNVDAUSDT")
        self.assertEqual(get.call_count, 1)
        sleep.assert_not_called()

    def test_server_error_is_retried_then_raised(self):
        import requests
        with mock.patch.object(fetch_bitget.SESSION, "get", return_value=self._resp(503)) as get, \
             mock.patch.object(fetch_bitget.time, "sleep"):
            with self.assertRaises(requests.exceptions.HTTPError):
                fetch_bitget.fetch_ticker("rNVDAUSDT")
        self.assertEqual(get.call_count, fetch_bitget.MAX_RETRIES)

    def test_unexpected_payload_shapes_do_not_crash(self):
        for payload in ({"data": None}, {"data": {"ts": "1"}}, {"data": []}, []):
            with mock.patch.object(fetch_bitget.SESSION, "get", return_value=self._resp(200, payload)):
                fetch_bitget.fetch_ticker("rNVDAUSDT")   # must not raise

    def test_raw_diagnostics_whitelist(self):
        raw = {"ts": "1", "lastPrice": "2", "bid1Size": "3", "ask1Size": "4", "volume24h": "5",
               "turnover24h": "6", "openPrice24h": "7", "someQuoteVolume": "8"}
        out = fetch_bitget.raw_diagnostics(raw)
        self.assertIn("volume24h", out)
        self.assertIn("turnover24h", out)
        raw.update({"bid1Price": "9", "ask1Price": "10"})
        out = fetch_bitget.raw_diagnostics(raw)
        self.assertIn("bid1Price", out)          # spread must be reconstructable from the log
        self.assertIn("ask1Price", out)
        self.assertNotIn("openPrice24h", out)


# ── End-to-end behaviour of run() ─────────────────────────────

class TestRunFixes(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._cwd = os.getcwd()
        os.chdir(self._tmp.name)
        os.makedirs("data")
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        self._clock = mock.patch("main.datetime", FixedDatetime)
        self._clock.start()

    def tearDown(self):
        self._clock.stop()
        self._env.stop()
        os.chdir(self._cwd)
        self._tmp.cleanup()

    # fixtures ---------------------------------------------------
    def _positions(self, nvda="held", entered=None, sold_hours_ago=48, cash=100.0):
        pos = {"USDT": {"balance_usdt": cash}}
        for s in main.STOCKS:
            pos[s] = {"status": "held", "entry_price": 100.0, "exit_price": None,
                      "quantity": 3.0, "cost_basis_usd": 300.0}
        if nvda == "held":
            pos["NVDA"] = {"status": "held", "entry_price": 200.0, "exit_price": None,
                           "quantity": 1.5, "cost_basis_usd": 300.0}
            if entered is not None:
                pos["NVDA"]["entered_timestamp"] = entered
        else:
            sold_at = FIXED_NOW - timedelta(hours=sold_hours_ago)
            pos["NVDA"] = {"status": "sold", "entry_price": 200.0, "exit_price": 190.0,
                           "quantity_sold": 1.5, "proceeds_usd": 285.0, "realized_pnl_usd": -15.0,
                           "sold_timestamp": sold_at.isoformat()}
            main.append_jsonl(main.TRANSACTION_LOG_PATH, {
                "event_id": "sell_NVDA_seed", "timestamp": sold_at.isoformat(), "instrument": "NVDA",
                "direction": "SELL", "price": 190.0, "quantity": 1.5, "balance_change": 285.0})
        return pos

    def _history(self, scores, stressed=False, n=main.MIN_HISTORY_POINTS):
        depth = [200.0 + (i % 3) for i in range(n)]
        if stressed:
            depth[-2:] = [4.0, 4.0]
        return {"NVDA": {"volume": [50e6 + (i % 3) * 1000 for i in range(n)],
                         "price": [200.0 + (i % 2) * 0.1 for i in range(n)], "depth": depth,
                         "ts": open_market_stamps(n), "score": scores}}

    def _healthy(self):
        return {"underlying": "NVDA", "symbol": "rNVDAUSDT", "status": "ok", "price": 200.05,
                "bid": 199.9, "ask": 200.1, "bid_size": 100.0, "ask_size": 100.0,
                "volume_24h": 50e6, "spread_pct": 0.1}

    def _stressed(self):
        return {**self._healthy(), "bid": 198.0, "ask": 202.0, "bid_size": 2.0, "ask_size": 2.0,
                "volume_24h": 5e6, "spread_pct": 2.0}

    def _write(self, history, positions):
        main.save_json(main.HISTORY_PATH, history)
        main.save_json(main.POSITIONS_PATH, positions)

    def _run(self, entry, llm_text):
        reply = {"success": True, "content": llm_text, "provider_used": "test", "attempts": []}
        with mock.patch("main.fetch_bitget_data", return_value=[entry]), \
             mock.patch("main.call_llm", return_value=reply) as llm:
            main.run()
        return llm

    def _last_event(self):
        return main.read_jsonl(main.EVENT_LOG_PATH)[-1]

    def _positions_now(self):
        return main.load_json(main.POSITIONS_PATH, None)

    SELL_JSON = '{"decision": "SELL", "reason": "spread and volume both degraded"}'
    BUY_JSON = '{"decision": "BUY_BACK", "reason": "components recovered"}'

    # sell path ----------------------------------------------------
    def test_sell_executes_and_logs_short_reason(self):
        self._write(self._history([0.3, 0.3], stressed=True), self._positions())
        llm = self._run(self._stressed(), self.SELL_JSON)
        llm.assert_called_once()                      # one call: sold token is NOT re-evaluated for buy-back
        pos = self._positions_now()
        self.assertEqual(pos["NVDA"]["status"], "sold")
        self.assertGreater(pos["USDT"]["balance_usdt"], 100.0)
        event = self._last_event()
        self.assertEqual(event["type"], "sell")
        self.assertEqual(event["llm_reasoning"], "spread and volume both degraded")
        self.assertEqual(len(main.read_jsonl(main.TRANSACTION_LOG_PATH)), 1)
        self.assertEqual(len(main.read_jsonl(main.PERFORMANCE_LOG_PATH)), 1)
        self.assertEqual(main.read_jsonl(main.ROUNDTRIP_LOG_PATH), [])   # not a round trip yet

    # deterministic gates skip the LLM ---------------------------
    def test_score_above_ceiling_does_not_consult_llm(self):
        self._write(self._history([0.9, 0.9]), self._positions())
        llm = self._run(self._healthy(), self.SELL_JSON)
        llm.assert_not_called()
        self.assertEqual(self._positions_now()["NVDA"]["status"], "held")
        self.assertIn("blocked by hard rule", self._last_event()["action_taken"])

    def test_single_bad_reading_needs_confirmation(self):
        self._write(self._history([0.9, 0.9], stressed=True), self._positions())
        llm = self._run(self._stressed(), self.SELL_JSON)
        llm.assert_not_called()
        self.assertEqual(self._positions_now()["NVDA"]["status"], "held")
        self.assertIn("awaiting confirmation", self._last_event()["action_taken"])

    def test_minimum_hold_after_reentry(self):
        entered = (FIXED_NOW - timedelta(hours=1)).isoformat()
        self._write(self._history([0.3, 0.3], stressed=True), self._positions(entered=entered))
        llm = self._run(self._stressed(), self.SELL_JSON)
        llm.assert_not_called()
        self.assertIn("minimum hold", self._last_event()["action_taken"])

    def test_unusable_llm_reply_holds_and_says_so(self):
        self._write(self._history([0.3, 0.3], stressed=True), self._positions())
        with mock.patch("main.fetch_bitget_data", return_value=[self._stressed()]), \
             mock.patch("main.call_llm", return_value={"success": False, "content": None,
                                                        "provider_used": None,
                                                        "attempts": [{"provider": "a", "status": "unusable_response",
                                                                      "error": "truncated"}]}):
            main.run()
        self.assertEqual(self._positions_now()["NVDA"]["status"], "held")
        self.assertIn("no usable LLM reply", self._last_event()["action_taken"])

    # buy-back / round trip --------------------------------------
    def test_buyback_completes_round_trip_and_resizes_to_own_proceeds(self):
        self._write(self._history([0.95, 0.95]), self._positions(nvda="sold", cash=285.0))
        llm = self._run(self._healthy(), self.BUY_JSON)
        llm.assert_called_once()
        pos = self._positions_now()
        self.assertEqual(pos["NVDA"]["status"], "held")
        self.assertEqual(pos["NVDA"]["cost_basis_usd"], 285.0)     # own proceeds, not a flat $300
        self.assertEqual(pos["USDT"]["balance_usdt"], 0.0)
        self.assertIn("entered_timestamp", pos["NVDA"])
        trips = main.read_jsonl(main.ROUNDTRIP_LOG_PATH)
        self.assertEqual(len(trips), 1)
        self.assertEqual(trips[0]["exit_price"], 190.0)
        summary = main.load_json(main.PERFORMANCE_SUMMARY_PATH, None)
        self.assertEqual(summary["round_trips_completed"], 1)

    def test_buyback_cooldown(self):
        self._write(self._history([0.95, 0.95]), self._positions(nvda="sold", sold_hours_ago=1.0))
        llm = self._run(self._healthy(), self.BUY_JSON)
        llm.assert_not_called()
        self.assertIn("cooldown", self._last_event()["action_taken"])

    # state / migration / diagnostics ----------------------------
    def test_legacy_history_is_migrated_and_backfilled_from_heartbeats(self):
        n = 15
        stamps = open_market_stamps(n)
        beats = [{"timestamp": stamps[i], "scores": {"NVDA": 0.9}, "labels": {}} for i in range(n)]
        for b in beats:
            main.append_jsonl(main.HEARTBEAT_LOG_PATH, b)
        legacy = {"NVDA": {"volume": [50e6] * n, "price": [200.0 + (i % 2) * 0.1 for i in range(n)],
                           "depth": [200.0] * n}}
        self._write(legacy, self._positions())
        self._run(self._healthy(), self.SELL_JSON)
        h = main.load_json(main.HISTORY_PATH, None)
        self.assertEqual(h["schema_version"], main.SCHEMA_VERSION)
        self.assertEqual(len(h["NVDA"]["ts"]), len(h["NVDA"]["price"]))
        self.assertEqual(h["NVDA"]["ts"][0], beats[0]["timestamp"])
        self.assertIsNotNone(main.history_span_hours(h["NVDA"]))

    def test_frozen_snapshot_is_flagged_and_not_appended(self):
        hist = self._history([0.9, 0.9])
        last = {"price": hist["NVDA"]["price"][-1], "volume_24h": hist["NVDA"]["volume"][-1],
                "bid_size": hist["NVDA"]["depth"][-1], "ask_size": 0.0}
        entry = {**self._healthy(), **last}
        self._write(hist, self._positions())
        self._run(entry, self.SELL_JSON)
        beat = main.read_jsonl(main.HEARTBEAT_LOG_PATH)[-1]
        self.assertEqual(beat["unchanged_snapshot"], ["NVDA"])
        self.assertEqual(beat["labels"]["NVDA"], "no_fresh_signal")
        self.assertEqual(len(main.load_json(main.HISTORY_PATH, None)["NVDA"]["price"]), len(hist["NVDA"]["price"]))

    def test_heartbeat_records_fetch_status_and_mode(self):
        self._write(self._history([0.9, 0.9]), self._positions())
        self._run(self._healthy(), self.SELL_JSON)
        beat = main.read_jsonl(main.HEARTBEAT_LOG_PATH)[-1]
        self.assertEqual(beat["fetch_status"]["NVDA"], "ok")
        self.assertEqual(beat["fetch_status"]["TSLA"], "error")

    def test_raw_ticker_log_is_written_and_capped(self):
        self._write(self._history([0.9, 0.9]), self._positions())
        entry = {**self._healthy(), "raw_diagnostics": {"volume24h": "5", "ts": "1"}}
        for _ in range(main.RAW_TICKER_LOG_RUNS + 5):
            self._run(entry, self.SELL_JSON)
        log = main.load_json(main.RAW_TICKER_LOG_PATH, None)
        self.assertEqual(len(log), main.RAW_TICKER_LOG_RUNS)
        self.assertEqual(log[-1]["tickers"]["NVDA"]["volume24h"], "5")

    def test_event_ids_do_not_collide_within_a_second(self):
        a = main._event_id("sell", "NVDA", utc(2026, 9, 19, 8, 0, 0, 1))
        b = main._event_id("sell", "NVDA", utc(2026, 9, 19, 8, 0, 0, 2))
        self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main()

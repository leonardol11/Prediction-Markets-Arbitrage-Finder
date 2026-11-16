"""Edge-first pipeline: depth pricing, staleness, LLM verification, and the cache.

Only the network boundary is faked (exchange calls and Groq's HTTP endpoint).
"""

import json
import os
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from matcher import core
from matcher.config import ScanConfig
from matcher.core import (Reporter, best_quote, find_arbitrage, make_book, pair_id,
                          quote_state, run_scan, verify_matches_with_llm)
from matcher.sheets import build_sheets, build_summary
from matcher.verify_cache import VerifyCache

END = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
QUIET = Reporter(on_log=lambda _m: None)


def poly_market(n=1, **extra):
    return {"id": f"p{n}", "conditionId": f"0x{n}", "question": f"Will event {n} happen?",
            "description": f"Resolves Yes if event {n} happens by the end date.",
            "outcomes": '["Yes", "No"]', "endDate": END, "volume": "50000",
            "clobTokenIds": json.dumps([f"yes{n}", f"no{n}"]),
            "feesEnabled": False, **extra}


def kalshi_market(n=1, **extra):
    return {"ticker": f"EVT-{n}", "title": f"Will event {n} happen?", "status": "active",
            "rules_primary": f"If event {n} happens, the market resolves to Yes.",
            "close_time": END, "expected_expiration_time": END, "volume_fp": "50000",
            "yes_ask_dollars": "0.60", "no_ask_dollars": "0.45",
            "yes_ask_size_fp": "100", "yes_bid_size_fp": "100",
            "_fee_type": "quadratic", "_fee_multiplier": 0, **extra}


class _Reply:
    def __init__(self, status=200, content=None, headers=None):
        self.status_code = status
        self.headers = {"x-ratelimit-limit-tokens": "100000",
                        "x-ratelimit-remaining-tokens": "100000", **(headers or {})}
        self._content = content
        self.text = "error"

    def json(self):
        return {"choices": [{"message": {"content": json.dumps(self._content)}}],
                "usage": {"total_tokens": 100}}


def groq(decide, log=None):
    """A fake Groq endpoint. `decide(kind, index, block)` returns a row or None."""
    def post(_url, headers=None, json=None, timeout=None):
        prompt = json["messages"][0]["content"]
        kind = "strict" if "SAME bet" in prompt else "screen"
        blocks = prompt.split("Pair ")[1:]
        if log is not None:
            log.append((kind, len(blocks)))
        rows = []
        for i, block in enumerate(blocks):
            row = decide(kind, i, block)
            if row is not None:
                rows.append({"pair_index": i, **row})
        return _Reply(content=rows)
    return post


YES = {"match": True, "confidence": 0.9, "shared_keywords": ["event"]}
NO = {"match": False, "confidence": 0.1}


class DepthPricingTests(unittest.TestCase):
    def quote(self, poly_yes, kalshi_no, poly=None, kalshi=None, ts=None):
        poly, kalshi = poly or poly_market(), kalshi or kalshi_market()
        pob = make_book(yes=poly_yes, no=[(0.99, 1000)], ts=ts)
        kob = make_book(yes=[(0.99, 1000)], no=kalshi_no, ts=ts)
        return best_quote(poly, kalshi, pob, kob, ScanConfig())

    def test_takes_every_profitable_level_and_stops_at_break_even(self):
        # Fee-free. 10 @ 40¢+50¢ (10¢ edge), then 20 @ 44¢+50¢ (6¢), then a
        # 52¢ level that would lose money and must not be taken.
        q = self.quote([(0.40, 10), (0.44, 20), (0.52, 500)], [(0.50, 1000)])
        self.assertAlmostEqual(q["max_contracts"], 30)
        self.assertAlmostEqual(q["max_profit"], 10 * 0.10 + 20 * 0.06)
        self.assertAlmostEqual(q["margin"], 2.2 / 30)
        self.assertAlmostEqual(q["a_price"], (10 * 0.40 + 20 * 0.44) / 30)
        self.assertAlmostEqual(q["a_best"], 0.40)
        self.assertEqual(q["levels"], 2)

    def test_size_is_limited_by_the_thinner_book(self):
        q = self.quote([(0.40, 500)], [(0.50, 7), (0.55, 5), (0.70, 900)])
        self.assertAlmostEqual(q["max_contracts"], 12)       # 7 @ 50¢ + 5 @ 55¢
        self.assertAlmostEqual(q["max_profit"], 7 * 0.10 + 5 * 0.05)

    def test_deeper_levels_pay_their_own_fees(self):
        # Standard Kalshi fee and a 4% Polymarket fee: the second level's 1¢
        # gross edge is less than its fees, so only the first level is taken.
        poly = poly_market(feesEnabled=True, feeSchedule={"rate": 0.04})
        kalshi = kalshi_market(_fee_multiplier=1)
        q = self.quote([(0.40, 100), (0.49, 100)], [(0.50, 1000)], poly, kalshi)
        self.assertAlmostEqual(q["max_contracts"], 100)
        self.assertAlmostEqual(q["fees_per_pair"], 0.04 * 0.4 * 0.6 + 0.0175)
        self.assertAlmostEqual(q["margin"], 0.10 - 0.0096 - 0.0175)

    def test_order_below_polymarket_minimum_is_not_executable(self):
        q = self.quote([(0.40, 3)], [(0.50, 1000)], poly_market(orderMinSize=5))
        self.assertGreater(q["margin"], 0)
        self.assertFalse(q["executable"])
        self.assertEqual(quote_state(q, ScanConfig()), "small")

    def test_near_miss_is_flagged_within_the_threshold(self):
        q = self.quote([(0.51, 100)], [(0.50, 100)])           # 1¢ short
        self.assertEqual(quote_state(q, ScanConfig(near_margin=0.02)), "near")
        self.assertEqual(quote_state(q, ScanConfig(near_margin=0.005)), "none")


class LiveArbitrageTests(unittest.TestCase):
    def setUp(self):
        self.poly, self.kalshi = poly_market(), kalshi_market()
        self.pairs = [(self.poly, self.kalshi, {"event"}, 0.9)]

    def arbs(self, ts=None, cfg=None):
        pob = {0: make_book(yes=[(0.40, 50)], no=[(0.99, 50)], ts=ts)}
        kob = {0: make_book(yes=[(0.99, 50)], no=[(0.50, 50)], ts=ts)}
        return find_arbitrage(self.pairs, [self.poly], [self.kalshi], pob, kob,
                              cfg or ScanConfig(), QUIET)

    def test_fresh_quotes_are_live(self):
        (arb,) = self.arbs()
        self.assertFalse(arb["stale"])
        self.assertAlmostEqual(arb["max_profit"], 5.0)

    def test_stale_quotes_are_excluded(self):
        self.assertEqual(self.arbs(ts=time.time() - 120), [])
        self.assertEqual(len(self.arbs(ts=time.time() - 120,
                                       cfg=ScanConfig(stale_quote_seconds=600))), 1)

    def test_one_stale_leg_makes_the_pair_stale(self):
        pob = {0: make_book(yes=[(0.40, 50)], no=[(0.99, 50)])}
        kob = {0: make_book(yes=[(0.99, 50)], no=[(0.50, 50)], ts=time.time() - 120)}
        self.assertEqual(find_arbitrage(self.pairs, [self.poly], [self.kalshi],
                                        pob, kob, ScanConfig(), QUIET), [])


class LlmVerificationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ScanConfig(groq_api_key="k", use_verify_cache=False,
                              groq_batch_size=4, llm_budget_seconds=0)
        self.pairs = [(poly_market(n), kalshi_market(n), set(), 0.9) for n in range(10)]
        patcher = patch.object(core, "GROQ_BACKOFF_BASE", 0.01)
        patcher.start()
        self.addCleanup(patcher.stop)

    def keys(self, pairs):
        return {pair_id(p, k) for p, k, _s, _c in pairs}

    def test_required_pairs_get_the_strict_check_in_bounded_batches(self):
        log = []
        with patch("matcher.core.requests.post", side_effect=groq(lambda *_: YES, log)):
            verdicts = verify_matches_with_llm(self.pairs, [], self.cfg, QUIET)
        self.assertEqual(set(verdicts), self.keys(self.pairs))
        self.assertTrue(all(v["match"] for v in verdicts.values()))
        self.assertEqual(sorted(log), [("strict", 2), ("strict", 4), ("strict", 4)])

    def test_title_screen_rejects_before_the_strict_check(self):
        log = []

        def decide(kind, i, _block):
            if kind == "screen":
                return {"match": i < 3}
            return YES if i == 0 else NO

        with patch("matcher.core.requests.post", side_effect=groq(decide, log)):
            verdicts = verify_matches_with_llm([], self.pairs, self.cfg, QUIET)
        self.assertEqual(len(verdicts), 10)
        self.assertEqual(sum(v["match"] for v in verdicts.values()), 1)
        self.assertEqual(log, [("screen", 10), ("strict", 3)])

    def test_failed_calls_never_become_matches(self):
        for failure in (_Reply(status=500), _Reply(status=400),
                        core.requests.Timeout("slow")):
            with patch("matcher.core.requests.post", side_effect=[failure] * 20), \
                    patch("matcher.core._sleep"):
                verdicts = verify_matches_with_llm(self.pairs, self.pairs, self.cfg, QUIET)
            self.assertEqual(verdicts, {})

    def test_rate_limit_backs_off_exponentially_then_succeeds(self):
        ok, calls, pauses = groq(lambda *_: YES), [], []

        def post(*args, **kwargs):
            calls.append(1)
            if len(calls) <= 3:
                return _Reply(status=429, headers={"retry-after": "0"})
            return ok(*args, **kwargs)

        real_block = core._GroqLimiter.block

        def block(self, seconds):
            pauses.append(seconds)
            real_block(self, seconds)

        with patch("matcher.core.requests.post", side_effect=post), \
                patch.object(core._GroqLimiter, "block", block):
            verdicts = verify_matches_with_llm(self.pairs[:2], [], self.cfg, QUIET)
        self.assertEqual(len(verdicts), 2)
        self.assertEqual(len(pauses), 3)
        # 1×, 2×, 4× the base, each with up to 25% jitter.
        for pause, factor in zip(pauses, (1, 2, 4)):
            self.assertTrue(0.01 * factor <= pause <= 0.0125 * factor, pauses)

    def test_rate_limit_that_never_clears_leaves_pairs_unverified(self):
        with patch("matcher.core.requests.post",
                   return_value=_Reply(status=429, headers={"retry-after": "0"})):
            self.assertEqual(verify_matches_with_llm(self.pairs[:2], [], self.cfg, QUIET), {})

    def test_pairs_missing_from_the_reply_stay_unverified(self):
        decide = lambda _kind, i, _block: YES if i == 0 else None
        with patch("matcher.core.requests.post", side_effect=groq(decide)):
            verdicts = verify_matches_with_llm(self.pairs[:4], [], self.cfg, QUIET)
        self.assertEqual(set(verdicts), self.keys(self.pairs[:1]))

    def test_malformed_verdicts_are_ignored(self):
        bad = {"match": "yes", "confidence": 0.99}            # not a boolean
        with patch("matcher.core.requests.post", side_effect=groq(lambda *_: bad)):
            self.assertEqual(verify_matches_with_llm(self.pairs[:2], [], self.cfg, QUIET), {})

    def test_spent_budget_leaves_pairs_unverified(self):
        log = []
        cfg = ScanConfig(groq_api_key="k", use_verify_cache=False,
                         llm_budget_seconds=0.0001)
        with patch("matcher.core.requests.post", side_effect=groq(lambda *_: YES, log)), \
                patch.object(core._GroqLimiter, "acquire", return_value=0):
            verdicts = verify_matches_with_llm(self.pairs[:2], self.pairs[2:], cfg, QUIET)
        self.assertEqual((verdicts, log), ({}, []))

    def test_rejected_key_stops_without_matches(self):
        with patch("matcher.core.requests.post", return_value=_Reply(status=401)) as post:
            self.assertEqual(verify_matches_with_llm(self.pairs, self.pairs, self.cfg, QUIET), {})
        self.assertLessEqual(post.call_count, self.cfg.groq_concurrency)


class VerifyCacheTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, "cache.json")
        self.poly, self.kalshi = poly_market(), kalshi_market()

    def cache(self, **kw):
        return VerifyCache.load(self.path, kw.pop("ttl", 30), "model-a", "1", **kw)

    def test_round_trip_through_disk(self):
        cache = self.cache()
        cache.put(self.poly, self.kalshi, True, 0.9, {"event"})
        self.assertTrue(cache.save())
        entry = self.cache().get(self.poly, self.kalshi)
        self.assertTrue(entry["match"])
        self.assertEqual(entry["shared"], ["event"])

    def test_price_changes_do_not_invalidate(self):
        cache = self.cache()
        cache.put(self.poly, self.kalshi, True, 0.9)
        moved = dict(self.kalshi, yes_ask_dollars="0.10", volume_fp="999999")
        self.assertIsNotNone(cache.get(dict(self.poly, volume="1"), moved))

    def test_rule_outcome_or_deadline_changes_invalidate(self):
        cache = self.cache()
        cache.put(self.poly, self.kalshi, True, 0.9)
        changed = [
            (dict(self.poly, description="Resolves Yes only if confirmed by Reuters."), self.kalshi),
            (dict(self.poly, outcomes='["Up", "Down"]'), self.kalshi),
            (dict(self.poly, endDate="2031-01-01T00:00:00Z"), self.kalshi),
            (dict(self.poly, resolutionSource="https://example.com"), self.kalshi),
            (self.poly, dict(self.kalshi, rules_primary="If event 1 happens twice, Yes.")),
            (self.poly, dict(self.kalshi, close_time="2031-01-01T00:00:00Z")),
            (self.poly, dict(self.kalshi, yes_sub_title="Something else")),
        ]
        for poly, kalshi in changed:
            self.assertIsNone(cache.get(poly, kalshi))

    def test_model_prompt_and_age_invalidate(self):
        cache = self.cache()
        cache.put(self.poly, self.kalshi, True, 0.9)
        cache.save()
        other_model = VerifyCache.load(self.path, 30, "model-b", "1")
        self.assertIsNone(other_model.get(self.poly, self.kalshi))
        other_prompt = VerifyCache.load(self.path, 30, "model-a", "2")
        self.assertIsNone(other_prompt.get(self.poly, self.kalshi))
        cache.entries[next(iter(cache.entries))]["ts"] -= 40 * 86400
        self.assertIsNone(cache.get(self.poly, self.kalshi))

    def test_browser_seed_is_validated_and_merged(self):
        donor = self.cache()
        donor.put(self.poly, self.kalshi, True, 0.9)
        seed = donor.export([(self.poly, self.kalshi)])
        seed["junk"] = {"fp": 1, "match": "yes"}
        seeded = VerifyCache.load("", 30, "model-a", "1", seed=seed)
        self.assertEqual(len(seeded.entries), 1)
        self.assertTrue(seeded.get(self.poly, self.kalshi)["match"])

    def test_damaged_file_is_an_empty_cache(self):
        with open(self.path, "w") as handle:
            handle.write("{not json")
        self.assertEqual(self.cache().entries, {})


class ScanFlowTests(unittest.TestCase):
    """run_scan end to end with the exchanges and Groq faked."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        # Pair 1 is a real arb (50¢ + 45¢). Pair 2 is priced at break-even.
        self.poly = [poly_market(1), poly_market(2)]
        self.kalshi = [kalshi_market(1), kalshi_market(2)]
        self.poly_asks = {"yes1": [(0.50, 50)], "no1": [(0.70, 50)],
                          "yes2": [(0.55, 50)], "no2": [(0.56, 50)]}
        self.kalshi_calls, self.groq_log = [], []
        self.decide = lambda *_: YES

    def cfg(self, **kw):
        return ScanConfig(groq_api_key="k", llm_budget_seconds=0,
                          verify_cache_path=os.path.join(self.dir.name, "c.json"), **kw)

    def scan(self, cfg=None):
        def poly_books(tokens):
            return {t: self.poly_asks[t] for t in tokens}

        def kalshi_book(ticker):
            self.kalshi_calls.append(ticker)
            return make_book(yes=[(0.60, 100)], no=[(0.45, 100)])

        def pairs(poly, kalshi, _cfg, _rep):
            return [(p, k, {"event"}, 0.9) for p, k in zip(poly, kalshi)]

        post = groq(lambda *a: self.decide(*a), self.groq_log)
        with patch.object(core, "fetch_poly_markets", return_value=self.poly), \
                patch.object(core, "fetch_kalshi_markets", return_value=self.kalshi), \
                patch.object(core, "find_matches", side_effect=pairs), \
                patch.object(core, "fetch_poly_orderbooks", side_effect=poly_books), \
                patch.object(core, "fetch_kalshi_orderbook", side_effect=kalshi_book), \
                patch.object(core, "fetch_kalshi_series_fees"), \
                patch("matcher.core.requests.post", side_effect=post):
            return run_scan(cfg or self.cfg(), QUIET)

    def test_verified_pair_with_an_edge_is_live_arbitrage(self):
        res = self.scan()
        self.assertEqual(len(res.verified), 2)
        (arb,) = res.arbs
        self.assertIs(arb["poly"], self.poly[0])
        self.assertAlmostEqual(arb["margin"], 0.05)            # 1 − 0.50 − 0.45
        self.assertAlmostEqual(arb["max_contracts"], 50)
        for stage in ("fetch", "match", "orderbooks", "llm", "confirm", "total"):
            self.assertIn(stage, res.timings)

    def test_unverified_edge_is_never_arbitrage(self):
        no_key = self.cfg()
        no_key.groq_api_key = ""                               # LLM on, but no key
        for cfg in (self.cfg(use_llm=False), no_key):
            res = self.scan(cfg)
            self.assertEqual((res.verified, res.arbs), ([], []))
            sheets = {s["key"]: s for s in build_sheets(res)}
            self.assertEqual(sheets["arbitrage"]["rows"], [])
            status = sheets["matches"]["columns"].index(
                next(c for c in sheets["matches"]["columns"] if c["key"] == "status"))
            self.assertEqual(sheets["matches"]["rows"][0][status], "Unverified · edge")
            self.assertEqual(build_summary(res)["pending_count"], 2)
        self.assertEqual(self.groq_log, [])

    def test_rejected_or_failed_verification_is_never_arbitrage(self):
        self.decide = lambda *_: NO
        self.assertEqual(self.scan(self.cfg()).arbs, [])
        with patch.object(core, "_groq_json", return_value=None):
            res = self.scan(ScanConfig(groq_api_key="k", use_verify_cache=False))
        self.assertEqual((res.verified, res.arbs, res.pair_status), ([], [], {}))

    def test_low_confidence_match_is_not_verified(self):
        self.decide = lambda *_: {"match": True, "confidence": 0.4}
        res = self.scan()
        self.assertEqual((res.verified, res.arbs), ([], []))

    def test_edge_pairs_are_checked_before_the_rest(self):
        res = self.scan()
        # Pair 1 has a plausible edge: strict check alone. Pair 2 is screened first.
        self.assertEqual(self.groq_log, [("strict", 1), ("screen", 1), ("strict", 1)])
        self.assertEqual(res.llm_stats["new_verdicts"], 2)

    def test_second_scan_reuses_verdicts_and_refreshes_prices(self):
        self.scan()
        calls = len(self.groq_log)
        self.poly_asks["yes1"] = [(0.58, 50)]                  # the edge is gone
        res = self.scan()
        self.assertEqual(len(self.groq_log), calls)            # no new LLM calls
        self.assertEqual(res.llm_stats["cache_hits"], 2)
        self.assertEqual(len(res.verified), 2)                 # still listed
        self.assertEqual(res.arbs, [])                         # but repriced

    def test_changed_rules_are_verified_again(self):
        self.scan()
        calls = len(self.groq_log)
        self.kalshi[0]["rules_primary"] = "If event 1 happens before noon, Yes."
        res = self.scan()
        self.assertEqual(res.llm_stats["cache_hits"], 1)
        self.assertEqual(len(self.groq_log), calls + 1)

    def test_only_verified_pairs_get_kalshi_books(self):
        self.decide = lambda kind, i, block: YES if "event 1" in block else NO
        res = self.scan()
        self.assertEqual(set(self.kalshi_calls), {"EVT-1"})
        self.assertEqual([m["ticker"] for m in res.live_kalshi], ["EVT-1"])

    def test_browser_seed_replaces_a_missing_cache_file(self):
        first = self.scan()
        calls = len(self.groq_log)
        cfg = ScanConfig(groq_api_key="k", verify_cache_seed=first.verify_cache,
                         verify_cache_path=os.path.join(self.dir.name, "other.json"))
        res = self.scan(cfg)
        self.assertEqual(len(self.groq_log), calls)
        self.assertEqual(len(res.verified), 2)


if __name__ == "__main__":
    unittest.main()

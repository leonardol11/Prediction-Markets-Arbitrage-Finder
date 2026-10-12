"""Stateless scan / refresh / export flow, with only the exchange boundary faked."""

import gzip
import io
import json
import os
import time
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from openpyxl import load_workbook

import app as web
from matcher.config import ScanConfig
from matcher.core import Reporter, ScanResult, find_arbitrage


def parse_sse(text):
    events = []
    for block in text.split("\n\n"):
        event, data = None, ""
        for line in block.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data += line[5:].strip()
        if event:
            events.append((event, json.loads(data)))
    return events


class StatelessFlowTests(unittest.TestCase):
    def setUp(self):
        self.client = web.app.test_client()
        cfg = ScanConfig(use_llm=False, groq_api_key="secret-not-for-client")
        result = ScanResult(cfg)
        poly = {"question": "Will the example event happen?", "id": "p1",
                "slug": "example", "clobTokenIds": '["yes", "no"]'}
        kalshi = {"title": poly["question"], "ticker": "EXAMPLE-YES"}
        unmatched = {"question": "Unrelated market", "id": "p2", "slug": "other"}
        result.poly = [poly, unmatched]
        result.matched_poly = [poly]
        result.kalshi = result.matched_kalshi = [kalshi]
        result.matches = [(poly, kalshi, {"example", "event"}, 0.9)]
        result.poly_ob, result.kalshi_ob = self.books(0.4)
        result.prices_updated_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        result.arbs = find_arbitrage(result.matches, result.matched_poly,
                                    result.matched_kalshi, result.poly_ob,
                                    result.kalshi_ob, cfg, Reporter())
        result.finished_at = datetime.now(timezone.utc)
        with patch.object(web, "run_scan", return_value=result):
            response = self.client.post("/api/scan", json={})
        self.assertEqual(response.status_code, 200)
        self.raw_stream = response.get_data(as_text=True)
        self.events = parse_sse(self.raw_stream)
        self.snapshot = dict(self.events)["done"]

    @staticmethod
    def books(yes_price):
        return ({0: {"yes": (yes_price, 20), "no": (0.8, 10)}},
                {0: ((0.7, 15), (0.5, 12))})

    def refresh(self, books, context=None, gzip_body=False):
        body = json.dumps({"context": context or self.snapshot["context"]}).encode()
        headers = {"Content-Type": "application/json"}
        if gzip_body:
            body, headers["X-Content-Encoding"] = gzip.compress(body), "gzip"
        with patch("matcher.core.fetch_all_orderbooks", return_value=books):
            return self.client.post("/api/refresh", data=body, headers=headers)

    def export(self, sheets):
        response = self.client.post("/api/workbook", json={"sheets": sheets, "stamp": "2026-01-01"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("market-arbitrage-2026-01-01.xlsx",
                      response.headers["Content-Disposition"])
        book = load_workbook(io.BytesIO(response.data), read_only=True)
        try:
            return any("Will the example event happen?" in row
                       for row in book["Arbitrage"].iter_rows(values_only=True))
        finally:
            book.close()

    def test_scan_streams_results_and_ends(self):
        names = [name for name, _ in self.events]
        self.assertEqual(names[0], "start")
        self.assertEqual(names[-2:], ["done", "end"])
        self.assertEqual(self.events[-1][1]["status"], "done")
        self.assertEqual(self.snapshot["summary"]["arb_count"], 1)
        self.assertTrue(self.snapshot["refresh_available"])

    def test_context_is_compact_and_has_no_secrets(self):
        self.assertNotIn("secret-not-for-client", self.raw_stream)
        self.assertNotIn("groq_api_key", self.raw_stream)
        context = self.snapshot["context"]
        self.assertEqual(len(context["poly"]), 1)     # unmatched market is dropped
        self.assertEqual(context["poly_count"], 2)

    def test_repricing_removes_opportunities_and_rebuilds_export(self):
        self.assertTrue(self.export(self.snapshot["sheets"]))
        response = self.refresh(self.books(0.6))
        self.assertEqual(response.status_code, 200)
        after = response.get_json()
        self.assertEqual([s["key"] for s in after["sheets"]], list(web.REPRICED_SHEETS))
        self.assertEqual(after["summary"]["arb_count"], 0)
        self.assertEqual(after["summary"]["poly_count"], 2)
        self.assertEqual(after["sheets"][0]["rows"], [])
        self.assertEqual(after["summary"]["finished_at"],
                         self.snapshot["summary"]["finished_at"])
        self.assertNotEqual(after["prices_updated_at"], self.snapshot["prices_updated_at"])
        self.assertFalse(self.export(after["sheets"]))

    def test_refreshed_context_refreshes_again_with_gzip_body(self):
        first = self.refresh(self.books(0.6)).get_json()
        response = self.refresh(self.books(0.3), context=first["context"], gzip_body=True)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["summary"]["arb_count"], 1)
        self.assertAlmostEqual(data["summary"]["best_margin"], 0.2)
        self.assertEqual(data["sheets"][2]["rows"][0][7]["v"], 30)

    def test_unavailable_books_report_failure(self):
        self.assertEqual(self.refresh(({}, {})).status_code, 502)

    def test_partial_books_do_not_reuse_old_asks(self):
        response = self.refresh(({0: {"no": (0.8, 10)}}, self.books(0.4)[1]))
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["summary"]["arb_count"], 0)
        self.assertIsNone(data["sheets"][2]["rows"][0][7]["v"])

    def test_invalid_context_is_rejected(self):
        bad = dict(self.snapshot["context"], matched_poly=[99])
        self.assertEqual(self.refresh(self.books(0.4), context=bad).status_code, 400)
        old = dict(self.snapshot["context"], version=1)
        self.assertEqual(self.refresh(self.books(0.4), context=old).status_code, 400)
        response = self.client.post("/api/refresh", data=b"not gzip",
                                    headers={"X-Content-Encoding": "gzip"})
        self.assertEqual(response.status_code, 400)

    def test_scan_stops_at_time_limit(self):
        def slow_scan(cfg, rep):
            time.sleep(0.05)
            rep.check_cancelled()
        with patch.dict(os.environ, {"SCAN_TIME_LIMIT": "0.01"}), \
                patch.object(web, "run_scan", side_effect=slow_scan):
            events = dict(parse_sse(self.client.post("/api/scan", json={})
                                    .get_data(as_text=True)))
        self.assertIn("time limit", events["scan_error"]["error"])
        self.assertEqual(events["end"]["status"], "error")

    def test_static_assets_served_locally(self):
        response = self.client.get("/static/js/app.js")
        self.assertEqual(response.status_code, 200)
        response.close()
        self.assertIn(b"/static/js/app.js", self.client.get("/").data)


if __name__ == "__main__":
    unittest.main()

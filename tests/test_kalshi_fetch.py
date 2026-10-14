"""Kalshi discovery pages events and drops illiquid, closed, and late markets."""

import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from matcher.config import ScanConfig
from matcher.core import Reporter, fetch_kalshi_markets


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def _market(ticker, volume, close, status="active"):
    return {"ticker": ticker, "status": status, "volume_fp": volume,
            "close_time": close, "title": ticker}


class KalshiFetchTests(unittest.TestCase):
    def test_keeps_only_liquid_markets_before_the_cutoff(self):
        pages = [
            {"events": [{"markets": [
                _market("KEEP", "30000", "2026-11-01T00:00:00Z"),
                _market("THIN", "10", "2026-11-01T00:00:00Z"),
                _market("LATE", "90000", "2027-06-01T00:00:00Z"),
                _market("SHUT", "90000", "2026-11-01T00:00:00Z", status="closed"),
            ]}], "cursor": "next"},
            {"events": [{"markets": [
                _market("KEEP", "30000", "2026-11-01T00:00:00Z"),
                _market("ALSO", "40000.5", "2026-11-15T00:00:00Z", status="open"),
            ]}], "cursor": ""},
        ]
        calls = []

        def fake_get(url, params=None, **_kwargs):
            calls.append((url, params))
            return _Response(pages[len(calls) - 1])

        cfg = ScanConfig(min_volume=25_000, use_llm=False,
                         cutoff_date=datetime(2026, 12, 1, tzinfo=timezone.utc))
        with patch("matcher.core._kalshi_get", side_effect=fake_get):
            found = fetch_kalshi_markets(cfg, Reporter())

        self.assertEqual([m["ticker"] for m in found], ["ALSO", "KEEP"])
        self.assertIn("/events", calls[0][0])
        self.assertEqual(calls[0][1]["status"], "open")
        self.assertEqual(calls[0][1]["with_nested_markets"], "true")
        self.assertEqual(calls[0][1]["limit"], 200)
        self.assertNotIn("cursor", calls[0][1])
        self.assertEqual(calls[1][1]["cursor"], "next")

    def test_past_cutoff_skips_the_network(self):
        cfg = ScanConfig(use_llm=False,
                         cutoff_date=datetime(2020, 1, 1, tzinfo=timezone.utc))
        with patch("matcher.core._kalshi_get") as get:
            self.assertEqual(fetch_kalshi_markets(cfg, Reporter()), [])
        get.assert_not_called()


if __name__ == "__main__":
    unittest.main()

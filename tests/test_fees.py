"""Fee formulas against hand-computed values from each venue's published schedule."""

import unittest
from datetime import datetime, timedelta, timezone

from matcher.config import ScanConfig
from matcher.core import Reporter, find_arbitrage
from matcher.fees import kalshi_taker_fee, poly_taker_fee

POLITICS = {"feesEnabled": True, "feeSchedule": {"rate": 0.04, "exponent": 1}}
KALSHI_STD = {"_fee_type": "quadratic", "_fee_multiplier": 1}


class PolymarketFeeTests(unittest.TestCase):
    def test_matches_published_table(self):
        # docs.polymarket.com/trading/fees: 100 shares at 50¢, crypto (0.07) → $1.75.
        crypto = {"feesEnabled": True, "feeSchedule": {"rate": 0.07}}
        self.assertAlmostEqual(poly_taker_fee(crypto, 100, 0.5), 1.75)
        self.assertAlmostEqual(poly_taker_fee(POLITICS, 100, 0.5), 1.00)

    def test_symmetric_in_price(self):
        self.assertAlmostEqual(poly_taker_fee(POLITICS, 100, 0.3),
                               poly_taker_fee(POLITICS, 100, 0.7))

    def test_fee_free_market(self):
        self.assertEqual(poly_taker_fee({"feesEnabled": False}, 100, 0.5), 0)

    def test_missing_data_assumes_highest_rate(self):
        self.assertAlmostEqual(poly_taker_fee({}, 100, 0.5), 1.75)


class KalshiFeeTests(unittest.TestCase):
    def test_standard_fee(self):
        # 0.07 × 100 × 0.5 × 0.5 = $1.75 exactly.
        self.assertAlmostEqual(kalshi_taker_fee(KALSHI_STD, 100, 0.5), 1.75)

    def test_rounds_up_to_the_cent_per_order(self):
        # One contract at 50¢: 0.0175 → 2¢.
        self.assertAlmostEqual(kalshi_taker_fee(KALSHI_STD, 1, 0.5), 0.02)
        # 10 at 30¢: 0.07 × 10 × 0.21 = 0.147 → 15¢.
        self.assertAlmostEqual(kalshi_taker_fee(KALSHI_STD, 10, 0.3), 0.15)

    def test_series_multiplier(self):
        half = {"_fee_type": "quadratic", "_fee_multiplier": 0.5}
        free = {"_fee_type": "quadratic_with_maker_fees", "_fee_multiplier": 0}
        self.assertAlmostEqual(kalshi_taker_fee(half, 100, 0.5), 0.88)   # 0.875 → 88¢
        self.assertEqual(kalshi_taker_fee(free, 100, 0.5), 0)

    def test_unknown_fee_type_assumes_standard(self):
        self.assertAlmostEqual(kalshi_taker_fee({"_fee_type": "flat"}, 100, 0.5), 1.75)
        self.assertAlmostEqual(kalshi_taker_fee({}, 100, 0.5), 1.75)


class NetArbitrageTests(unittest.TestCase):
    def arbs(self, poly_yes, kalshi_no, size=100):
        end = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        poly = {**POLITICS, "endDate": end}
        kalshi = {**KALSHI_STD, "expected_expiration_time": end}
        pob = {0: {"yes": (poly_yes, size), "no": (0.99, size)}}
        kob = {0: ((0.99, size), (kalshi_no, size))}
        return find_arbitrage([(poly, kalshi, set(), 0.9)], [poly], [kalshi],
                              pob, kob, ScanConfig(), Reporter())

    def test_edge_is_net_of_both_fees(self):
        # Poly YES 45¢ + Kalshi NO 50¢ = 5¢ gross on 100 pairs.
        # Poly fee 100 × 0.04 × 0.45 × 0.55 = $0.99; Kalshi fee $1.75.
        (arb,) = self.arbs(0.45, 0.50)
        self.assertAlmostEqual(arb["gross_margin"], 0.05)
        self.assertAlmostEqual(arb["fees_per_pair"], 0.0274)
        self.assertAlmostEqual(arb["margin"], 0.0226)
        self.assertAlmostEqual(arb["max_profit"], 2.26)
        self.assertAlmostEqual(arb["roi"], 0.0226 / 0.9774)
        self.assertFalse(arb["fees_estimated"])

    def test_fees_can_erase_a_gross_edge(self):
        # 2¢ gross, ~2.7¢ in fees: not an arb.
        self.assertEqual(self.arbs(0.48, 0.50), [])


if __name__ == "__main__":
    unittest.main()

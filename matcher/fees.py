"""Taker fees for both venues, so arbitrage edges are net of what you actually pay.

Every arb leg *buys at the best ask*, so it takes liquidity: only taker fees
apply. Maker fees and maker rebates never enter the calculation.

Polymarket (global CLOB) — https://docs.polymarket.com/trading/fees
    fee = C × feeRate × p × (1 − p), in USDC, rounded to 5 decimals.
    feeRate is per market, from Gamma's `feeSchedule.rate` (e.g. politics 0.04,
    crypto 0.07, some older sports markets 0.03). `feesEnabled: false` means no
    fee (geopolitics, and some older markets). Makers are never charged.

Kalshi — https://docs.kalshi.com/getting_started/fee_rounding and the series API
    fee = 0.07 × C × P × (1 − P) × fee_multiplier, rounded up to the next cent
    per order (trade fees round up to $0.000001 per fill, then a rounding fee
    lifts the order's total to the next whole cent for retail accounts).
    `fee_type` and `fee_multiplier` come from GET /series/{ticker}; live values
    are 1, 0.5 (e.g. some baseball props) and 0 (fee-free series).
    The 0.07 coefficient is from Kalshi's fee schedule PDF as reported by
    multiple sources; it is not returned by the API.

When a market's fee data is missing, the highest standard rate is assumed, so a
gap in the data can hide an arb but never invent one.
"""

import math

POLY_MAX_RATE = 0.07         # Highest published category rate (crypto).
KALSHI_TAKER_COEF = 0.07     # General Trading Fees Table coefficient.

# fee_type values whose taker fee is the general quadratic formula.
KALSHI_QUADRATIC = {"quadratic", "quadratic_with_maker_fees",
                    "quadratic_with_combo_maker_fees"}


def poly_fee_rate(market: dict):
    """(rate, known) for a Polymarket market's taker fee."""
    enabled = market.get("feesEnabled")
    if enabled is False:
        return 0.0, True
    schedule = market.get("feeSchedule") or {}
    rate = schedule.get("rate")
    if enabled and isinstance(rate, (int, float)) and rate >= 0:
        # The docs publish the fee curve with exponent 1. Any other exponent is
        # undocumented; p(1-p) ≤ 0.25 so exponent 1 is the larger, safer fee.
        return float(rate), True
    return POLY_MAX_RATE, False


def poly_taker_fee(market: dict, contracts: float, price: float) -> float:
    """Total USDC fee for buying `contracts` shares at `price`."""
    rate, _ = poly_fee_rate(market)
    return round(contracts * rate * price * (1 - price), 5)


def kalshi_fee_multiplier(market: dict):
    """(multiplier, known) for a Kalshi market, from its series' fee settings.

    The scan stores the series' fee_type / fee_multiplier on the market dict
    under `_fee_type` / `_fee_multiplier`.
    """
    fee_type = market.get("_fee_type")
    mult = market.get("_fee_multiplier")
    if fee_type in KALSHI_QUADRATIC and isinstance(mult, (int, float)) and mult >= 0:
        return float(mult), True
    # "flat" (Specific Trading Fees Table) or missing: assume the standard fee.
    return 1.0, False


def kalshi_taker_fee(market: dict, contracts: float, price: float) -> float:
    """Total dollar fee for one order of `contracts` at `price`, rounded up to the cent."""
    mult, _ = kalshi_fee_multiplier(market)
    raw = KALSHI_TAKER_COEF * mult * contracts * price * (1 - price)
    # Round to 9 places first so float noise (e.g. 1.7500000001) can't add a cent.
    return math.ceil(round(raw * 100, 9)) / 100


def leg_fee(platform: str, market: dict, contracts: float, price: float) -> float:
    if platform == "Polymarket":
        return poly_taker_fee(market, contracts, price)
    return kalshi_taker_fee(market, contracts, price)


def fee_known(platform: str, market: dict) -> bool:
    if platform == "Polymarket":
        return poly_fee_rate(market)[1]
    return kalshi_fee_multiplier(market)[1]

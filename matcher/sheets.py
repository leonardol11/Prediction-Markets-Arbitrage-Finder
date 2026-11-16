"""Turn a ScanResult into sheet models.

One sheet model feeds both the website's grid and the .xlsx export, so the two
can never drift apart.

    sheet  = {"key", "name", "columns": [...], "rows": [[cell, ...], ...]}
    column = {"key", "label", "align", "width", "kind"}
    cell   = a scalar, or {"v": sortable value, "d": display text,
                           "u": link, "f": Excel number format}
"""

import json
from datetime import datetime, timezone

from .core import (_num, best_ask, best_quote, days_until, kalshi_end,
                   kalshi_listed_book, kalshi_title, pair_days, pair_id, poly_end, poly_title, quote_state)


# ─────────────────────────────── cell helpers ─────────────────────────────────

DASH = "—"


def cell(value, display=None, url=None, fmt=None):
    if display is None and url is None and fmt is None:
        return value
    out = {"v": value, "d": DASH if display is None else display}
    if url:
        out["u"] = url
    if fmt and value is not None:
        out["f"] = fmt
    return out


CENTS_FMT, QTY_FMT, MONEY_FMT, PCT_FMT = '0"¢"', '#,##0', '$#,##0', '0.00"%"'
DAYS_FMT = '0"d"'
CENTS2_FMT = '0.00"¢"' 


def cents(v):
    """0.37 → '37¢' (kept as a sortable 0-100 number underneath)."""
    if v is None:
        return cell(None, DASH)
    try:
        c = round(float(v) * 100)
    except (TypeError, ValueError):
        return cell(None, DASH)
    return cell(c, f"{c}¢", fmt=CENTS_FMT)


def cents_fine(v, suffix=""):
    """Sub-cent precision for fees: 0.0175 → '1.75¢'."""
    if v is None:
        return cell(None, DASH)
    c = round(float(v) * 100, 2)
    return cell(c, f"{c:.2f}¢{suffix}", fmt=CENTS2_FMT)


def qty(v):
    try:
        x = float(v or 0)
    except (TypeError, ValueError):
        return cell(None, DASH)
    return cell(round(x), f"{round(x):,}", fmt=QTY_FMT) if x else cell(None, DASH)


def money(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return cell(None, DASH)
    return cell(round(x), f"${round(x):,}", fmt=MONEY_FMT)


def dollars(v):
    """Profit in dollars and cents: 2.26 → '$2.26'."""
    if not v:
        return cell(None, DASH)
    return cell(round(float(v), 2), f"${float(v):,.2f}", fmt='$#,##0.00')


def pct(v, places=1):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return cell(None, DASH)
    return cell(round(x * 100, 4), f"{x * 100:.{places}f}%", fmt=PCT_FMT)


def days_cell(d):
    """Fractional days → whole-day display; '<1d' for same-day settlement."""
    if d is None:
        return cell(None, DASH)
    return cell(round(d, 2), "<1d" if d < 1 else f"{d:.0f}d", fmt=DAYS_FMT)


def day(s):
    s = (s or "")[:10]
    return cell(s, s) if s else cell(None, DASH)


# ─────────────────────────── market field extraction ──────────────────────────

def _poly_json_list(market, key):
    raw = market.get(key)
    try:
        return json.loads(raw) if isinstance(raw, str) else (raw or [])
    except (json.JSONDecodeError, TypeError):
        return []


def poly_price(market, label, fallback_idx):
    outcomes = _poly_json_list(market, "outcomes")
    prices = _poly_json_list(market, "outcomePrices")
    idx = next((i for i, o in enumerate(outcomes)
                if str(o).lower() == label.lower()), None)
    if idx is not None and idx < len(prices):
        return _num(prices[idx], None)
    return _num(prices[fallback_idx], None) if len(prices) > fallback_idx else None


def poly_spread(market):
    s = market.get("spread")
    if s is None:
        bid, ask = market.get("bestBid"), market.get("bestAsk")
        if bid is None or ask is None:
            return None
        s = _num(ask) - _num(bid)
    return _num(s, None)


def kalshi_spread(market):
    bid, ask = market.get("yes_bid_dollars"), market.get("yes_ask_dollars")
    if bid is None or ask is None:
        return None
    return _num(ask) - _num(bid)


def poly_url(market) -> str:
    events = market.get("events") or []
    if events and isinstance(events[0], dict) and events[0].get("slug"):
        return f"https://polymarket.com/event/{events[0]['slug']}"
    return f"https://polymarket.com/market/{market.get('slug', '')}"


def kalshi_url(market) -> str:
    ticker = market.get("event_ticker") or market.get("ticker") or ""
    series = ticker.split("-")[0].lower()
    return f"https://kalshi.com/markets/{series}" if series else "https://kalshi.com"


def kalshi_day(market) -> str:
    return (market.get("expected_expiration_time")
            or market.get("expiration_time")
            or market.get("close_time") or "")[:10]


def poly_day(market) -> str:
    return (market.get("endDate") or market.get("end_date_iso") or "")[:10]


def _ask_cells(book, side):
    """Best ask and the size resting at it, from a live book."""
    price, size = best_ask(book, side)
    if price is None:
        return cell(None, DASH), cell(None, DASH)
    return cents(price), qty(size)


def seconds_cell(v):
    if v is None:
        return cell(None, DASH)
    return cell(round(v, 1), f"{v:.0f}s", fmt='0"s"')


# ───────────────────────────────── column sets ────────────────────────────────

def _col(key, label, align="left", width=110, kind="text"):
    return {"key": key, "label": label, "align": align, "width": width, "kind": kind}


ARB_COLUMNS = [
    _col("num", "#", "center", 44, "num"),
    _col("apr", "Annualized", "center", 96, "num"),
    _col("margin", "Net edge", "center", 84, "num"),
    _col("roi", "ROI", "center", 72, "num"),
    _col("days", "Days", "center", 64, "num"),
    _col("p_title", "Polymarket market", "left", 300, "text"),
    _col("buy_poly", "Buy", "center", 62, "side"),
    _col("poly_price", "Avg price", "center", 76, "num"),
    _col("poly_qty", "Ask qty", "right", 78, "num"),
    _col("k_title", "Kalshi market", "left", 300, "text"),
    _col("buy_kalshi", "Buy", "center", 62, "side"),
    _col("kalshi_price", "Avg price", "center", 76, "num"),
    _col("kalshi_qty", "Ask qty", "right", 78, "num"),
    _col("gross", "Gross edge", "center", 88, "num"),
    _col("fees", "Fees / pair", "center", 96, "num"),
    _col("cost", "Cost / pair", "center", 88, "num"),
    _col("size", "Exec. size", "right", 86, "num"),
    _col("profit", "Est. profit", "right", 92, "num"),
    _col("levels", "Levels", "center", 64, "num"),
    _col("age", "Quote age", "center", 82, "num"),
    _col("match", "Match", "center", 70, "num"),
    _col("expires", "Resolves", "center", 96, "text"),
    _col("p_vol", "Poly vol", "right", 100, "num"),
    _col("k_vol", "Kalshi vol", "right", 100, "num"),
]


def _match_columns(score_label):
    return [
        _col("num", "#", "center", 48, "num"),
        _col("score", score_label, "center", 92, "num"),
        _col("status", "Status", "left", 150, "text"),
        _col("edge", "Net edge", "center", 84, "num"),
        _col("days", "Days", "center", 64, "num"),
        _col("p_title", "Polymarket market", "left", 330, "text"),
        _col("p_date", "Resolves", "center", 100, "text"),
        _col("p_yes", "Yes", "center", 68, "num"),
        _col("p_no", "No", "center", 68, "num"),
        _col("p_spread", "Spread", "center", 78, "num"),
        _col("p_yes_ask", "Yes ask", "center", 78, "num"),
        _col("p_yes_qty", "Yes qty", "right", 80, "num"),
        _col("p_no_ask", "No ask", "center", 78, "num"),
        _col("p_no_qty", "No qty", "right", 80, "num"),
        _col("p_vol", "Poly vol", "right", 106, "num"),
        _col("k_title", "Kalshi market", "left", 330, "text"),
        _col("k_date", "Resolves", "center", 100, "text"),
        _col("k_yes", "Yes", "center", 68, "num"),
        _col("k_no", "No", "center", 68, "num"),
        _col("k_spread", "Spread", "center", 78, "num"),
        _col("k_yes_ask", "Yes ask", "center", 78, "num"),
        _col("k_yes_qty", "Yes qty", "right", 80, "num"),
        _col("k_no_ask", "No ask", "center", 78, "num"),
        _col("k_no_qty", "No qty", "right", 80, "num"),
        _col("k_vol", "Kalshi vol", "right", 106, "num"),
        _col("shared", "Shared keywords", "left", 260, "text"),
    ]


POLY_COLUMNS = [
    _col("num", "#", "center", 48, "num"),
    _col("title", "Market", "left", 460, "text"),
    _col("date", "Resolves", "center", 100, "text"),
    _col("days", "Days", "center", 64, "num"),
    _col("yes", "Yes", "center", 68, "num"),
    _col("no", "No", "center", 68, "num"),
    _col("spread", "Spread", "center", 78, "num"),
    _col("vol24", "Vol 24h", "right", 112, "num"),
    _col("vol", "Vol total", "right", 118, "num"),
    _col("liq", "Liquidity", "right", 112, "num"),
]

KALSHI_COLUMNS = [
    _col("num", "#", "center", 48, "num"),
    _col("title", "Market", "left", 460, "text"),
    _col("date", "Resolves", "center", 100, "text"),
    _col("days", "Days", "center", 64, "num"),
    _col("yes", "Yes", "center", 68, "num"),
    _col("no", "No", "center", 68, "num"),
    _col("spread", "Spread", "center", 78, "num"),
    _col("vol24", "Vol 24h", "right", 112, "num"),
    _col("vol", "Vol total", "right", 118, "num"),
    _col("ticker", "Ticker", "left", 220, "text"),
]


# ─────────────────────────────── sheet builders ───────────────────────────────

def _arb_rows(arbs):
    rows = []
    for i, arb in enumerate(arbs):
        p, k = arb["poly"], arb["kalshi"]
        p_side = arb["a_side"] if arb["a_plat"] == "Polymarket" else arb["b_side"]
        k_side = arb["b_side"] if arb["b_plat"] == "Kalshi" else arb["a_side"]
        p_price = arb["a_price"] if arb["a_plat"] == "Polymarket" else arb["b_price"]
        p_qty = arb["a_ask_size"] if arb["a_plat"] == "Polymarket" else arb["b_ask_size"]
        k_price = arb["b_price"] if arb["b_plat"] == "Kalshi" else arb["a_price"]
        k_qty = arb["b_ask_size"] if arb["b_plat"] == "Kalshi" else arb["a_ask_size"]
        expires = kalshi_day(k) or poly_day(p)

        rows.append([
            i + 1,
            pct(arb["apr"], 1) if arb.get("apr") is not None else cell(None, DASH),
            pct(arb["margin"], 2),
            pct(arb.get("roi"), 2),
            days_cell(arb.get("days")),
            cell(poly_title(p), poly_title(p), poly_url(p)),
            p_side,
            cents_fine(p_price),
            qty(p_qty),
            cell(kalshi_title(k), kalshi_title(k), kalshi_url(k)),
            k_side,
            cents_fine(k_price),
            qty(k_qty),
            pct(arb.get("gross_margin", arb["margin"]), 2),
            cents_fine(arb.get("fees_per_pair"),
                       " est." if arb.get("fees_estimated") else ""),
            cents_fine(arb.get("cost", arb["prob_sum"])),
            qty(arb.get("max_contracts")),
            dollars(arb.get("max_profit")),
            arb.get("levels") or 1,
            seconds_cell(arb.get("age")),
            pct(arb.get("match_score"), 0),
            day(expires),
            money(_num(p.get("volume"))),
            money(_num(k.get("volume_fp"))),
        ])
    return rows


EDGE_LABELS = {"live": "Live arb", "stale": "Stale quote", "small": "Below min size",
               "near": "Near", "none": "No edge", "unpriced": "No live book"}


def _status(verdict, state, verified_sheet, listed=None):
    """One short label: has an LLM confirmed the pair, and is there an edge?"""
    if verified_sheet:
        return EDGE_LABELS[state]
    if verdict == "rejected":
        return "Rejected by LLM"
    if verdict == "verified":
        return f"Verified · {EDGE_LABELS[state].lower()}"
    if state in ("live", "stale", "small") or (listed and listed["margin"] > 0):
        return "Unverified · edge"
    return "Unverified · near" if state == "near" else "Unverified"


def _match_rows(pairs, result, poly_idx, kalshi_idx, verified_sheet=False):
    now = datetime.now(timezone.utc)
    cfg = result.config
    rows = []
    for i, (p, k, shared, score) in enumerate(pairs):
        pob = result.poly_ob.get(poly_idx.get(id(p), -1))
        kob = result.kalshi_ob.get(kalshi_idx.get(id(k), -1))
        p_yes_ask, p_yes_qty = _ask_cells(pob, "yes")
        p_no_ask, p_no_qty = _ask_cells(pob, "no")
        k_yes_ask, k_yes_qty = _ask_cells(kob, "yes")
        k_no_ask, k_no_qty = _ask_cells(kob, "no")
        quote = best_quote(p, k, pob, kob, cfg)
        # No live Kalshi book (the pair is unverified): judge by the listed quote,
        # for the label only. The Net edge column shows live books and nothing else.
        listed = None if quote else best_quote(p, k, pob, kalshi_listed_book(k), cfg)

        rows.append([
            i + 1,
            pct(score, 0),
            _status(result.pair_status.get(pair_id(p, k)), quote_state(quote, cfg),
                    verified_sheet, listed),
            pct(quote["margin"], 2) if quote else cell(None, DASH),
            days_cell(pair_days(p, k, now)),
            cell(poly_title(p), poly_title(p), poly_url(p)),
            day(poly_day(p)),
            cents(poly_price(p, "Yes", 0)),
            cents(poly_price(p, "No", 1)),
            cents(poly_spread(p)),
            p_yes_ask, p_yes_qty, p_no_ask, p_no_qty,
            money(_num(p.get("volume"))),
            cell(kalshi_title(k), kalshi_title(k), kalshi_url(k)),
            day(kalshi_day(k)),
            cents(_num(k.get("yes_ask_dollars"), None)),
            cents(_num(k.get("no_ask_dollars"), None)),
            cents(kalshi_spread(k)),
            k_yes_ask, k_yes_qty, k_no_ask, k_no_qty,
            money(_num(k.get("volume_fp"))),
            ", ".join(sorted(shared)) if shared else DASH,
        ])
    return rows


def _poly_rows(markets):
    now = datetime.now(timezone.utc)
    return [[
        i + 1,
        cell(poly_title(m), poly_title(m), poly_url(m)),
        day(poly_day(m)),
        days_cell(days_until(poly_end(m), now)),
        cents(poly_price(m, "Yes", 0)),
        cents(poly_price(m, "No", 1)),
        cents(poly_spread(m)),
        money(_num(m.get("volume24hr"))),
        money(_num(m.get("volume"))),
        money(_num(m.get("liquidityNum") or m.get("liquidity"))),
    ] for i, m in enumerate(markets)]


def _kalshi_rows(markets):
    now = datetime.now(timezone.utc)
    return [[
        i + 1,
        cell(kalshi_title(m), kalshi_title(m), kalshi_url(m)),
        day(kalshi_day(m)),
        days_cell(days_until(kalshi_end(m), now)),
        cents(_num(m.get("yes_ask_dollars"), None)),
        cents(_num(m.get("no_ask_dollars"), None)),
        cents(kalshi_spread(m)),
        money(_num(m.get("volume_24h_fp"))),
        money(_num(m.get("volume_fp"))),
        m.get("ticker") or DASH,
    ] for i, m in enumerate(markets)]


def build_sheets(result) -> list:
    """Ordered list of sheet models — arbitrage first, raw markets last."""
    poly_idx = {id(m): i for i, m in enumerate(result.matched_poly)}
    kalshi_idx = {id(m): i for i, m in enumerate(result.matched_kalshi)}

    return [
        {"key": "arbitrage", "name": "Live Arbitrage",
         "hint": ("LLM-verified pairs that are profitable right now. Edges are net of both "
                  "venues' taker fees and averaged over the size the order books can fill "
                  "at a profit (Exec. size), ranked by annualized return. \"est.\" = fee "
                  f"data missing, highest standard rate assumed. Quotes older than "
                  f"{result.config.stale_quote_seconds:.0f}s are left out."),
         "columns": ARB_COLUMNS, "rows": _arb_rows(result.arbs),
         "empty": ("No live arbitrage: no LLM-verified pair costs under $1.00 per pair "
                   "after fees on current order books.")},

        {"key": "verified", "name": "LLM Verified",
         "hint": (f"Every pair an LLM confirmed resolves on the same outcome "
                  f"(≥{result.config.groq_confidence_cutoff:.0%} confidence), including "
                  "verdicts reused from earlier scans, whether or not there is an edge now."),
         "columns": _match_columns("Confidence"),
         "rows": _match_rows(result.verified, result, poly_idx, kalshi_idx,
                             verified_sheet=True),
         "empty": ("No verified pairs yet. LLM verification was off for this scan."
                   if not result.llm_used else
                   "The LLM has not confirmed any of the candidate pairs.")},

        {"key": "matches", "name": "Matches",
         "hint": (f"Title pairs above {result.config.match_display_threshold:.0%} combined "
                  "TF-IDF + Jaccard similarity. Unverified pairs are never counted as "
                  "arbitrage; live Kalshi books are fetched only for pairs near an edge."),
         "columns": _match_columns("Similarity"),
         "rows": _match_rows(result.matches, result, poly_idx, kalshi_idx),
         "empty": "No title pairs cleared the similarity threshold. Try lowering it."},

        {"key": "polymarket", "name": "Polymarket",
         "hint": "Every eligible Polymarket market this scan considered.",
         "columns": POLY_COLUMNS, "rows": _poly_rows(result.poly),
         "empty": "No eligible Polymarket markets."},

        {"key": "kalshi", "name": "Kalshi",
         "hint": "Every eligible Kalshi market this scan considered.",
         "columns": KALSHI_COLUMNS, "rows": _kalshi_rows(result.kalshi),
         "empty": "No eligible Kalshi markets."},
    ]


def build_summary(result) -> dict:
    # Arbs are ranked by annualized return, so the biggest raw edge may not be first.
    best = max((a["margin"] for a in result.arbs), default=None)
    judged = set(result.pair_status) | {pair_id(p, k) for p, k, _s, _c in result.verified}
    return {
        "poly_count": len(result.poly),
        "kalshi_count": len(result.kalshi),
        "match_count": len(result.matches),
        "verified_count": len(result.verified),
        "arb_count": len(result.arbs),
        "pending_count": sum(1 for p, k, _s, _c in result.matches
                             if pair_id(p, k) not in judged),
        "timings": result.timings,
        "llm_stats": result.llm_stats,
        "best_margin": best,
        "best_margin_display": f"{best * 100:.2f}%" if best else DASH,
        "arb_source": result.arb_source_label,
        "llm_used": result.llm_used,
        "duration": round(result.duration_seconds, 1),
        "finished_at": (result.finished_at or result.started_at).isoformat(),
        "min_volume": result.config.min_volume,
        "cutoff": result.config.cutoff_date.strftime("%Y-%m-%d"),
    }

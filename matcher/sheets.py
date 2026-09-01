"""Turn a ScanResult into sheet models.

One sheet model feeds both the website's grid and the .xlsx export, so the two
can never drift apart.

    sheet  = {"key", "name", "columns": [...], "rows": [[cell, ...], ...]}
    column = {"key", "label", "align", "width", "kind"}
    cell   = a scalar, or {"v": sortable value, "d": display text,
                           "u": link, "f": Excel number format}
"""

import json

from .core import _num, kalshi_title, poly_title


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


def cents(v):
    """0.37 → '37¢' (kept as a sortable 0-100 number underneath)."""
    if v is None:
        return cell(None, DASH)
    try:
        c = round(float(v) * 100)
    except (TypeError, ValueError):
        return cell(None, DASH)
    return cell(c, f"{c}¢", fmt=CENTS_FMT)


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


def pct(v, places=1):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return cell(None, DASH)
    return cell(round(x * 100, 4), f"{x * 100:.{places}f}%", fmt=PCT_FMT)


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


def _poly_ask_pair(ob_side):
    if not ob_side or ob_side[0] is None:
        return cell(None, DASH), cell(None, DASH)
    return cents(ob_side[0]), qty(ob_side[1])


def _kalshi_ask_pairs(kob):
    if not kob:
        return (cell(None, DASH),) * 4
    (a, aq), (b, bq) = kob
    return cents(a), qty(aq), cents(b), qty(bq)


# ───────────────────────────────── column sets ────────────────────────────────

def _col(key, label, align="left", width=110, kind="text"):
    return {"key": key, "label": label, "align": align, "width": width, "kind": kind}


ARB_COLUMNS = [
    _col("num", "#", "center", 44, "num"),
    _col("margin", "Edge", "center", 76, "num"),
    _col("p_title", "Polymarket market", "left", 300, "text"),
    _col("buy_poly", "Buy", "center", 62, "side"),
    _col("poly_price", "Price", "center", 66, "num"),
    _col("poly_qty", "Ask qty", "right", 78, "num"),
    _col("k_title", "Kalshi market", "left", 300, "text"),
    _col("buy_kalshi", "Buy", "center", 62, "side"),
    _col("kalshi_price", "Price", "center", 66, "num"),
    _col("kalshi_qty", "Ask qty", "right", 78, "num"),
    _col("cost", "Cost / pair", "center", 88, "num"),
    _col("size", "Max pairs", "right", 86, "num"),
    _col("profit", "Max profit", "right", 92, "num"),
    _col("match", "Match", "center", 70, "num"),
    _col("expires", "Resolves", "center", 96, "text"),
    _col("p_vol", "Poly vol", "right", 100, "num"),
    _col("k_vol", "Kalshi vol", "right", 100, "num"),
]


def _match_columns(score_label):
    return [
        _col("num", "#", "center", 48, "num"),
        _col("score", score_label, "center", 92, "num"),
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
            pct(arb["margin"], 2),
            cell(poly_title(p), poly_title(p), poly_url(p)),
            p_side,
            cents(p_price),
            qty(p_qty),
            cell(kalshi_title(k), kalshi_title(k), kalshi_url(k)),
            k_side,
            cents(k_price),
            qty(k_qty),
            cents(arb["prob_sum"]),
            qty(arb.get("max_contracts")),
            money(arb["max_profit"]) if arb.get("max_profit") else cell(None, DASH),
            pct(arb.get("match_score"), 0),
            day(expires),
            money(_num(p.get("volume"))),
            money(_num(k.get("volume_fp"))),
        ])
    return rows


def _match_rows(pairs, poly_ob, kalshi_ob, poly_idx, kalshi_idx):
    rows = []
    for i, (p, k, shared, score) in enumerate(pairs):
        pi = poly_idx.get(id(p), -1)
        ki = kalshi_idx.get(id(k), -1)
        pob = poly_ob.get(pi, {}) if pi >= 0 else {}
        kob = kalshi_ob.get(ki) if ki >= 0 else None
        p_yes_ask, p_yes_qty = _poly_ask_pair(pob.get("yes"))
        p_no_ask, p_no_qty = _poly_ask_pair(pob.get("no"))
        k_yes_ask, k_yes_qty, k_no_ask, k_no_qty = _kalshi_ask_pairs(kob)

        rows.append([
            i + 1,
            pct(score, 0),
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
    return [[
        i + 1,
        cell(poly_title(m), poly_title(m), poly_url(m)),
        day(poly_day(m)),
        cents(poly_price(m, "Yes", 0)),
        cents(poly_price(m, "No", 1)),
        cents(poly_spread(m)),
        money(_num(m.get("volume24hr"))),
        money(_num(m.get("volume"))),
        money(_num(m.get("liquidityNum") or m.get("liquidity"))),
    ] for i, m in enumerate(markets)]


def _kalshi_rows(markets):
    return [[
        i + 1,
        cell(kalshi_title(m), kalshi_title(m), kalshi_url(m)),
        day(kalshi_day(m)),
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
        {"key": "arbitrage", "name": "Arbitrage",
         "hint": ("Buy both legs at these asks; one leg always pays $1.00. "
                  "Confirm the two questions really resolve alike before trading."),
         "columns": ARB_COLUMNS, "rows": _arb_rows(result.arbs),
         "empty": "No arbitrage on current best asks — the matched books all price above $1.00 combined."},

        {"key": "verified", "name": "LLM Verified",
         "hint": f"Pairs an LLM confirmed resolve on the same outcome (≥{result.config.groq_confidence_cutoff:.0%} confidence).",
         "columns": _match_columns("Confidence"),
         "rows": _match_rows(result.verified, result.poly_ob, result.kalshi_ob,
                             poly_idx, kalshi_idx),
         "empty": ("LLM verification was off for this scan."
                   if not result.llm_used else
                   "The LLM confirmed none of the candidate pairs.")},

        {"key": "matches", "name": "Matches",
         "hint": f"Title pairs above {result.config.match_display_threshold:.0%} combined TF-IDF + Jaccard similarity.",
         "columns": _match_columns("Similarity"),
         "rows": _match_rows(result.matches, result.poly_ob, result.kalshi_ob,
                             poly_idx, kalshi_idx),
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
    best = result.arbs[0]["margin"] if result.arbs else None
    return {
        "poly_count": len(result.poly),
        "kalshi_count": len(result.kalshi),
        "match_count": len(result.matches),
        "verified_count": len(result.verified),
        "arb_count": len(result.arbs),
        "best_margin": best,
        "best_margin_display": f"{best * 100:.2f}%" if best else DASH,
        "arb_source": result.arb_source_label,
        "llm_used": result.llm_used,
        "duration": round(result.duration_seconds, 1),
        "finished_at": (result.finished_at or result.started_at).isoformat(),
        "min_volume": result.config.min_volume,
        "cutoff": result.config.cutoff_date.strftime("%Y-%m-%d"),
    }

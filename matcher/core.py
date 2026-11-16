"""Cross-platform prediction-market matcher and arbitrage scanner.

Pipeline
    1. Fetch eligible Polymarket + Kalshi markets (concurrently).
    2. TF-IDF pre-filter to find candidate title pairs.
    3. Price the candidates: full-depth order books, net of both venues' fees.
    4. LLM verification, edge first. Cached verdicts are reused, pairs with an
       edge (or close to one) are always checked, the rest as time allows.
    5. Re-quote the verified pairs and keep the ones that are live arbitrage.

Only pairs an LLM verified can become arbitrage. A failed, timed-out or
skipped verification leaves a pair unverified; it never counts as a match.

Every long step reports through a `Reporter`, so a CLI and the web UI can
render the same progress.
"""

import json
import random
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import copy
from datetime import datetime, timezone

import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_sim

from .config import ScanConfig
from .fees import fee_known, marginal_fee, order_fee
from .verify_cache import VerifyCache, default_path as default_cache_path

TOTAL_STEPS = 5


class Cancelled(Exception):
    """Raised inside the pipeline when the caller asks for an early stop."""


class GroqAuthError(Exception):
    """The Groq key was rejected — retrying other batches would just waste time."""


class Reporter:
    """Collects log lines / progress. Subclass or pass callbacks to stream."""

    def __init__(self, on_log=None, on_progress=None, should_cancel=None):
        self._on_log = on_log
        self._on_progress = on_progress
        self._should_cancel = should_cancel or (lambda: False)

    def log(self, message: str):
        if self._on_log:
            self._on_log(str(message))
        else:
            print(message)

    def step(self, n: int, label: str):
        self.progress(n, label)
        self.log(f"[{n}/{TOTAL_STEPS}] {label}")

    def progress(self, step: int, label: str, pct: float = None):
        if self._on_progress:
            self._on_progress(step, label, pct)

    def check_cancelled(self):
        if self._should_cancel():
            raise Cancelled()


def _parse_iso(s: str):
    try:
        return datetime.fromisoformat((s or "").replace("Z", "+00:00"))
    except ValueError:
        return None


def poly_end(market):
    return _parse_iso(market.get("endDate") or market.get("end_date_iso"))


def kalshi_end(market):
    # Expected expiration is when Kalshi actually settles; close_time is when trading stops.
    return _parse_iso(market.get("expected_expiration_time")
                      or market.get("expiration_time") or market.get("close_time"))


def days_until(dt, now=None):
    """Fractional days from now until dt (never negative), or None if unknown."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    return max(0.0, (dt - now).total_seconds() / 86400)


def pair_days(poly, kalshi, now=None):
    """Capital is locked until both legs settle, so use the later date."""
    days = [d for d in (days_until(poly_end(poly), now), days_until(kalshi_end(kalshi), now))
            if d is not None]
    return max(days) if days else None


# Floor for annualizing: a market closing in an hour would otherwise show a
# meaningless five-digit APR, and settlement/withdrawal rarely beats a day.
MIN_ANNUALIZE_DAYS = 1.0


def annualized_return(roi, days):
    if roi is None or days is None:
        return None
    return roi * 365.0 / max(days, MIN_ANNUALIZE_DAYS)


def _num(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ───────────────────────────────── Polymarket ─────────────────────────────────

POLY_BASE = "https://gamma-api.polymarket.com"
POLY_PAGE = 100          # Gamma caps `limit` at 100 regardless of what we ask for


def _iso_z(dt) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_poly_markets(cfg: ScanConfig, rep: Reporter) -> list:
    """Volume- and date-filtered server-side, then paged until exhausted.

    Two Gamma quirks shape this. `limit` is silently truncated to 100, so paging
    has to key off the page size actually returned — asking for 500 and stopping
    at `len(batch) < 500` caps the scan at the first 100 markets. And `offset`
    is rejected past 2000, so deep pages come from the cursor-based
    `/markets/keyset` endpoint instead — whose cursor parameter is
    `after_cursor`; any other name is accepted and silently ignored, which
    replays page one forever.
    """
    rep.log("Fetching Polymarket markets…")
    now = datetime.now(timezone.utc)
    params = {
        "active": "true", "closed": "false", "limit": POLY_PAGE,
        "order": "volumeNum", "ascending": "false",
        "volume_num_min": int(cfg.min_volume),
        "end_date_min": _iso_z(now),
        "end_date_max": _iso_z(cfg.cutoff_date),
    }

    markets, cursor = [], None
    while len(markets) < cfg.poly_max_markets:
        rep.check_cancelled()
        page = {**params}
        if cursor:
            page["after_cursor"] = cursor
        try:
            r = requests.get(f"{POLY_BASE}/markets/keyset", params=page, timeout=20)
            r.raise_for_status()
            data = r.json()
        except requests.RequestException as exc:
            rep.log(f"  ⚠ Polymarket paging stopped early: {exc}")
            break

        batch = data.get("markets") or []
        if not batch:
            break
        markets.extend(batch)
        rep.progress(1, f"Polymarket: {len(markets):,} markets")

        cursor = data.get("next_cursor")
        if not cursor or len(batch) < POLY_PAGE:
            break

    # Page boundaries can repeat a row.
    markets = list({m.get("id", i): m for i, m in enumerate(markets)}.values())
    rep.log(f"  {len(markets):,} Polymarket markets returned by the filter")

    # Re-check locally: the API filter is a hint, not a guarantee.
    eligible = [
        m for m in markets
        if _num(m.get("volume")) >= cfg.min_volume
        and (d := _parse_iso(m.get("endDate") or m.get("end_date_iso")))
        and now <= d < cfg.cutoff_date
    ]
    eligible.sort(key=lambda m: _num(m.get("volume")), reverse=True)
    rep.log(f"  → {len(eligible):,} eligible (vol ≥ ${cfg.min_volume:,.0f}, "
            f"resolves before {cfg.cutoff_date:%b %d %Y})")
    return eligible


# ─────────────────────────────────── Kalshi ───────────────────────────────────

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_EVENT_PAGE = 200          # `/events` rejects anything above 200
# Paging stays one request at a time on purpose. Measured Nov 2026: walking
# /markets in parallel close-time windows (6 workers, 16 req/s) drew 64 HTTP
# 429s and took 63s, against ~23s for this serial /events walk, and /events
# needs fewer requests for the same markets (98 pages vs 218).
KALSHI_RATE_PER_SEC = 10.0
_session = requests.Session()


class _RateLimiter:
    """Global pace-setter so parallel window walks stay under Kalshi's limit."""

    def __init__(self, per_second: float):
        self._interval = 1.0 / per_second
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + self._interval
        delay = slot - now
        if delay > 0:
            time.sleep(delay)

    def penalize(self, seconds: float = 2.0):
        """Pause every worker briefly after a 429, not just the one that saw it.

        Deliberately not additive: a dozen workers all seeing 429s at once would
        otherwise stack a dozen delays onto the same shared clock and stall the
        fetch for a minute.
        """
        with self._lock:
            self._next = max(self._next, time.monotonic() + seconds)


_kalshi_limiter = _RateLimiter(KALSHI_RATE_PER_SEC)


def _kalshi_get(url, params=None, attempts=10, timeout=20, session=None):
    sess = session or _session
    last = None
    for attempt in range(attempts):
        _kalshi_limiter.wait()
        try:
            r = sess.get(url, params=params, timeout=timeout)
            if r.status_code == 429:
                _kalshi_limiter.penalize(2.0)
                retry_after = _num(r.headers.get("Retry-After"), 0)
                time.sleep(max(retry_after, min(8, 0.5 * 2 ** attempt)))
                last = RuntimeError("429 rate limited")
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as exc:
            last = exc
            if attempt == attempts - 1:
                break
            time.sleep(min(20, 1.0 * 2 ** attempt))
    raise RuntimeError(f"Kalshi request failed after {attempts} attempts: {last}")


def _kalshi_eligible(market: dict, cfg: ScanConfig) -> bool:
    """Volume and resolution filters Kalshi will not apply for us."""
    if market.get("status") not in {"open", "active"}:
        return False
    if _num(market.get("volume_fp")) < cfg.min_volume:
        return False
    closes = _parse_iso(market.get("close_time") or market.get("expiration_time")
                        or market.get("expected_expiration_time"))
    return bool(closes and closes < cfg.cutoff_date)


def fetch_kalshi_markets(cfg: ScanConfig, rep: Reporter) -> list:
    """Page open events and keep the liquid markets nested inside them.

    `/markets` has no volume filter, and most of its rows are combo contracts,
    so walking it runs past a serverless time limit. `/events` leaves those
    combos out and returns each event's markets in one page.
    """
    rep.log("Fetching Kalshi markets…")
    now = datetime.now(timezone.utc)
    if cfg.cutoff_date <= now:
        rep.log("  ⚠ Cutoff date is in the past — no Kalshi markets can qualify.")
        return []

    rep.log("  Paging open events — combo markets are left out")
    kept, seen, cursor, page = {}, 0, None, 0
    session = requests.Session()
    try:
        while True:
            rep.check_cancelled()
            params = {
                "status": "open",
                "limit": KALSHI_EVENT_PAGE,
                "with_nested_markets": "true",
                "min_close_ts": int(now.timestamp()),
            }
            if cursor:
                params["cursor"] = cursor
            data = _kalshi_get(f"{KALSHI_BASE}/events", params=params,
                               timeout=30, session=session).json()
            events = data.get("events") or []
            if not events:
                break
            page += 1
            for event in events:
                for market in event.get("markets") or []:
                    seen += 1
                    # Fees are set per series; nested markets don't carry it.
                    market.setdefault("series_ticker", event.get("series_ticker"))
                    ticker = market.get("ticker")
                    if ticker and _kalshi_eligible(market, cfg):
                        kept[ticker] = market
            rep.progress(1, f"Kalshi: page {page} · {seen:,} scanned · "
                            f"{len(kept):,} eligible")
            cursor = data.get("cursor")
            if not cursor:
                break
    finally:
        session.close()

    unique = sorted(kept.values(), key=lambda m: _num(m.get("volume_fp")), reverse=True)
    rep.log(f"  {seen:,} Kalshi markets scanned across {page} event pages")
    rep.log(f"  → {len(unique):,} eligible (vol ≥ ${cfg.min_volume:,.0f})")
    return unique


_series_fees = {}                # series ticker -> (fetched at, series info)
_series_fees_lock = threading.Lock()
SERIES_FEE_TTL = 6 * 3600        # fee schedules change rarely


def fetch_kalshi_series_fees(markets, cfg: ScanConfig, rep: Reporter):
    """Attach each market's series fee_type / fee_multiplier (one call per series).

    Only markets that get priced need this. Series already looked up in this
    process are reused. A failed lookup leaves the fields unset, and fees.py
    then assumes the standard fee.
    """
    todo = [m for m in markets if m.get("series_ticker") and "_fee_type" not in m]
    series = {m["series_ticker"] for m in todo}
    if not series:
        return
    now = time.time()
    with _series_fees_lock:
        found = {t: info for t, (at, info) in _series_fees.items()
                 if t in series and now - at < SERIES_FEE_TTL}

    def one(ticker):
        try:
            data = _kalshi_get(f"{KALSHI_BASE}/series/{ticker}", attempts=3).json()
            return ticker, data.get("series") or {}
        except Exception:
            return ticker, None

    missing = sorted(series - set(found))
    if missing:
        with ThreadPoolExecutor(max_workers=min(16, len(missing))) as ex:
            for ticker, info in ex.map(one, missing):
                if info:
                    found[ticker] = info
                    with _series_fees_lock:
                        _series_fees[ticker] = (now, info)
    for m in todo:
        info = found.get(m["series_ticker"])
        if info:
            m["_fee_type"] = info.get("fee_type")
            m["_fee_multiplier"] = info.get("fee_multiplier")
    rep.log(f"  Kalshi fee schedule loaded for {len(found)}/{len(series)} series")


# ───────────────────────────────── Orderbooks ─────────────────────────────────
#
# A book is {"yes": levels, "no": levels, "ts": fetched-at epoch seconds}.
# `levels` is the ask ladder for buying that side, cheapest first:
# [(price, size), ...]. A side that could not be fetched is absent; a side
# with nothing on offer is an empty list.

POLY_CLOB = "https://clob.polymarket.com"
POLY_CLOB_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; Market-Matcher/1.0)"}
POLY_BOOK_BATCH = 250            # tokens per POST /books call
KALSHI_BOOK_DEPTH = 15           # price levels per side; arbs rarely reach past a few


def _levels(raw) -> list:
    """Clean an ask ladder: real prices and sizes only, cheapest first."""
    out = []
    for price, size in raw or []:
        p, s = _num(price, None), _num(size, None)
        if p is not None and s is not None and 0 < p < 1 and s > 0:
            out.append((p, s))
    out.sort()
    return out


def make_book(yes=None, no=None, ts=None) -> dict:
    book = {"ts": time.time() if ts is None else ts}
    if yes is not None:
        book["yes"] = _levels(yes)
    if no is not None:
        book["no"] = _levels(no)
    return book


def best_ask(book, side):
    """(price, size) at the cheapest ask, or (None, None)."""
    levels = (book or {}).get(side) or []
    return levels[0] if levels else (None, None)


def _poly_asks(payload: dict) -> list:
    return _levels((x.get("price"), x.get("size")) for x in payload.get("asks") or [])


def fetch_poly_orderbook(token_id: str):
    """Ask ladder for one Polymarket CLOB token, or None if the call failed."""
    try:
        r = requests.get(f"{POLY_CLOB}/book", params={"token_id": token_id},
                         headers=POLY_CLOB_HEADERS, timeout=8)
        r.raise_for_status()
        return _poly_asks(r.json())
    except Exception:
        return None


def fetch_poly_orderbooks(token_ids: list) -> dict:
    """Ask ladders for many tokens in one call: {token_id: levels}.

    Tokens the call did not return are simply missing from the result.
    """
    try:
        r = requests.post(f"{POLY_CLOB}/books",
                          json=[{"token_id": t} for t in token_ids],
                          headers=POLY_CLOB_HEADERS, timeout=15)
        r.raise_for_status()
        return {b["asset_id"]: _poly_asks(b) for b in r.json()
                if isinstance(b, dict) and b.get("asset_id")}
    except Exception:
        return {}


def fetch_kalshi_orderbook(ticker: str):
    """Full-depth book for one Kalshi market, or None.

    Kalshi lists resting bids only. A NO bid at q is an offer to sell YES at
    1 − q, so each side's ask ladder is the other side's bids mirrored.
    """
    try:
        r = _kalshi_get(f"{KALSHI_BASE}/markets/{ticker}/orderbook",
                        params={"depth": KALSHI_BOOK_DEPTH}, attempts=3, timeout=5)
        ob = r.json().get("orderbook_fp") or {}

        def mirror(bids):
            return [(round(1.0 - _num(p), 4), s) for p, s in bids or []]

        book = make_book(yes=mirror(ob.get("no_dollars")),
                         no=mirror(ob.get("yes_dollars")))
        return book if book["yes"] or book["no"] else None
    except Exception:
        return None


def _clob_yes_no_tokens(m: dict):
    raw = m.get("clobTokenIds", "[]")
    try:
        t = json.loads(raw) if isinstance(raw, str) else (raw or [])
    except (json.JSONDecodeError, TypeError):
        return None
    return (t[0], t[1]) if len(t) >= 2 else None


def fetch_all_orderbooks(poly_markets, kalshi_markets, cfg: ScanConfig, rep: Reporter,
                         kalshi_only=None, step: int = 3):
    """Books for these markets, keyed by their position in each list.

    Polymarket books come in batches; a token a batch misses is retried on its
    own. `kalshi_only` limits the Kalshi side to those list positions, since
    Kalshi allows only about ten calls a second.
    """
    poly_ob, kalshi_ob = {}, {}
    tokens = {}
    for i, m in enumerate(poly_markets):
        if pair := _clob_yes_no_tokens(m):
            tokens[pair[0]], tokens[pair[1]] = (i, "yes"), (i, "no")
    kalshi_tasks = [(j, m["ticker"]) for j, m in enumerate(kalshi_markets)
                    if m.get("ticker") and (kalshi_only is None or j in kalshi_only)]
    total = len(tokens) + len(kalshi_tasks)
    if total == 0:
        return poly_ob, kalshi_ob

    rep.log(f"  Fetching {len(tokens)} Polymarket + {len(kalshi_tasks)} Kalshi "
            f"orderbooks…")

    def store_poly(token, levels):
        i, side = tokens[token]
        poly_ob.setdefault(i, {"ts": time.time()})[side] = levels

    done = 0
    ids = list(tokens)
    with ThreadPoolExecutor(max_workers=max(1, cfg.orderbook_workers)) as ex:
        futures = {}
        for start in range(0, len(ids), POLY_BOOK_BATCH):
            chunk = ids[start:start + POLY_BOOK_BATCH]
            futures[ex.submit(fetch_poly_orderbooks, chunk)] = ("poly_batch", chunk)
        for j, ticker in kalshi_tasks:
            futures[ex.submit(fetch_kalshi_orderbook, ticker)] = ("kalshi", j)

        pending = set(futures)
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                kind, ref = futures[fut]
                try:
                    res = fut.result()
                except Exception:
                    res = None
                if kind == "poly_batch":
                    got = res or {}
                    for token in ref:
                        if token in got:
                            store_poly(token, got[token])
                            done += 1
                        else:                      # retry alone before giving up
                            retry = ex.submit(fetch_poly_orderbook, token)
                            futures[retry] = ("poly_one", token)
                            pending.add(retry)
                elif kind == "poly_one":
                    done += 1
                    if res is not None:
                        store_poly(ref, res)
                else:
                    done += 1
                    if res:
                        kalshi_ob[ref] = res
            rep.progress(step, f"Orderbooks {done}/{total}", done / total)

    rep.log(f"  → {len(poly_ob)} Polymarket + {len(kalshi_ob)} Kalshi books returned")
    return poly_ob, kalshi_ob


# ────────────────────────────── Text normalization ────────────────────────────

STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or",
    "is", "be", "will", "by", "with", "from", "that", "this", "it", "as",
    "are", "was", "were", "has", "have", "had", "do", "does", "did", "not",
    "no", "but", "if", "than", "then", "so", "up", "out", "its", "their",
    "who", "which", "what", "when", "how", "any", "all", "more", "most",
    "over", "under", "into", "about", "before", "after", "between",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "2024", "2025", "2026", "2027",
    "above", "below", "least", "most", "per", "end", "date",
    "us", "st", "nd", "rd", "th", "say", "said", "says",
}

# Longer phrases first (avoid partial matches on shorter rules).
_SYNONYM_SUBS = [
    (re.compile(r'\brate[- ]cut\b', re.I), 'ratecut'),
    (re.compile(r'\brate[- ]decrease\b', re.I), 'ratecut'),
    (re.compile(r'\brate[- ]lower\b', re.I), 'ratecut'),
    (re.compile(r'\brate[- ]hike\b', re.I), 'ratehike'),
    (re.compile(r'\brate[- ]increase\b', re.I), 'ratehike'),
    (re.compile(r'\brate[- ]raise\b', re.I), 'ratehike'),
    (re.compile(r'\bno[- ]rate[- ]change\b', re.I), 'ratehold'),
    (re.compile(r'\brates?[- ]unchanged\b', re.I), 'ratehold'),
    (re.compile(r'\bno[- ]change\b', re.I), 'ratehold'),
    (re.compile(r'\b(\d+)\s*bps\b', re.I), r'\1basispoints'),
    (re.compile(r'\b(\d+)\s*basis\s*points?\b', re.I), r'\1basispoints'),
    (re.compile(r'\bconsumer[- ]price[- ]index\b', re.I), 'cpi'),
    (re.compile(r'\bnonfarm[- ]payrolls?\b', re.I), 'payroll'),
    (re.compile(r'\bjobs?[- ]report\b', re.I), 'payroll'),
    (re.compile(r'\bgross[- ]domestic[- ]product\b', re.I), 'gdp'),
    (re.compile(r'\bfederal[- ]reserve\b', re.I), 'fed'),
    (re.compile(r'\bfederal[- ]funds[- ]rate\b', re.I), 'federalfundsrate'),
    (re.compile(r'\bpersonal[- ]consumption\b', re.I), 'pce'),
    (re.compile(r'\bexecutive[- ]order\b', re.I), 'executiveorder'),
    (re.compile(r'\bsupreme[- ]court\b', re.I), 'supremecourt'),
    (re.compile(r'\battorney[- ]general\b', re.I), 'attorneygeneral'),
    (re.compile(r'\bprime[- ]minister\b', re.I), 'primeminister'),
    (re.compile(r'\btrade[- ]war\b', re.I), 'tradewar'),
    (re.compile(r'\btrade[- ]deal\b', re.I), 'tradedeal'),
    (re.compile(r'\btrade[- ]deficit\b', re.I), 'tradedeficit'),
    (re.compile(r'\belections?\b', re.I), 'election'),
    (re.compile(r'\bimpeach(?:ment|ed)?\b', re.I), 'impeach'),
    (re.compile(r'\bresign(?:ation|s)?\b', re.I), 'resign'),
    (re.compile(r'\bsanctions?\b', re.I), 'sanction'),
    (re.compile(r'\btariffs?\b', re.I), 'tariff'),
    (re.compile(r'\bapprov(?:al|ed)\b', re.I), 'approve'),
    (re.compile(r'\bnomin(?:ation|ated)\b', re.I), 'nominate'),
    (re.compile(r'\bconfirm(?:ation|ed)\b', re.I), 'confirm'),
    (re.compile(r'\bvetoed?\b', re.I), 'veto'),
    (re.compile(r'\bpayrolls\b', re.I), 'payroll'),
    (re.compile(r'\brecessions?\b', re.I), 'recession'),
    (re.compile(r'\bdeficits?\b', re.I), 'deficit'),
]

_RATE_CONTEXT = re.compile(
    r'\b(rate|rates|fed|fomc|bps|basis\s*points?|federal\s*funds|'
    r'interest|monetary|policy|hike|cut|central\s*bank)\b', re.I)


def _apply_rate_synonyms(text: str) -> str:
    """Only substitute rate-action verbs when a rate-context word is present."""
    if not _RATE_CONTEXT.search(text):
        return text
    for pat, rep in (
        (r'\bcuts?\b', 'ratecut'), (r'\bdecrease[sd]?\b', 'ratecut'),
        (r'\blowere?d?\b', 'ratecut'), (r'\bhikes?\b', 'ratehike'),
        (r'\bincrease[sd]?\b', 'ratehike'), (r'\braise[sd]?\b', 'ratehike'),
        (r'\bunchanged\b', 'ratehold'), (r'\bpaused?\b', 'ratehold'),
    ):
        text = re.sub(pat, rep, text, flags=re.I)
    return text


def normalize_for_matching(text: str) -> str:
    for pat, rep in _SYNONYM_SUBS:
        text = pat.sub(rep, text)
    return _apply_rate_synonyms(text)


def tokenize(text: str) -> list:
    words = re.findall(r"[a-z]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


_MONTHS = ("january february march april may june july august september "
           "october november december").split()
_MONTH_MAP = {**{m: i + 1 for i, m in enumerate(_MONTHS)},
              **{m[:3]: i + 1 for i, m in enumerate(_MONTHS)}}

_ENTITY_PATTERNS = {
    "people": re.compile(
        r'\b(?:trump|biden|harris|musk|powell|yellen|desantis|pence|obama|'
        r'zelensky|zelenskyy|putin|xi\s+jinping|modi|trudeau|macron|'
        r'starmer|sunak|netanyahu|gensler|bezos|zuckerberg|altman)\b', re.I),
    "tickers": re.compile(
        r'\b(?:s&p\s*500|s&p|sp500|spx|nasdaq|qqq|dow\s*jones|djia|'
        r'btc|bitcoin|eth|ethereum|sol|solana|xrp|doge|dogecoin|'
        r'aapl|apple|msft|microsoft|goog|google|googl|amzn|amazon|'
        r'tsla|tesla|nvda|nvidia|meta|nflx|netflix)\b', re.I),
    "orgs": re.compile(
        r'\b(?:fed(?:eral\s+reserve)?|ecb|boe|boj|sec|cftc|nato|'
        r'who|imf|opec|un|eu|fomc|fbi|cia|doj|epa|fda|'
        r'spacex|openai|anthropic)\b', re.I),
    "countries": re.compile(
        r'\b(?:united\s+states|usa|u\.s\.?|china|russia|ukraine|india|'
        r'japan|germany|france|uk|united\s+kingdom|brazil|'
        r'canada|australia|mexico|iran|israel|taiwan|'
        r'north\s+korea|south\s+korea|saudi\s+arabia|turkey)\b', re.I),
}

_ENTITY_NORMALIZE = {
    "s&p 500": "sp500", "s&p": "sp500", "spx": "sp500",
    "dow jones": "djia",
    "bitcoin": "btc", "ethereum": "eth", "solana": "sol", "dogecoin": "doge",
    "apple": "aapl", "microsoft": "msft", "google": "goog", "googl": "goog",
    "amazon": "amzn", "tesla": "tsla", "nvidia": "nvda", "netflix": "nflx",
    "federal reserve": "fed",
    "united states": "usa", "u.s.": "usa", "u.s": "usa",
    "united kingdom": "uk",
    "north korea": "north_korea", "south korea": "south_korea",
    "saudi arabia": "saudi_arabia",
    "xi jinping": "xi", "zelensky": "zelenskyy",
}


def extract_entities(text: str) -> set:
    entities = set()
    for pattern in _ENTITY_PATTERNS.values():
        for m in pattern.finditer(text):
            raw = m.group(0).lower().strip()
            entities.add(_ENTITY_NORMALIZE.get(raw, raw))
    return entities


_ABOVE_RE = re.compile(r'\b(?:above|over|more\s+than|greater\s+than|exceed(?:s|ing)?|'
                       r'at\s+least|higher\s+than)\b', re.I)
_BELOW_RE = re.compile(r'\b(?:below|under|less\s+than|fewer\s+than|at\s+most|'
                       r'lower\s+than)\b', re.I)


def _extract_direction(text: str):
    a, b = bool(_ABOVE_RE.search(text)), bool(_BELOW_RE.search(text))
    return ("above" if a else "below") if a != b else None


def _sets_clash(a: set, b: set) -> bool:
    return bool(a and b and not a & b)


def numbers_compatible(a: str, b: str) -> bool:
    a_nums = set(re.findall(r'\d+', a))
    b_nums = set(re.findall(r'\d+', b))
    if a_nums and b_nums and not (a_nums & b_nums):
        return False
    if a_nums & b_nums:
        dir_a, dir_b = _extract_direction(a), _extract_direction(b)
        if dir_a and dir_b and dir_a != dir_b:
            return False
    return True


def months_compatible(a: str, b: str) -> bool:
    def months(t):
        return {_MONTH_MAP[w] for w in re.findall(r'[a-z]+', t.lower())
                if w in _MONTH_MAP}
    return not _sets_clash(months(a), months(b))


def kalshi_title(market: dict) -> str:
    title = market.get("title") or market.get("question") or ""
    sub = (market.get("yes_sub_title") or market.get("subtitle") or "").strip()
    return f"{title}  [{sub}]" if sub and sub.lower() not in title.lower() else title


def poly_title(market: dict) -> str:
    return market.get("question") or market.get("title") or ""


# ─────────────────────────────── TF-IDF matching ──────────────────────────────

def find_matches(poly_markets, kalshi_markets, cfg: ScanConfig, rep: Reporter) -> list:
    if not poly_markets or not kalshi_markets:
        rep.log("  Nothing to match (one side is empty)")
        return []

    poly_texts = [poly_title(m) for m in poly_markets]
    kalshi_texts = [kalshi_title(m) for m in kalshi_markets]

    rep.log(f"  Building TF-IDF matrix ({len(poly_texts):,} × {len(kalshi_texts):,})…")
    all_norm = [normalize_for_matching(t) for t in poly_texts + kalshi_texts]
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words=list(STOPWORDS),
                          sublinear_tf=True)
    tfidf = vec.fit_transform(all_norm)
    sim_matrix = sk_cosine_sim(tfidf[:len(poly_texts)], tfidf[len(poly_texts):])

    candidates = np.argwhere(sim_matrix >= cfg.tfidf_threshold)
    rep.log(f"  {len(candidates):,} pairs above TF-IDF threshold "
            f"({cfg.tfidf_threshold:.2f})")
    rep.check_cancelled()

    rep.log("  Filtering on numbers, months, entities, dates, Jaccard…")
    poly_tok = [set(tokenize(t)) for t in poly_texts]
    kalshi_tok = [set(tokenize(t)) for t in kalshi_texts]
    poly_ent = [extract_entities(t) for t in poly_texts]
    kalshi_ent = [extract_entities(t) for t in kalshi_texts]
    poly_dates = [_parse_iso(m.get("endDate") or m.get("end_date_iso"))
                  for m in poly_markets]
    kalshi_dates = [_parse_iso(m.get("close_time") or m.get("expiration_time")
                               or m.get("expected_expiration_time"))
                    for m in kalshi_markets]

    matches = []
    rej = {"numbers": 0, "months": 0, "entities": 0, "dates": 0, "score": 0}

    for i, j in candidates:
        pt, kt = poly_texts[i], kalshi_texts[j]
        if not numbers_compatible(pt, kt):
            rej["numbers"] += 1
            continue
        if not months_compatible(pt, kt):
            rej["months"] += 1
            continue
        if _sets_clash(poly_ent[i], kalshi_ent[j]):
            rej["entities"] += 1
            continue
        pd, kd = poly_dates[i], kalshi_dates[j]
        if pd and kd and abs((pd - kd).days) > cfg.max_date_gap_days:
            rej["dates"] += 1
            continue

        p_set, k_set = poly_tok[i], kalshi_tok[j]
        union = p_set | k_set
        jaccard = len(p_set & k_set) / len(union) if union else 0.0
        combined = 0.7 * float(sim_matrix[i, j]) + 0.3 * jaccard
        if combined < cfg.combined_threshold:
            rej["score"] += 1
            continue

        matches.append((poly_markets[i], kalshi_markets[j], p_set & k_set, combined))

    matches.sort(key=lambda x: x[3], reverse=True)
    rep.log("  Rejected: " + ", ".join(f"{v:,} {k}" for k, v in rej.items()))
    rep.log(f"  → {len(matches):,} candidate pairs passed the pre-filter")
    return matches



# ───────────────────────────── Groq LLM verification ──────────────────────────
#
# Two passes keep the token bill small, which is what bounds speed on a
# rate-limited key. A cheap title screen throws out pairs that are plainly
# different questions; only the survivors get the strict pass, which reads
# both markets' resolution rules. A pair is verified only by the strict pass.

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
VERIFY_PROMPT_VERSION = "3"      # bump when the prompts change: it voids the cache
GROQ_ASSUMED_TPM = 6000          # until the first response reports the real limit
GROQ_BACKOFF_BASE = 2.0          # seconds; doubles on each rate-limit retry
GROQ_BACKOFF_MAX = 60.0
LLM_HARD_CAP_SECONDS = 900.0     # ceiling when no budget is set
SCREEN_BATCH_PAIRS = 24          # pairs per title-screen request, and per slice
POLY_RULES_CLIP, KALSHI_RULES_CLIP, KALSHI_EXTRA_CLIP = 420, 320, 200


def pair_id(poly, kalshi):
    return id(poly), id(kalshi)


def _clip(text, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " …"


def _screen_block(i, poly, kalshi) -> str:
    return (f"Pair {i}:\n"
            f"  Polymarket: {poly_title(poly)} (ends {(poly.get('endDate') or '?')[:10]})\n"
            f"  Kalshi:     {kalshi_title(kalshi)} "
            f"(closes {(kalshi.get('close_time') or '?')[:10]})")


def _strict_block(i, poly, kalshi) -> str:
    outcomes = poly.get("outcomes") or '["Yes", "No"]'
    lines = [
        f"Pair {i}:",
        f"  Polymarket question: {poly_title(poly)}",
        f"  Polymarket outcomes: {outcomes} (the first one is the YES side)",
        f"  Polymarket end date: {(poly.get('endDate') or 'unknown')[:16]}",
        f"  Polymarket rules: {_clip(poly.get('description'), POLY_RULES_CLIP) or 'not given'}",
    ]
    if poly.get("resolutionSource"):
        lines.append(f"  Polymarket source: {_clip(poly['resolutionSource'], 120)}")
    lines += [
        f"  Kalshi market: {kalshi_title(kalshi)}",
        f"  Kalshi closes: {(kalshi.get('close_time') or 'unknown')[:16]}",
        f"  Kalshi rules: {_clip(kalshi.get('rules_primary'), KALSHI_RULES_CLIP) or 'not given'}",
    ]
    extra = _clip(kalshi.get("rules_secondary"), KALSHI_EXTRA_CLIP)
    if extra:
        lines.append(f"  Kalshi also: {extra}")
    return "\n".join(lines)


_SCREEN_PROMPT = (
    "You are a prediction-market analyst. A text-similarity algorithm flagged "
    "the candidate pairs below as possibly the same question on Polymarket and "
    "Kalshi. Screen out the ones that are clearly different questions.\n\n"
    "{pairs}\n\n"
    "Set match=false when the pair has conflicting numbers or thresholds, "
    "different people, teams, countries or assets, clearly different dates or "
    "deadlines, or asks fundamentally different things. Set match=true when "
    "the two could plausibly be the same bet; a later, stricter check reads "
    "the full rules. A year stated in one title and omitted in the other is "
    "not a conflict.\n\n"
    "Return ONLY a JSON array (no markdown fences, no explanation) with one "
    "element for EVERY pair:\n"
    '{{"pair_index": <int>, "match": <true|false>}}'
)

_STRICT_PROMPT = (
    "You are a prediction-market analyst checking whether pairs of markets on "
    "Polymarket and Kalshi are the SAME bet. Money is traded on your answer: "
    "buying YES on one venue and NO on the other is only risk-free if both "
    "markets always resolve the same way.\n\n"
    "{pairs}\n\n"
    "For each pair, set match=true ONLY if Polymarket's YES side and Kalshi's "
    "YES always resolve identically. Set match=false if any of these differ:\n"
    "- the event, or the person, team, company, country or asset it is about;\n"
    "- a number, threshold, strike or range;\n"
    "- the deadline or the measurement time or window. 'On' a date is not "
    "'by' a date, 'reaches at any time' is not 'is above at a set time', and "
    "a price read at a different time of day is a different bet;\n"
    "- the settlement source or method, when that could change the result;\n"
    "- the direction: if Polymarket YES corresponds to Kalshi NO, that is "
    "match=false;\n"
    "- the scope: one market is only a part or a superset of the other, or "
    "one asks who is FIRST or NEXT among several candidates while the other "
    "asks about one candidate alone.\n"
    "Different wording is fine when the resolution is identical, and a year "
    "stated on one side and implied on the other is fine. If the rules shown "
    "are not enough to be sure, set match=false.\n\n"
    "Return ONLY a JSON array (no markdown fences, no explanation) with one "
    "element for EVERY pair:\n"
    '{{"pair_index": <int>, "match": <true|false>, '
    '"confidence": <float 0-1 that they are the same bet>, '
    '"shared_keywords": [<strings>]}}'
)


def _estimate_tokens(prompt: str, n_pairs: int, per_pair_out: int) -> int:
    """Rough request cost — input, answer and a little reasoning."""
    return int(len(prompt) / 3.6) + per_pair_out * n_pairs + 150


def _parse_duration(text) -> float:
    """Groq reset headers: '2.67s', '1m26.4s', '120ms'."""
    total, found = 0.0, False
    for value, unit in re.findall(r"([\d.]+)(ms|s|m|h)", str(text or "")):
        found = True
        total += float(value) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
    return total if found else _num(text, 0.0)


def _sleep(rep: Reporter, seconds: float):
    """Sleep in short slices so a cancel or time limit is noticed quickly."""
    end = time.monotonic() + seconds
    while True:
        rep.check_cancelled()
        left = end - time.monotonic()
        if left <= 0:
            return
        time.sleep(min(left, 0.5))


class _GroqLimiter:
    """Token bucket kept in step with Groq's own rate-limit headers.

    Groq refills a per-minute token allowance continuously. Each request
    reserves its estimated cost up front, and every response corrects the
    balance from `x-ratelimit-remaining-tokens`, so concurrent requests stay
    inside the limit the key actually has rather than a guessed pause.
    """

    def __init__(self, hard_deadline: float = None):
        self._lock = threading.Lock()
        self.limit = float(GROQ_ASSUMED_TPM)
        self._tokens = self.limit
        self._stamp = time.monotonic()
        self._in_flight = 0.0
        self._blocked_until = 0.0
        self.hard_deadline = hard_deadline
        self.requests = 0
        self.rate_limited = 0
        self.tokens_used = 0
        self._scale = 1.2            # actual / estimated cost, learned from replies

    def _balance(self, now):
        return min(self.limit, self._tokens + (now - self._stamp) * self.limit / 60.0)

    def acquire(self, cost: float, rep: Reporter, deadline: float = None) -> bool:
        """Wait for `cost` tokens and return the amount reserved.

        Returns 0 if they cannot arrive before a deadline.
        """
        limits = [d for d in (deadline, self.hard_deadline) if d is not None]
        cutoff = min(limits) if limits else None
        while True:
            rep.check_cancelled()
            with self._lock:
                now = time.monotonic()
                # An oversized request must still fit inside the bucket.
                cost = min(cost * self._scale, self.limit * 0.9)
                balance = self._balance(now)
                delay = max(0.0, self._blocked_until - now,
                            (cost - balance) * 60.0 / self.limit)
                if delay <= 0:
                    self._tokens, self._stamp = balance - cost, now
                    self._in_flight += cost
                    self.requests += 1
                    return cost
            if cutoff is not None and now + delay > cutoff:
                return 0
            time.sleep(min(delay, 0.5))

    def settle(self, cost: float, headers=None, used: float = None,
               estimate: float = None, refund: bool = False):
        """A request finished: release its reservation and adopt Groq's numbers."""
        headers = headers or {}
        with self._lock:
            now = time.monotonic()
            self._in_flight = max(0.0, self._in_flight - cost)
            if refund:                  # no reply at all: Groq never counted it
                self._tokens = min(self.limit, self._balance(now) + cost)
                self._stamp = now
            if used and estimate:
                self.tokens_used += int(used)
                # Lean toward the larger figure: running out mid-minute costs a backoff.
                self._scale = min(3.0, max(0.5, 0.5 * self._scale + 0.5 * used / estimate))
            limit = _num(headers.get("x-ratelimit-limit-tokens"), 0)
            remaining = headers.get("x-ratelimit-remaining-tokens")
            if limit > 0:
                self.limit = limit
            if remaining is not None:
                # Requests still in flight are not in Groq's figure yet.
                self._tokens = min(self.limit, _num(remaining)) - self._in_flight
                self._stamp = now

    def block(self, seconds: float):
        """Rate limited: hold every worker, not only the one that was told."""
        with self._lock:
            self.rate_limited += 1
            self._blocked_until = max(self._blocked_until, time.monotonic() + seconds)


def _groq_json(prompt: str, cost: int, cfg: ScanConfig, rep: Reporter,
               limiter: _GroqLimiter, deadline: float = None, max_attempts: int = 5):
    """The model's JSON array, or None if the call did not succeed.

    None means "no answer". Callers must leave those pairs unverified; a
    timeout, an error or a rate limit is never read as a match.
    """
    for attempt in range(max_attempts):
        reserved = limiter.acquire(cost, rep, deadline)
        if not reserved:
            return None
        backoff = min(GROQ_BACKOFF_MAX, GROQ_BACKOFF_BASE * 2 ** attempt)
        backoff += random.uniform(0, backoff * 0.25)          # jitter
        try:
            r = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {cfg.groq_api_key}",
                         "Content-Type": "application/json"},
                json={"model": cfg.groq_model, "temperature": 0,
                      "reasoning_effort": "low", "include_reasoning": False,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=60,
            )
        except requests.RequestException as exc:
            limiter.settle(reserved, refund=True)
            rep.log(f"    Groq request failed ({type(exc).__name__}) — "
                    f"attempt {attempt + 1}/{max_attempts}")
            if attempt < max_attempts - 1:
                _sleep(rep, backoff)
            continue

        body = None
        if r.status_code == 200:
            try:
                body = r.json()
            except ValueError:
                body = None
        used = ((body or {}).get("usage") or {}).get("total_tokens") if isinstance(body, dict) else None
        limiter.settle(reserved, r.headers, used, cost)
        if r.status_code in (401, 403):
            raise GroqAuthError(r.status_code)
        if r.status_code == 429:
            # Groq says exactly how long to wait; fall back to exponential
            # backoff with jitter when it does not.
            told = _parse_duration(r.headers.get("retry-after"))
            pause = told * 1.1 + 0.25 if told > 0 else backoff
            limiter.block(pause)
            rep.log(f"    Groq rate limit — backing off {pause:.1f}s "
                    f"(attempt {attempt + 1}/{max_attempts})")
            if deadline is not None and time.monotonic() + pause > deadline:
                return None
            continue
        if r.status_code >= 500:
            rep.log(f"    Groq server error {r.status_code} — "
                    f"attempt {attempt + 1}/{max_attempts}")
            if attempt < max_attempts - 1:
                _sleep(rep, backoff)
            continue
        if r.status_code >= 400:
            rep.log(f"    ⚠ Groq rejected the request (HTTP {r.status_code}): "
                    f"{r.text[:160]}")
            return None
        try:
            raw = body["choices"][0]["message"]["content"].strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw)
            result = json.loads(raw)
            if isinstance(result, dict):
                result = next((v for v in result.values() if isinstance(v, list)), None)
            if not isinstance(result, list):
                raise ValueError("expected a JSON array")
            return result
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
            rep.log(f"    Groq returned unusable JSON ({exc}) — "
                    f"attempt {attempt + 1}/{max_attempts}")
    return None


def _batches(pairs, block, template, max_pairs, max_tokens, per_pair_out):
    """Split pairs into requests that stay under the pair and token ceilings."""
    out, current = [], []
    for pair in pairs:
        trial = current + [pair]
        text = "\n".join(block(i, p, k) for i, (p, k, *_rest) in enumerate(trial))
        cost = _estimate_tokens(template.format(pairs=text), len(trial), per_pair_out)
        if current and (len(trial) > max_pairs or cost > max_tokens):
            out.append(current)
            current = [pair]
        else:
            current = trial
    if current:
        out.append(current)
    return out


def _judge(pairs, strict: bool, cfg: ScanConfig, rep: Reporter,
           limiter: _GroqLimiter, deadline: float = None) -> dict:
    """Ask the model about these pairs: {pair_id: verdict} for answered pairs only.

    A pair missing from the result got no usable answer (failed call, skipped
    for time, or left out of the reply) and stays unverified.
    """
    if not pairs:
        return {}
    block = _strict_block if strict else _screen_block
    template = _STRICT_PROMPT if strict else _SCREEN_PROMPT
    per_pair_out = 130 if strict else 60        # answer plus the model's reasoning
    batches = _batches(pairs, block, template,
                       max(1, cfg.groq_batch_size) if strict else SCREEN_BATCH_PAIRS,
                       cfg.groq_batch_tokens, per_pair_out)

    def run(batch):
        text = "\n".join(block(i, p, k) for i, (p, k, *_rest) in enumerate(batch))
        prompt = template.format(pairs=text)
        cost = _estimate_tokens(prompt, len(batch), per_pair_out)
        rows = _groq_json(prompt, cost, cfg, rep, limiter, deadline)
        verdicts = {}
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            idx, match = row.get("pair_index"), row.get("match")
            if type(idx) is not int or not 0 <= idx < len(batch) or type(match) is not bool:
                continue
            p, k = batch[idx][0], batch[idx][1]
            keywords = row.get("shared_keywords")
            verdicts[pair_id(p, k)] = {
                "match": match,
                "confidence": min(1.0, max(0.0, _num(row.get("confidence"), 0.0))),
                "shared": {str(w) for w in keywords} if isinstance(keywords, list) else set(),
            }
        return verdicts

    answered = {}
    for batch in batches:
        answered.update(run(batch))
    return answered


def verify_matches_with_llm(direct: list, screened: list, cfg: ScanConfig,
                            rep: Reporter) -> dict:
    """Verify candidate pairs in the order given, most promising first.

    Returns {pair_id: {"match", "confidence", "shared"}} for every pair the
    model gave a verdict on; anything absent is unverified.

    `direct` pairs (a plausible edge right now) go first and straight to the
    strict rule check. `screened` pairs follow, a slice at a time: a cheap
    title screen, then the strict check on the survivors. Work stops when
    `llm_budget_seconds` runs out, and the rest wait for the next scan, which
    reuses every verdict made here.
    """
    required, optional = direct, screened
    total = len(required) + len(optional)
    if not total:
        rep.log("  Nothing new to verify.")
        return {}
    if not cfg.llm_enabled:
        rep.log("  LLM verification disabled (no Groq API key) — "
                f"{total:,} pairs stay unverified.")
        return {}

    start = time.monotonic()
    budget_end = start + cfg.llm_budget_seconds if cfg.llm_budget_seconds > 0 else None
    limiter = _GroqLimiter(hard_deadline=budget_end or start + LLM_HARD_CAP_SECONDS)
    budget = f"{cfg.llm_budget_seconds:.0f}s budget" if budget_end else "no time limit"
    rep.log(f"  {len(required):,} pairs with a plausible edge, then {len(optional):,} "
            f"others ({budget}) — model {cfg.groq_model}, up to "
            f"{cfg.groq_concurrency} requests at once")
    verdicts, lock = {}, threading.Lock()
    counts = {"screened": 0, "passed": 0, "checked": 0, "same": 0}
    auth_failed = threading.Event()

    def record(screened, survivors, strict):
        with lock:
            for key, verdict in screened.items():
                if not verdict["match"]:
                    verdicts[key] = {"match": False, "confidence": 0.0, "shared": set()}
            verdicts.update(strict)
            counts["screened"] += len(screened)
            counts["passed"] += len(survivors)
            counts["checked"] += len(strict)
            counts["same"] += sum(1 for v in strict.values() if v["match"])
            rep.progress(4, f"LLM · {len(verdicts):,}/{total:,} pairs judged",
                         len(verdicts) / total)

    def guarded(work):
        def run(chunk):
            if auth_failed.is_set():
                return
            try:
                work(chunk)
            except GroqAuthError as exc:
                if not auth_failed.is_set():
                    auth_failed.set()
                    rep.log(f"  ⚠ Groq rejected the API key (HTTP {exc}) — remaining "
                            "pairs stay unverified. Get a fresh key at "
                            "https://console.groq.com/keys")
        return run

    def strict_only(chunk):
        if budget_end and time.monotonic() > budget_end:
            return
        record({}, [], _judge(chunk, True, cfg, rep, limiter))

    def screen_then_strict(chunk):
        if budget_end and time.monotonic() > budget_end:
            return
        screened = _judge(chunk, False, cfg, rep, limiter, budget_end)
        survivors = [pair for pair in chunk
                     if screened.get(pair_id(pair[0], pair[1]), {}).get("match")]
        record(screened, survivors, _judge(survivors, True, cfg, rep, limiter, budget_end))

    def in_parallel(work, chunks):
        with ThreadPoolExecutor(max_workers=max(1, cfg.groq_concurrency)) as ex:
            futures = [ex.submit(guarded(work), chunk) for chunk in chunks]
            try:
                for fut in futures:
                    fut.result()
            except BaseException:
                for fut in futures:
                    fut.cancel()
                raise

    size = max(1, cfg.groq_batch_size)
    in_parallel(strict_only, [required[i:i + size] for i in range(0, len(required), size)])
    rep.check_cancelled()
    in_parallel(screen_then_strict, [optional[i:i + SCREEN_BATCH_PAIRS]
                                     for i in range(0, len(optional), SCREEN_BATCH_PAIRS)])

    waiting = total - len(verdicts)
    rep.log(f"    {counts['screened']:,} title-screened ({counts['passed']:,} plausible), "
            f"{counts['checked']:,} rule-checked, {counts['same']:,} same bet")
    rep.log(f"  → {len(verdicts):,} verdicts from {limiter.requests} Groq request(s), "
            f"{limiter.tokens_used:,} tokens"
            + (f", {limiter.rate_limited} rate-limit backoff(s)" if limiter.rate_limited else "")
            + (f"; {waiting:,} pairs left for the next scan" if waiting else ""))
    return verdicts


# ──────────────────────────────── Arbitrage scan ──────────────────────────────

# Size used to quote fees when a book shows no profitable size.
FEE_QUOTE_SIZE = 100
SIZE_EPS = 1e-9


def _direction_quote(poly, kalshi, p_side, p_levels, k_side, k_levels):
    """Price one way round: buy `p_side` on Polymarket and `k_side` on Kalshi.

    Walks both ask ladders together, taking size while one more pair is still
    worth more than its prices plus fees. The result covers everything that
    can be filled at a profit, not only the best ask.
    """
    if not p_levels or not k_levels:
        return None

    def unit_edge(p, k):
        return (1.0 - p - k - marginal_fee("Polymarket", poly, p)
                - marginal_fee("Kalshi", kalshi, k))

    fills, i, j = [], 0, 0
    p_left, k_left = p_levels[0][1], k_levels[0][1]
    while i < len(p_levels) and j < len(k_levels):
        p, k = p_levels[i][0], k_levels[j][0]
        if unit_edge(p, k) <= 0:
            break
        qty = min(p_left, k_left)
        fills.append((qty, p, k))
        p_left, k_left = p_left - qty, k_left - qty
        if p_left <= SIZE_EPS:
            i += 1
            p_left = p_levels[i][1] if i < len(p_levels) else 0.0
        if k_left <= SIZE_EPS:
            j += 1
            k_left = k_levels[j][1] if j < len(k_levels) else 0.0

    (p_best, p_best_size), (k_best, k_best_size) = p_levels[0], k_levels[0]
    if fills:
        size = sum(q for q, _p, _k in fills)
        p_fills = [(q, p) for q, p, _k in fills]
        k_fills = [(q, k) for q, _p, k in fills]
        levels = len({p for _q, p, _k in fills}) + len({k for _q, _p, k in fills}) - 1
    else:                       # nothing fills at a profit: quote the best asks
        size = min(p_best_size, k_best_size) or FEE_QUOTE_SIZE
        p_fills, k_fills, levels = [(size, p_best)], [(size, k_best)], 0

    p_fee = order_fee("Polymarket", poly, p_fills)
    k_fee = order_fee("Kalshi", kalshi, k_fills)
    p_avg = sum(q * p for q, p in p_fills) / size
    k_avg = sum(q * k for q, k in k_fills) / size
    fees_per_pair = (p_fee + k_fee) / size
    gross = 1.0 - p_avg - k_avg
    margin = gross - fees_per_pair
    cost = p_avg + k_avg + fees_per_pair
    min_size = _num(poly.get("orderMinSize"), 0.0)
    return {
        "poly": poly, "kalshi": kalshi,
        "a_plat": "Polymarket", "a_side": p_side,
        "a_price": p_avg, "a_best": p_best, "a_ask_size": p_best_size,
        "b_plat": "Kalshi", "b_side": k_side,
        "b_price": k_avg, "b_best": k_best, "b_ask_size": k_best_size,
        "top_gross": 1.0 - p_best - k_best,
        "gross_margin": gross,
        "a_fee": p_fee, "b_fee": k_fee, "fee_size": size,
        "fees_per_pair": fees_per_pair,
        "fees_estimated": not (fee_known("Polymarket", poly)
                               and fee_known("Kalshi", kalshi)),
        "prob_sum": p_avg + k_avg, "cost": cost,
        "margin": margin,
        "roi": margin / cost if cost > 0 else None,
        "levels": levels,
        "max_contracts": size if fills else None,
        "max_profit": margin * size if fills else None,
        "min_size": min_size,
        # Fillable at a profit, and big enough for Polymarket to accept the order.
        "executable": bool(fills) and margin > 0 and size + SIZE_EPS >= min_size,
    }


def pair_quotes(poly, kalshi, pbook, kbook, cfg: ScanConfig, now: float = None) -> list:
    """Both ways to trade a pair, priced from live books (zero, one or two quotes)."""
    if not pbook or not kbook:
        return []
    now = time.time() if now is None else now
    age = max(0.0, now - min(pbook.get("ts", now), kbook.get("ts", now)))
    quotes = []
    for p_side, k_side in (("yes", "no"), ("no", "yes")):
        quote = _direction_quote(poly, kalshi, p_side.upper(), pbook.get(p_side),
                                 k_side.upper(), kbook.get(k_side))
        if quote:
            quote["age"] = age
            quote["stale"] = age > cfg.stale_quote_seconds
            quote["poly_book"], quote["kalshi_book"] = pbook, kbook
            quotes.append(quote)
    return quotes


def best_quote(poly, kalshi, pbook, kbook, cfg: ScanConfig, now: float = None):
    """The better of a pair's two directions, or None if it cannot be priced."""
    quotes = pair_quotes(poly, kalshi, pbook, kbook, cfg, now)
    return max(quotes, key=lambda q: q["margin"]) if quotes else None


def quote_state(quote, cfg: ScanConfig) -> str:
    """Where a priced pair stands: 'live', 'stale', 'small', 'near' or 'none'."""
    if quote is None:
        return "unpriced"
    if quote["margin"] > 0 and quote["max_contracts"]:
        if quote["stale"]:
            return "stale"
        if not quote["executable"]:
            return "small"
        return "live" if quote["margin"] >= cfg.min_margin else "near"
    return "near" if quote["margin"] >= -cfg.near_margin else "none"


def find_arbitrage(matches, poly_markets, kalshi_markets,
                   poly_ob, kalshi_ob, cfg: ScanConfig, rep: Reporter) -> list:
    """Live cross-platform arbitrage among `matches`, from order books only.

    Pass verified pairs: every pair given here can come back as an arb. A pair
    is one when YES on one venue plus NO on the other, plus both venues' taker
    fees, costs < $1.00 across the size that can actually be filled, since
    exactly one of the two legs pays out $1.00 at resolution. Stale quotes and
    sizes too small to trade are counted but never returned.
    """
    p_ix = {id(m): i for i, m in enumerate(poly_markets)}
    k_ix = {id(m): i for i, m in enumerate(kalshi_markets)}
    now_dt, now = datetime.now(timezone.utc), time.time()
    arbs, fee_killed, stale, small = [], 0, 0, 0

    for poly, kalshi, shared, score in matches:
        pi, ki = p_ix.get(id(poly)), k_ix.get(id(kalshi))
        if pi is None or ki is None:
            continue
        for quote in pair_quotes(poly, kalshi, poly_ob.get(pi), kalshi_ob.get(ki),
                                 cfg, now):
            if quote["top_gross"] <= 0:
                continue
            if quote["margin"] <= 0 or not quote["max_contracts"]:
                fee_killed += 1
                continue
            if quote["stale"]:
                stale += 1
                continue
            if not quote["executable"]:
                small += 1
                continue
            if quote["margin"] < cfg.min_margin:
                continue
            days = pair_days(poly, kalshi, now_dt)
            quote.update(days=days, apr=annualized_return(quote["roi"], days),
                         shared=shared, match_score=score)
            arbs.append(quote)

    # Rank by annualized return so a quick 2% beats a 3% locked up for months.
    # Arbs with no known resolution date fall to the bottom, ordered by edge.
    arbs.sort(key=lambda x: (x["apr"] is not None, x["apr"] or 0, x["margin"]),
              reverse=True)
    rep.log(f"  {len(arbs)} arbitrage opportunit{'y' if len(arbs) == 1 else 'ies'} "
            f"found on executable order-book depth, net of taker fees")
    if fee_killed:
        rep.log(f"  {fee_killed} more had a gross edge that fees wiped out")
    if stale:
        rep.log(f"  {stale} more were left out because their quotes were older "
                f"than {cfg.stale_quote_seconds:.0f}s")
    if small:
        rep.log(f"  {small} more were below Polymarket's minimum order size")
    return arbs


# ──────────────────────────────── Full pipeline ───────────────────────────────

# A real cross-venue mispricing on liquid markets is small. Two *different*
# markets, though, nearly always look mispriced against each other, so a huge
# apparent edge points to a bad match. Pairs with a modest edge are verified first.
PLAUSIBLE_EDGE = 0.10


class ScanResult:
    def __init__(self, cfg: ScanConfig):
        self.config = cfg
        self.started_at = datetime.now(timezone.utc)
        self.finished_at = None
        self.prices_updated_at = None
        self.poly = []
        self.kalshi = []
        self.matches = []
        self.verified = []
        self.arbs = []
        self.poly_ob = {}
        self.kalshi_ob = {}
        self.matched_poly = []
        self.matched_kalshi = []
        self.live_kalshi = None      # Kalshi markets a refresh re-quotes (None = all)
        self.pair_status = {}        # pair_id -> "verified" | "rejected"; absent = unverified
        self.verify_cache = {}       # cache entries for this scan's pairs
        self.timings = {}            # seconds per stage
        self.llm_stats = {}
        self.arb_source_label = "LLM Verified"
        self.llm_used = False

    @property
    def duration_seconds(self) -> float:
        end = self.finished_at or datetime.now(timezone.utc)
        return (end - self.started_at).total_seconds()


def _timed(fn, *args):
    start = time.monotonic()
    return fn(*args), time.monotonic() - start


def kalshi_listed_book(market: dict, ts: float = None):
    """A one-level book from the quote Kalshi sends with its market list.

    Good enough to rank unverified pairs without spending a rate-limited call
    on each. Nothing is ever reported as arbitrage from it: verified pairs get
    a real full-depth book.
    """
    def side(price_key, size_key):
        price = _num(market.get(price_key), None)
        if price is None or not 0 < price < 1:
            return []
        return [(price, _num(market.get(size_key), 0.0) or FEE_QUOTE_SIZE)]

    # Buying NO lifts resting YES bids, so the NO ask's size is the YES bid size.
    yes = side("yes_ask_dollars", "yes_ask_size_fp")
    no = side("no_ask_dollars", "yes_bid_size_fp")
    return make_book(yes=yes, no=no, ts=ts) if yes or no else None


def run_scan(cfg: ScanConfig, rep: Reporter = None) -> ScanResult:
    rep = rep or Reporter()
    res = ScanResult(cfg)
    timings = res.timings
    clock = time.monotonic
    t_start = clock()

    def stage_done(name, since):
        timings[name] = round(clock() - since, 2)
        rep.log(f"  ⏱ {name.replace('_', ' ')}: {timings[name]:.1f}s")

    # 1 ── discovery ──────────────────────────────────────────────────────────
    rep.step(1, "Fetching Polymarket + Kalshi markets")
    t = clock()
    with ThreadPoolExecutor(max_workers=2) as ex:
        poly_fut = ex.submit(_timed, fetch_poly_markets, cfg, rep)
        kalshi_fut = ex.submit(_timed, fetch_kalshi_markets, cfg, rep)
        res.poly, timings["fetch_polymarket"] = poly_fut.result()
        res.kalshi, timings["fetch_kalshi"] = kalshi_fut.result()
    timings["fetch_polymarket"] = round(timings["fetch_polymarket"], 2)
    timings["fetch_kalshi"] = round(timings["fetch_kalshi"], 2)
    rep.log(f"  ⏱ Polymarket {timings['fetch_polymarket']:.1f}s, "
            f"Kalshi {timings['fetch_kalshi']:.1f}s (in parallel)")
    stage_done("fetch", t)
    rep.check_cancelled()

    # 2 ── text matching ──────────────────────────────────────────────────────
    rep.step(2, "TF-IDF pre-filter")
    t = clock()
    candidates = find_matches(res.poly, res.kalshi, cfg, rep)
    res.matches = [c for c in candidates if c[3] >= cfg.match_display_threshold]
    rep.log(f"  {len(res.matches):,} at ≥{cfg.match_display_threshold:.0%} similarity")
    stage_done("match", t)
    rep.check_cancelled()

    # Verdicts already on file cost nothing, so look them up before pricing.
    cache = VerifyCache.load(
        (cfg.verify_cache_path or default_cache_path()) if cfg.use_verify_cache else "",
        cfg.verify_cache_ttl_days, cfg.groq_model, VERIFY_PROMPT_VERSION,
        seed=cfg.verify_cache_seed if cfg.use_verify_cache else None)
    verdicts = {}
    for p, k, _shared, _score in res.matches:
        entry = cache.get(p, k)
        if entry:
            verdicts[pair_id(p, k)] = {"match": entry["match"],
                                       "confidence": entry["confidence"],
                                       "shared": set(entry.get("shared") or [])}
    cache_hits = len(verdicts)

    def is_verified(p, k):
        v = verdicts.get(pair_id(p, k))
        return bool(v and v["match"] and v["confidence"] >= cfg.groq_confidence_cutoff)

    # 3 ── Polymarket books and preliminary edge ─────────────────────────────
    rep.step(3, "Order books and preliminary edge")
    t = clock()
    res.matched_poly = list({id(p): p for p, _k, _s, _c in res.matches}.values())
    res.matched_kalshi = list({id(k): k for _p, k, _s, _c in res.matches}.values())
    p_ix = {id(m): i for i, m in enumerate(res.matched_poly)}
    k_ix = {id(m): i for i, m in enumerate(res.matched_kalshi)}
    rep.log(f"  {len(res.matches):,} candidate pairs across {len(res.matched_poly)} "
            f"Polymarket + {len(res.matched_kalshi)} Kalshi markets"
            + (f"; {cache_hits:,} verdicts reused from cache" if cache_hits else ""))

    # Polymarket books come in bulk, and pairs already verified get their real
    # Kalshi books in the same round. Unverified pairs are ranked on the quote
    # Kalshi listed moments ago: Kalshi allows ~10 calls a second, so a real
    # book for every candidate would cost more than it tells us at this point.
    known = {k_ix[id(k)] for p, k, _s, _c in res.matches if is_verified(p, k)}
    res.poly_ob, res.kalshi_ob = fetch_all_orderbooks(
        res.matched_poly, res.matched_kalshi, cfg, rep, kalshi_only=known)
    fetch_kalshi_series_fees([res.matched_kalshi[j] for j in sorted(known)], cfg, rep)
    if res.poly_ob or res.kalshi_ob:
        res.prices_updated_at = datetime.now(timezone.utc)
    rep.check_cancelled()

    listed_at = time.time()
    prelim = {}
    for p, k, _s, _c in res.matches:
        ki = k_ix[id(k)]
        kbook = res.kalshi_ob.get(ki) or kalshi_listed_book(k, listed_at)
        quote = best_quote(p, k, res.poly_ob.get(p_ix[id(p)]), kbook, cfg)
        prelim[pair_id(p, k)] = quote["margin"] if quote else None
    edges = [e for e in prelim.values() if e is not None]
    rep.log(f"  Preliminary net edge after fees: {sum(1 for e in edges if e > 0)} pairs "
            f"positive, {sum(1 for e in edges if -cfg.near_margin <= e <= 0)} within "
            f"{cfg.near_margin * 100:.1f}¢ of break-even, {len(res.matches) - len(edges)} "
            "unpriced (none count as arbitrage until verified)")
    stage_done("orderbooks", t)
    rep.check_cancelled()

    # 4 ── LLM verification, most promising first ─────────────────────────────
    rep.step(4, "LLM verification (edge first)")
    t = clock()

    def tier(m):
        """0 plausible edge · 1 edge too big to be likely · 2 near miss · 3 none · 4 unpriced."""
        e = prelim.get(pair_id(m[0], m[1]))
        if e is None:
            return 4
        if 0 < e <= PLAUSIBLE_EDGE:
            return 0
        if e > PLAUSIBLE_EDGE:
            return 1
        return 2 if e >= -cfg.near_margin else 3

    todo = [m for m in res.matches if pair_id(m[0], m[1]) not in verdicts]
    # Real mispricings are small, so the smallest positive edges go first.
    required = sorted((m for m in todo if tier(m) == 0),
                      key=lambda m: prelim[pair_id(m[0], m[1])])
    optional = sorted((m for m in todo if tier(m) != 0), key=lambda m: (tier(m), -m[3]))
    if todo:
        n = [sum(1 for m in todo if tier(m) == t) for t in range(5)]
        rep.log(f"  Queue: {n[0]} with a plausible edge, {n[1]} with an edge over "
                f"{PLAUSIBLE_EDGE:.0%} (usually a bad match), {n[2]} near break-even, "
                f"{n[3] + n[4]} with none")
    fresh = verify_matches_with_llm(required, optional, cfg, rep)
    by_id = {pair_id(m[0], m[1]): m for m in res.matches}
    for key, verdict in fresh.items():
        p, k = by_id[key][0], by_id[key][1]
        cache.put(p, k, verdict["match"], verdict["confidence"], verdict["shared"])
    verdicts.update(fresh)
    saved = cache.save()
    res.verify_cache = cache.export((m[0], m[1]) for m in res.matches)
    res.llm_used = cfg.llm_enabled

    for p, k, shared, _score in res.matches:
        key = pair_id(p, k)
        if key not in verdicts:
            continue
        if is_verified(p, k):
            res.pair_status[key] = "verified"
            res.verified.append((p, k, verdicts[key]["shared"] or shared,
                                 verdicts[key]["confidence"]))
        else:
            res.pair_status[key] = "rejected"
    res.verified.sort(key=lambda x: x[3], reverse=True)
    pending = len(res.matches) - len(res.pair_status)
    res.llm_stats = {"cache_hits": cache_hits, "new_verdicts": len(fresh),
                     "pending": pending}
    rep.log(f"  → {len(res.verified):,} verified, "
            f"{len(res.pair_status) - len(res.verified):,} rejected, "
            f"{pending:,} still unverified"
            + (" (cache saved)" if saved else ""))
    stage_done("llm", t)
    rep.check_cancelled()

    # 5 ── confirm on fresh full-depth books ──────────────────────────────────
    rep.step(5, "Confirming live arbitrage")
    t = clock()
    # Every verified pair is priced from real books on both venues. Books read
    # before the LLM pass are read again if that pass took a while.
    verified_p = sorted({p_ix[id(p)] for p, _k, _s, _c in res.verified})
    verified_k = sorted({k_ix[id(k)] for _p, k, _s, _c in res.verified})
    missing = [j for j in verified_k if j not in res.kalshi_ob]
    aged = time.time() - min([b["ts"] for b in res.poly_ob.values()]
                             + [b["ts"] for b in res.kalshi_ob.values()]
                             + [time.time()]) > cfg.stale_quote_seconds / 3
    if verified_k and (missing or aged):
        k_list = verified_k if aged else missing
        p_list = verified_p if aged else []
        rep.log(f"  Reading {len(p_list)} Polymarket + {len(k_list)} Kalshi books "
                "for verified pairs")
        new_p, new_k = fetch_all_orderbooks(
            [res.matched_poly[i] for i in p_list],
            [res.matched_kalshi[j] for j in k_list], cfg, rep, step=5)
        # A book that fails to refresh keeps its older quote, which then counts
        # as stale rather than live.
        for pos, book in new_p.items():
            if "yes" in book and "no" in book:
                res.poly_ob[p_list[pos]] = book
        for pos, book in new_k.items():
            res.kalshi_ob[k_list[pos]] = book
        fetch_kalshi_series_fees([res.matched_kalshi[j] for j in verified_k], cfg, rep)
        res.prices_updated_at = datetime.now(timezone.utc)
    res.live_kalshi = [res.matched_kalshi[j] for j in verified_k]
    res.arbs = find_arbitrage(res.verified, res.matched_poly, res.matched_kalshi,
                              res.poly_ob, res.kalshi_ob, cfg, rep)
    stage_done("confirm", t)

    res.finished_at = datetime.now(timezone.utc)
    timings["total"] = round(clock() - t_start, 2)
    rep.log("  ⏱ total {total:.1f}s = fetch {fetch:.1f}s + match {match:.1f}s + "
            "order books {orderbooks:.1f}s + LLM {llm:.1f}s + confirm {confirm:.1f}s"
            .format(**timings))
    rep.log(f"Done in {res.duration_seconds:.1f}s — "
            f"{len(res.matches):,} matches, {len(res.verified):,} verified, "
            f"{len(res.arbs):,} live arbs")
    return res


def refresh_scan(result: ScanResult, rep: Reporter = None) -> ScanResult:
    """Reprice existing matches without mutating the published discovery snapshot.

    No LLM calls: verdicts from the scan stand, only order books are re-read.
    """
    rep = rep or Reporter()
    if not result.matched_poly and not result.matched_kalshi:
        raise RuntimeError("No matched markets to refresh. Run a new full scan.")
    only = None
    if result.live_kalshi is not None:
        live = {id(m) for m in result.live_kalshi}
        only = {j for j, m in enumerate(result.matched_kalshi) if id(m) in live}
    # Keep market dictionaries shared: matching and sheet builders map them by id.
    refreshed = copy(result)
    refreshed.poly_ob, refreshed.kalshi_ob = fetch_all_orderbooks(
        result.matched_poly, result.matched_kalshi, result.config, rep,
        kalshi_only=only)
    if not refreshed.poly_ob and not refreshed.kalshi_ob:
        raise RuntimeError("No live orderbooks were available. Previous results were retained.")
    refreshed.prices_updated_at = datetime.now(timezone.utc)
    refreshed.arbs = find_arbitrage(
        result.verified, result.matched_poly, result.matched_kalshi,
        refreshed.poly_ob, refreshed.kalshi_ob, result.config, rep)
    return refreshed

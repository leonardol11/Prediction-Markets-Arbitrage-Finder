"""Cross-platform prediction-market matcher and arbitrage scanner.

Pipeline
    1. Fetch eligible Polymarket + Kalshi markets (concurrently).
    2. TF-IDF pre-filter to find candidate title pairs.
    3. Optional Groq LLM verification of those candidates.
    4. Fetch best-ask orderbooks for the matched markets only.
    5. Scan for cross-platform arbitrage (YES here + NO there < $1.00).

Every long step reports through a `Reporter`, so a CLI and the web UI can
render the same progress.
"""

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import copy
from datetime import datetime, timezone

import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_sim

from .config import ScanConfig

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


# ───────────────────────────────── Orderbooks ─────────────────────────────────

POLY_CLOB_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; Market-Matcher/1.0)"}


def fetch_poly_orderbook(token_id: str):
    """Lowest ask price on one Polymarket CLOB token, plus size at that price."""
    try:
        r = requests.get("https://clob.polymarket.com/book",
                         params={"token_id": token_id},
                         headers=POLY_CLOB_HEADERS, timeout=8)
        r.raise_for_status()
        asks = r.json().get("asks", [])
        if not asks:
            return None, None
        bp = min(_num(x.get("price")) for x in asks)
        sz = sum(_num(x.get("size")) for x in asks if _num(x.get("price")) == bp)
        return float(bp), float(sz)
    except Exception:
        return None, None


def fetch_kalshi_orderbook(ticker: str):
    """Return (yes_ask, yes_size), (no_ask, no_size) from a Kalshi orderbook.

    Kalshi returns bid and ask ladders separately. We read the ask side
    directly; if a side's ask ladder is missing we approximate it as
    (1 - opposite_bid), which is exact only at zero spread and otherwise
    underestimates the true ask.
    """
    try:
        r = _kalshi_get(f"{KALSHI_BASE}/markets/{ticker}/orderbook",
                        params={"depth": 1}, attempts=3, timeout=5)
        ob = (r.json().get("orderbook_fp") or {})

        def _best_ask(levels):          # lowest ask = first element
            return (float(levels[0][0]), float(levels[0][1])) if levels else (None, None)

        def _best_bid(levels):          # highest bid = last element
            return (float(levels[-1][0]), float(levels[-1][1])) if levels else (None, None)

        ya, ys = _best_ask(ob.get("yes_ask_dollars") or ob.get("yes_dollars_ask", []))
        na, ns = _best_ask(ob.get("no_ask_dollars") or ob.get("no_dollars_ask", []))

        if ya is None:
            nb, nbs = _best_bid(ob.get("no_bid_dollars") or ob.get("no_dollars_bid", [])
                                or ob.get("no_dollars", []))
            ya, ys = ((1.0 - nb), nbs) if nb is not None else (None, None)
        if na is None:
            yb, ybs = _best_bid(ob.get("yes_bid_dollars") or ob.get("yes_dollars_bid", [])
                                or ob.get("yes_dollars", []))
            na, ns = ((1.0 - yb), ybs) if yb is not None else (None, None)

        return (ya, ys), (na, ns)
    except Exception:
        return (None, None), (None, None)


def _clob_yes_no_tokens(m: dict):
    raw = m.get("clobTokenIds", "[]")
    try:
        t = json.loads(raw) if isinstance(raw, str) else (raw or [])
    except (json.JSONDecodeError, TypeError):
        return None
    return (t[0], t[1]) if len(t) >= 2 else None


def fetch_all_orderbooks(poly_markets, kalshi_markets, cfg: ScanConfig, rep: Reporter):
    poly_ob, kalshi_ob = {}, {}
    poly_tasks, total = [], 0
    for i, m in enumerate(poly_markets):
        if pair := _clob_yes_no_tokens(m):
            poly_tasks.append((i, pair[0], pair[1]))
            total += 2
    total += sum(1 for m in kalshi_markets if m.get("ticker"))
    if total == 0:
        return poly_ob, kalshi_ob

    rep.log(f"  Polymarket markets with CLOB tokens: {len(poly_tasks)}/{len(poly_markets)}")
    rep.log(f"  Fetching {total} orderbooks in parallel…")

    done = 0
    with ThreadPoolExecutor(max_workers=cfg.orderbook_workers) as ex:
        futures = {}
        for i, yes_tok, no_tok in poly_tasks:
            futures[ex.submit(fetch_poly_orderbook, yes_tok)] = ("poly_yes", i)
            futures[ex.submit(fetch_poly_orderbook, no_tok)] = ("poly_no", i)
        for j, m in enumerate(kalshi_markets):
            if ticker := m.get("ticker"):
                futures[ex.submit(fetch_kalshi_orderbook, ticker)] = ("kalshi", j)

        for fut in as_completed(futures):
            done += 1
            if done % 25 == 0 or done == total:
                rep.progress(4, f"Orderbooks {done}/{total}", done / total)
            key, idx = futures[fut]
            try:
                res = fut.result()
            except Exception:
                res = None
            if key == "poly_yes" and res and res[0] is not None:
                poly_ob.setdefault(idx, {})["yes"] = res
            elif key == "poly_no" and res and res[0] is not None:
                poly_ob.setdefault(idx, {})["no"] = res
            elif key == "kalshi" and res:
                (y_ap, _), (n_ap, _) = res
                if y_ap is not None or n_ap is not None:
                    kalshi_ob[idx] = res

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

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def _call_groq_verify(pairs_text: str, cfg: ScanConfig, rep: Reporter,
                      max_retries: int = 3) -> list:
    prompt = (
        "You are a prediction-market analyst. A text-similarity algorithm "
        "flagged the candidate pairs below as potential matches between "
        "Polymarket and Kalshi. Verify each one.\n\n"
        f"{pairs_text}\n\n"
        "For each pair, decide whether the two markets resolve on the EXACT "
        "same outcome.\n\n"
        "Rules:\n"
        "- Differently-worded titles are fine IF they resolve identically.\n"
        "- If one title includes a year (e.g. '2026') and the other omits it, "
        "that is still a match as long as both clearly refer to the same event.\n"
        "- REJECT pairs only when they have genuinely CONFLICTING numbers or "
        "thresholds (e.g. 'above 3%' vs 'above 4%', or '$100k' vs '$80k').\n"
        "- REJECT pairs about different entities (e.g. different people, "
        "teams, or countries).\n"
        "- REJECT pairs where the questions ask fundamentally different things, "
        "not just minor wording differences.\n\n"
        'Return ONLY a JSON array (no markdown fences, no explanation). '
        "Include ONLY confirmed matches. Each element:\n"
        '{"pair_index": <int>, "confidence": <float 0-1>, '
        '"shared_keywords": [<strings>]}\n\n'
        "If no pairs are valid matches, return: []"
    )

    for attempt in range(max_retries):
        last_try = attempt == max_retries - 1
        try:
            r = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {cfg.groq_api_key}",
                         "Content-Type": "application/json"},
                json={"model": cfg.groq_model, "temperature": 0,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=120,
            )
            if r.status_code in (401, 403):
                raise GroqAuthError(r.status_code)
            if r.status_code == 429:
                raise RuntimeError("rate_limit_exceeded")
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"].strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw)
            result = json.loads(raw)
            if not isinstance(result, list):
                raise ValueError(f"expected a JSON array, got {type(result).__name__}")
            return result
        except GroqAuthError:
            raise
        except (json.JSONDecodeError, ValueError) as exc:
            rep.log(f"    attempt {attempt + 1}/{max_retries}: bad JSON – {exc}")
            if not last_try:
                time.sleep(10)
        except Exception as exc:
            rate_limited = "429" in str(exc) or "rate_limit" in str(exc)
            wait = 60 if rate_limited else 10
            detail = "rate limited" if rate_limited else f"API error – {exc}"
            rep.log(f"    attempt {attempt + 1}/{max_retries}: {detail}"
                    + ("" if last_try else f" – waiting {wait}s…"))
            if not last_try:
                time.sleep(wait)

    rep.log("  ⚠ Groq gave up — check the key / quota at https://console.groq.com/keys")
    return []


def verify_matches_with_llm(matches: list, cfg: ScanConfig, rep: Reporter) -> list:
    if not matches:
        rep.log("  No candidates to verify — skipping the LLM pass.")
        return []
    if not cfg.llm_enabled:
        rep.log("  LLM verification disabled (no Groq API key) — skipping.")
        return []

    bs = max(1, cfg.groq_batch_size)
    n_batches = (len(matches) + bs - 1) // bs
    rep.log(f"  {len(matches):,} candidates → {n_batches} Groq call(s) "
            f"(batch size {bs}, model {cfg.groq_model})")

    verified = []
    for start in range(0, len(matches), bs):
        rep.check_cancelled()
        batch = matches[start:start + bs]
        batch_num = start // bs + 1
        rep.progress(3, f"LLM batch {batch_num}/{n_batches}", batch_num / n_batches)

        pairs_text = "\n".join(
            f"Pair {i}:\n"
            f"  Polymarket: {poly_title(p)}\n"
            f"  Kalshi:     {kalshi_title(k)}"
            for i, (p, k, _shared, _score) in enumerate(batch)
        )
        rep.log(f"  Sending batch {batch_num}/{n_batches} ({len(batch)} pairs)…")
        try:
            results = _call_groq_verify(pairs_text, cfg, rep)
        except GroqAuthError as exc:
            rep.log(f"  ⚠ Groq rejected the API key (HTTP {exc}) — skipping LLM "
                    "verification. Get a fresh key at https://console.groq.com/keys")
            return []

        confirmed = 0
        for r in results:
            idx = r.get("pair_index")
            if not isinstance(idx, int) or not (0 <= idx < len(batch)):
                continue
            p, k, old_shared, old_score = batch[idx]
            conf = _num(r.get("confidence"), old_score)
            if conf < cfg.groq_confidence_cutoff:
                continue
            shared = set(r.get("shared_keywords") or old_shared)
            verified.append((p, k, shared, conf))
            confirmed += 1
        rep.log(f"    confirmed {confirmed}/{len(batch)} in this batch")

        if start + len(batch) < len(matches):
            time.sleep(15)   # stay inside Groq's free-tier rate limit

    verified.sort(key=lambda x: x[3], reverse=True)
    rep.log(f"  → {len(verified):,} verified of {len(matches):,} candidates")
    return verified


# ──────────────────────────────── Arbitrage scan ──────────────────────────────

def _executable_ask(price, size):
    """An ask is only tradeable if both a price and a resting size exist."""
    if price is None or price <= 0 or size is None or _num(size) <= 0:
        return None, None
    return float(price), float(size)


def _poly_best_ask_from_ob(side):
    return _executable_ask(side[0], side[1]) if side else (None, None)


def _kalshi_best_asks_from_ob(kob):
    if not kob:
        return None, None, None, None
    (ya, yq), (na, nq) = kob
    return (*_executable_ask(ya, yq), *_executable_ask(na, nq))


def find_arbitrage(matches, poly_markets, kalshi_markets,
                   poly_ob, kalshi_ob, cfg: ScanConfig, rep: Reporter) -> list:
    """Cross-platform arb from CLOB / Kalshi best asks only (never listed prices).

    A pair is an arb when YES on one venue plus NO on the other costs < $1.00,
    since exactly one of the two legs pays out $1.00 at resolution.
    """
    p_ix = {id(m): i for i, m in enumerate(poly_markets)}
    k_ix = {id(m): i for i, m in enumerate(kalshi_markets)}
    arbs = []

    for poly, kalshi, shared, score in matches:
        pi, ki = p_ix.get(id(poly)), k_ix.get(id(kalshi))
        if pi is None or ki is None:
            continue
        pob = poly_ob.get(pi, {})
        yes_ob, no_ob = pob.get("yes"), pob.get("no")
        py, py_sz = _poly_best_ask_from_ob(yes_ob)
        pn, pn_sz = _poly_best_ask_from_ob(no_ob)
        ky, ky_sz, kn, kn_sz = _kalshi_best_asks_from_ob(kalshi_ob.get(ki))

        combos = [
            ("Polymarket", "YES", py, py_sz, "Kalshi", "NO", kn, kn_sz),
            ("Polymarket", "NO", pn, pn_sz, "Kalshi", "YES", ky, ky_sz),
        ]
        for a_plat, a_side, a_price, a_sz, b_plat, b_side, b_price, b_sz in combos:
            if a_price is None or b_price is None:
                continue
            # The Polymarket leg must come from a real CLOB book entry.
            if a_side == "YES" and not (yes_ob and yes_ob[0] is not None):
                continue
            if a_side == "NO" and not (no_ob and no_ob[0] is not None):
                continue

            ask_sum = a_price + b_price
            margin = 1.0 - ask_sum
            if margin <= 0 or margin < cfg.min_margin:
                continue

            max_contracts = min(a_sz, b_sz) if (a_sz and b_sz) else None
            arbs.append({
                "poly": poly, "kalshi": kalshi,
                "shared": shared, "match_score": score,
                "a_plat": a_plat, "a_side": a_side,
                "a_price": a_price, "a_ask_size": a_sz,
                "b_plat": b_plat, "b_side": b_side,
                "b_price": b_price, "b_ask_size": b_sz,
                "prob_sum": ask_sum,
                "margin": margin,
                "max_contracts": max_contracts,
                "max_profit": (margin * max_contracts) if max_contracts else None,
                "poly_ob_yes": yes_ob, "poly_ob_no": no_ob,
                "kalshi_ob": kalshi_ob.get(ki),
            })

    arbs.sort(key=lambda x: x["margin"], reverse=True)
    rep.log(f"  {len(arbs)} arbitrage opportunit{'y' if len(arbs) == 1 else 'ies'} "
            f"found on executable best asks")
    return arbs


# ──────────────────────────────── Full pipeline ───────────────────────────────

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
        self.arb_source_label = "Matches"
        self.llm_used = False

    @property
    def duration_seconds(self) -> float:
        end = self.finished_at or datetime.now(timezone.utc)
        return (end - self.started_at).total_seconds()


def run_scan(cfg: ScanConfig, rep: Reporter = None) -> ScanResult:
    rep = rep or Reporter()
    res = ScanResult(cfg)

    rep.step(1, "Fetching Polymarket + Kalshi markets")
    with ThreadPoolExecutor(max_workers=2) as ex:
        poly_fut = ex.submit(fetch_poly_markets, cfg, rep)
        kalshi_fut = ex.submit(fetch_kalshi_markets, cfg, rep)
        res.poly = poly_fut.result()
        res.kalshi = kalshi_fut.result()
    rep.check_cancelled()

    rep.step(2, "TF-IDF pre-filter")
    candidates = find_matches(res.poly, res.kalshi, cfg, rep)
    res.matches = [c for c in candidates if c[3] >= cfg.match_display_threshold]
    rep.log(f"  {len(res.matches):,} at ≥{cfg.match_display_threshold:.0%} similarity")
    rep.check_cancelled()

    rep.step(3, "LLM verification")
    res.verified = verify_matches_with_llm(res.matches, cfg, rep)
    res.llm_used = cfg.llm_enabled

    if res.verified:
        arb_source, res.arb_source_label = res.verified, "LLM Verified"
    else:
        arb_source, res.arb_source_label = res.matches, "Matches"
        if cfg.llm_enabled:
            rep.log("  ⚠ LLM verified none — falling back to TF-IDF matches for arbitrage")
    rep.check_cancelled()

    rep.step(4, "Fetching orderbooks for matched markets")
    res.matched_poly = list({id(p): p for p, _k, _s, _c in arb_source}.values())
    res.matched_kalshi = list({id(k): k for _p, k, _s, _c in arb_source}.values())
    rep.log(f"  {len(res.matched_poly)} Polymarket + {len(res.matched_kalshi)} "
            f"Kalshi markets need books")
    res.poly_ob, res.kalshi_ob = fetch_all_orderbooks(
        res.matched_poly, res.matched_kalshi, cfg, rep)
    if res.poly_ob or res.kalshi_ob:
        res.prices_updated_at = datetime.now(timezone.utc)
    rep.check_cancelled()

    rep.step(5, "Arbitrage scan")
    res.arbs = find_arbitrage(arb_source, res.matched_poly, res.matched_kalshi,
                              res.poly_ob, res.kalshi_ob, cfg, rep)

    res.finished_at = datetime.now(timezone.utc)
    rep.log(f"Done in {res.duration_seconds:.1f}s — "
            f"{len(res.matches):,} matches, {len(res.verified):,} verified, "
            f"{len(res.arbs):,} arbs")
    return res


def refresh_scan(result: ScanResult, rep: Reporter = None) -> ScanResult:
    """Reprice existing matches without mutating the published discovery snapshot."""
    rep = rep or Reporter()
    if not result.matched_poly and not result.matched_kalshi:
        raise RuntimeError("No matched markets to refresh. Run a new full scan.")
    # Keep market dictionaries shared: matching and sheet builders map them by id.
    refreshed = copy(result)
    refreshed.poly_ob, refreshed.kalshi_ob = fetch_all_orderbooks(
        result.matched_poly, result.matched_kalshi, result.config, rep)
    if not refreshed.poly_ob and not refreshed.kalshi_ob:
        raise RuntimeError("No live orderbooks were available. Previous results were retained.")
    refreshed.prices_updated_at = datetime.now(timezone.utc)
    source = result.verified or result.matches
    refreshed.arbs = find_arbitrage(
        source, result.matched_poly, result.matched_kalshi,
        refreshed.poly_ob, refreshed.kalshi_ob, result.config, rep)
    return refreshed

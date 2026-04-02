import os
import re
import json
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_sim
from groq import Groq

CUTOFF_DATE = datetime(2026, 7, 1, tzinfo=timezone.utc)
MIN_VOLUME  = 25_000

GROQ_API_KEY = "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"
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_VERIFY_BATCH = 50

_groq_client = Groq(api_key=GROQ_API_KEY)


def _parse_iso(s: str):
    try:
        return datetime.fromisoformat((s or "").replace("Z", "+00:00"))
    except ValueError:
        return None


POLY_BASE = "https://gamma-api.polymarket.com"


def fetch_poly_markets() -> list:
    print("Fetching Polymarket markets...")
    markets, offset = [], 0
    while True:
        r = requests.get(f"{POLY_BASE}/markets", params={
            "active": "true", "closed": "false", "limit": 500,
            "offset": offset, "order": "volume", "ascending": "false",
        }, timeout=15)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        markets.extend(batch)
        if len(batch) < 500:
            break
        offset += 500
    print(f"  {len(markets)} Polymarket markets fetched")

    eligible = [
        m for m in markets
        if float(m.get("volume") or 0) >= MIN_VOLUME
        and (d := _parse_iso(m.get("endDate") or m.get("end_date_iso")))
        and d < CUTOFF_DATE
    ]
    eligible.sort(key=lambda m: float(m.get("volume") or 0), reverse=True)
    print(f"  → {len(eligible)} eligible (vol ≥ ${MIN_VOLUME:,}, expires before Jul 1 2026)")
    return eligible


KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
_session = requests.Session()


def _kalshi_get(url, params=None, attempts=5, timeout=15, session=None):
    sess = session or _session
    for attempt in range(attempts):
        try:
            r = sess.get(url, params=params, timeout=timeout)
            if r.status_code == 429:
                time.sleep(1.5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r
        except requests.Timeout:
            if attempt == attempts - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
        except requests.RequestException:
            if attempt == attempts - 1:
                raise
            time.sleep(0.5)
    raise RuntimeError(f"Gave up after {attempts} attempts: {url}")


def fetch_kalshi_markets() -> list:
    print("Fetching Kalshi markets...")
    seen, eligible = set(), []
    fetched_count, cursor = 0, None
    rejected_status = 0
    rejected_volume = 0
    rejected_missing_date = 0
    rejected_cutoff = 0
    session = requests.Session()
    try:
        while True:
            params = {
                "status": "open",
                "limit": 1000,
                "order": "volume",
                "ascending": "false",
            }
            if cursor:
                params["cursor"] = cursor
            data = _kalshi_get(f"{KALSHI_BASE}/markets", params=params, session=session).json()
            batch = data.get("markets", [])
            if not batch:
                break

            for m in batch:
                if (tid := m.get("ticker")) in seen:
                    continue
                seen.add(tid)
                fetched_count += 1

                vol = float(m.get("volume_fp") or 0)

                if m.get("status") not in {"open", "active"}:
                    rejected_status += 1
                    continue
                if vol < MIN_VOLUME:
                    rejected_volume += 1
                    continue

                d = _parse_iso(
                    m.get("close_time")
                    or m.get("expiration_time")
                    or m.get("expected_expiration_time")
                )
                if not d:
                    rejected_missing_date += 1
                    continue
                if d >= CUTOFF_DATE:
                    rejected_cutoff += 1
                    continue

                eligible.append(m)

            cursor = data.get("cursor")
            if not cursor:
                break
    finally:
        session.close()

    print(f"  {fetched_count} raw Kalshi markets fetched")
    print(
        "  Kalshi rejected:"
        f" status={rejected_status},"
        f" volume<{MIN_VOLUME:,}={rejected_volume},"
        f" missing_date={rejected_missing_date},"
        f" on/after_cutoff={rejected_cutoff}"
    )
    eligible.sort(key=lambda m: float(m.get("volume_fp") or 0), reverse=True)
    print(f"  → {len(eligible)} eligible (vol ≥ ${MIN_VOLUME:,}, expires before Jul 1 2026)")
    return eligible


POLY_CLOB_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; Market-Matcher/1.0)"}


def fetch_poly_orderbook(token_id: str):
    try:
        r = requests.get(
            "https://clob.polymarket.com/book",
            params={"token_id": token_id},
            headers=POLY_CLOB_HEADERS,
            timeout=8,
        )
        r.raise_for_status()
        asks = r.json().get("asks", [])
        if not asks:
            return None, None
        bp = min(float(x.get("price") or 0) for x in asks)
        sz = sum(float(x.get("size") or 0) for x in asks if float(x.get("price") or 0) == bp)
        return float(bp), float(sz)
    except Exception:
        return None, None


def fetch_kalshi_orderbook(ticker: str):
    """Return (yes_ask, yes_size), (no_ask, no_size) from Kalshi orderbook.

    Kalshi returns bid and ask levels separately under yes_dollars / no_dollars.
    We read the ask side directly (lowest ask = first element of the ask ladder).
    If the ask ladder is absent for a side, we fall back to deriving the ask as
    (1.0 - opposite_bid), which is only exact when spread = 0 and will
    underestimate the real ask when a spread exists — treat those as approximate.
    """
    try:
        r = _kalshi_get(f"{KALSHI_BASE}/markets/{ticker}/orderbook", params={"depth": 1}, attempts=3, timeout=5)
        ob = (r.json().get("orderbook_fp") or {})

        def _best_ask(ask_levels):
            """Top of ask ladder: lowest ask price (first element)."""
            if not ask_levels:
                return None, None
            p, s = ask_levels[0]
            return float(p), float(s)

        def _best_bid(bid_levels):
            """Top of bid ladder: highest bid price (last element)."""
            if not bid_levels:
                return None, None
            p, s = bid_levels[-1]
            return float(p), float(s)

        yes_ask_levels = ob.get("yes_ask_dollars") or ob.get("yes_dollars_ask", [])
        no_ask_levels  = ob.get("no_ask_dollars")  or ob.get("no_dollars_ask",  [])
        yes_bid_levels = ob.get("yes_bid_dollars") or ob.get("yes_dollars_bid", []) or ob.get("yes_dollars", [])
        no_bid_levels  = ob.get("no_bid_dollars")  or ob.get("no_dollars_bid",  []) or ob.get("no_dollars",  [])

        ya, ys = _best_ask(yes_ask_levels)
        na, ns = _best_ask(no_ask_levels)

        if ya is None:
            # Fall back: approximate YES ask from NO bid (underestimates when spread > 0)
            nb, nbs = _best_bid(no_bid_levels)
            ya, ys = ((1.0 - nb), nbs) if nb is not None else (None, None)

        if na is None:
            # Fall back: approximate NO ask from YES bid (underestimates when spread > 0)
            yb, ybs = _best_bid(yes_bid_levels)
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


def fetch_all_orderbooks(poly_markets: list, kalshi_markets: list) -> tuple:
    poly_ob, kalshi_ob = {}, {}
    poly_tasks = []
    total = 0
    for i, m in enumerate(poly_markets):
        pair = _clob_yes_no_tokens(m)
        if pair:
            poly_tasks.append((i, pair[0], pair[1]))
            total += 2
    for m in kalshi_markets:
        if m.get("ticker"):
            total += 1

    if total == 0:
        return poly_ob, kalshi_ob

    valid_tokens = sum(1 for m in poly_markets if _clob_yes_no_tokens(m))
    print(f"  Poly markets with valid clobTokenIds: {valid_tokens}/{len(poly_markets)}")

    print(f"  Fetching {total} orderbooks in parallel…")
    done = 0
    with ThreadPoolExecutor(max_workers=80) as ex:
        futures = {}
        for i, yes_tok, no_tok in poly_tasks:
            futures[ex.submit(fetch_poly_orderbook, yes_tok)] = ("poly_yes", i)
            futures[ex.submit(fetch_poly_orderbook, no_tok)] = ("poly_no", i)
        for j, m in enumerate(kalshi_markets):
            ticker = m.get("ticker")
            if ticker:
                futures[ex.submit(fetch_kalshi_orderbook, ticker)] = ("kalshi", j)

        for fut in as_completed(futures):
            done += 1
            if done % 50 == 0 or done == total:
                print(f"  Orderbooks: {done}/{total}", end="\r")
            key, idx = futures[fut]
            try:
                res = fut.result()
            except Exception:
                res = None
            if key == "poly_yes":
                if res and res[0] is not None:
                    poly_ob.setdefault(idx, {})["yes"] = res
            elif key == "poly_no":
                if res and res[0] is not None:
                    poly_ob.setdefault(idx, {})["no"] = res
            elif key == "kalshi":
                if res:
                    (y_ap, _), (n_ap, _) = res
                    if y_ap is not None or n_ap is not None:
                        kalshi_ob[idx] = res
    print()
    return poly_ob, kalshi_ob


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
    "us", "be", "st", "nd", "rd", "th", "say", "said", "says",
}

TFIDF_THRESHOLD         = 0.45
COMBINED_THRESHOLD      = 0.45
MATCH_DISPLAY_THRESHOLD = 0.45
GROQ_CONFIDENCE_CUTOFF = 0.65

# Longer phrases first (avoid partial matches on shorter rules).
_SYNONYM_SUBS = [
    (re.compile(r'\brate[- ]cut\b',                  re.I), 'ratecut'),
    (re.compile(r'\brate[- ]decrease\b',             re.I), 'ratecut'),
    (re.compile(r'\brate[- ]lower\b',                re.I), 'ratecut'),
    (re.compile(r'\brate[- ]hike\b',                 re.I), 'ratehike'),
    (re.compile(r'\brate[- ]increase\b',             re.I), 'ratehike'),
    (re.compile(r'\brate[- ]raise\b',                re.I), 'ratehike'),
    (re.compile(r'\bno[- ]rate[- ]change\b',         re.I), 'ratehold'),
    (re.compile(r'\brates?[- ]unchanged\b',          re.I), 'ratehold'),
    (re.compile(r'\bno[- ]change\b',                 re.I), 'ratehold'),
    (re.compile(r'\b(\d+)\s*bps\b',                  re.I), r'\1basispoints'),
    (re.compile(r'\b(\d+)\s*basis\s*points?\b',      re.I), r'\1basispoints'),
    (re.compile(r'\bconsumer[- ]price[- ]index\b',   re.I), 'cpi'),
    (re.compile(r'\bnonfarm[- ]payrolls?\b',          re.I), 'payroll'),
    (re.compile(r'\bjobs?[- ]report\b',              re.I), 'payroll'),
    (re.compile(r'\bgross[- ]domestic[- ]product\b', re.I), 'gdp'),
    (re.compile(r'\bfederal[- ]reserve\b',           re.I), 'fed'),
    (re.compile(r'\bfederal[- ]funds[- ]rate\b',     re.I), 'federalfundsrate'),
    (re.compile(r'\bpersonal[- ]consumption\b',      re.I), 'pce'),
    (re.compile(r'\bexecutive[- ]order\b',           re.I), 'executiveorder'),
    (re.compile(r'\bsupreme[- ]court\b',             re.I), 'supremecourt'),
    (re.compile(r'\battorney[- ]general\b',          re.I), 'attorneygeneral'),
    (re.compile(r'\bprime[- ]minister\b',            re.I), 'primeminister'),
    (re.compile(r'\btrade[- ]war\b',                 re.I), 'tradewar'),
    (re.compile(r'\btrade[- ]deal\b',                re.I), 'tradedeal'),
    (re.compile(r'\btrade[- ]deficit\b',             re.I), 'tradedeficit'),
    (re.compile(r'\belections?\b',      re.I), 'election'),
    (re.compile(r'\bimpeach(?:ment|ed)?\b', re.I), 'impeach'),
    (re.compile(r'\bresign(?:ation|s)?\b',  re.I), 'resign'),
    (re.compile(r'\bsanctions?\b',      re.I), 'sanction'),
    (re.compile(r'\btariffs?\b',        re.I), 'tariff'),
    (re.compile(r'\bapprov(?:al|ed)\b', re.I), 'approve'),
    (re.compile(r'\bnomin(?:ation|ated)\b', re.I), 'nominate'),
    (re.compile(r'\bconfirm(?:ation|ed)\b', re.I), 'confirm'),
    (re.compile(r'\bvetoed?\b',         re.I), 'veto'),
    (re.compile(r'\bpayrolls\b',        re.I), 'payroll'),
    (re.compile(r'\brecessions?\b',     re.I), 'recession'),
    (re.compile(r'\bdeficits?\b',       re.I), 'deficit'),
]


_RATE_CONTEXT = re.compile(
    r'\b(rate|rates|fed|fomc|bps|basis\s*points?|federal\s*funds|'
    r'interest|monetary|policy|hike|cut|central\s*bank)\b', re.I
)


def _apply_rate_synonyms(text: str) -> str:
    """Only substitute rate-action verbs if a rate-context word is present."""
    if not _RATE_CONTEXT.search(text):
        return text
    text = re.sub(r'\bcuts?\b',         'ratecut',  text, flags=re.I)
    text = re.sub(r'\bdecrease[sd]?\b', 'ratecut',  text, flags=re.I)
    text = re.sub(r'\blowere?d?\b',     'ratecut',  text, flags=re.I)
    text = re.sub(r'\bhikes?\b',        'ratehike', text, flags=re.I)
    text = re.sub(r'\bincrease[sd]?\b', 'ratehike', text, flags=re.I)
    text = re.sub(r'\braise[sd]?\b',    'ratehike', text, flags=re.I)
    text = re.sub(r'\bunchanged\b',     'ratehold', text, flags=re.I)
    text = re.sub(r'\bpaused?\b',       'ratehold', text, flags=re.I)
    return text


def normalize_for_matching(text: str) -> str:
    for pat, rep in _SYNONYM_SUBS:
        text = pat.sub(rep, text)
    text = _apply_rate_synonyms(text)
    return text


def tokenize(text: str) -> list:
    words = re.findall(r"[a-z]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


_MONTHS = "january february march april may june july august september october november december".split()
_MONTH_MAP = {**{m: i+1 for i, m in enumerate(_MONTHS)},
              **{m[:3]: i+1 for i, m in enumerate(_MONTHS)}}


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
    "xi jinping": "xi",
    "zelensky": "zelenskyy",
}


def extract_entities(text: str) -> set:
    entities = set()
    for pattern in _ENTITY_PATTERNS.values():
        for m in pattern.finditer(text):
            raw = m.group(0).lower().strip()
            entities.add(_ENTITY_NORMALIZE.get(raw, raw))
    return entities


_ABOVE_RE = re.compile(
    r'\b(?:above|over|more\s+than|greater\s+than|exceed(?:s|ing)?|'
    r'at\s+least|higher\s+than)\b', re.I)
_BELOW_RE = re.compile(
    r'\b(?:below|under|less\s+than|fewer\s+than|at\s+most|lower\s+than)\b', re.I)


def _extract_direction(text: str):
    a, b = bool(_ABOVE_RE.search(text)), bool(_BELOW_RE.search(text))
    if a != b:
        return "above" if a else "below"
    return None


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
    months = lambda t: {_MONTH_MAP[w] for w in re.findall(r'[a-z]+', t.lower()) if w in _MONTH_MAP}
    return not _sets_clash(months(a), months(b))


def find_matches(poly_markets, kalshi_markets) -> list:
    poly_texts   = [m.get("question") or m.get("title") or "" for m in poly_markets]
    kalshi_texts = [_kalshi_title(m) for m in kalshi_markets]

    print(f"  Building TF-IDF matrix ({len(poly_texts)} × {len(kalshi_texts)})...")
    all_norm   = [normalize_for_matching(t) for t in poly_texts + kalshi_texts]
    vec        = TfidfVectorizer(ngram_range=(1, 2), stop_words=list(STOPWORDS), sublinear_tf=True)
    tfidf      = vec.fit_transform(all_norm)
    sim_matrix = sk_cosine_sim(tfidf[:len(poly_texts)], tfidf[len(poly_texts):])

    candidates = np.argwhere(sim_matrix >= TFIDF_THRESHOLD)
    print(f"  {len(candidates)} pairs above TF-IDF threshold ({TFIDF_THRESHOLD})")

    print(f"  Filtering (numbers, months, entities, dates, Jaccard)...")
    poly_tok   = [set(tokenize(t)) for t in poly_texts]
    kalshi_tok = [set(tokenize(t)) for t in kalshi_texts]
    poly_ent   = [extract_entities(t) for t in poly_texts]
    kalshi_ent = [extract_entities(t) for t in kalshi_texts]

    poly_dates = [
        _parse_iso(m.get("endDate") or m.get("end_date_iso"))
        for m in poly_markets
    ]
    kalshi_dates = [
        _parse_iso(m.get("close_time") or m.get("expiration_time")
                   or m.get("expected_expiration_time"))
        for m in kalshi_markets
    ]

    matches = []
    rej_num, rej_mon, rej_ent, rej_date, rej_combined = 0, 0, 0, 0, 0

    for i, j in candidates:
        pt, kt = poly_texts[i], kalshi_texts[j]

        if not numbers_compatible(pt, kt):
            rej_num += 1
            continue
        if not months_compatible(pt, kt):
            rej_mon += 1
            continue
        if _sets_clash(poly_ent[i], kalshi_ent[j]):
            rej_ent += 1
            continue

        pd, kd = poly_dates[i], kalshi_dates[j]
        if pd and kd and abs((pd - kd).days) > 45:
            rej_date += 1
            continue

        p_set, k_set = poly_tok[i], kalshi_tok[j]
        union = p_set | k_set
        jaccard = len(p_set & k_set) / len(union) if union else 0.0
        cosine  = float(sim_matrix[i, j])
        combined = 0.7 * cosine + 0.3 * jaccard

        if combined < COMBINED_THRESHOLD:
            rej_combined += 1
            continue

        matches.append((poly_markets[i], kalshi_markets[j],
                        p_set & k_set, combined))

    matches.sort(key=lambda x: x[3], reverse=True)
    print(f"  Rejected: {rej_num} numbers, {rej_mon} months, "
          f"{rej_ent} entities, {rej_date} dates, "
          f"{rej_combined} combined score")
    print(f"  → {len(matches)} candidates passed pre-filter")
    return matches


def _call_groq_verify(pairs_text: str, max_retries: int = 3) -> list:
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
        try:
            response = _groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            raw = response.choices[0].message.content.strip()

            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw)

            result = json.loads(raw)
            if not isinstance(result, list):
                raise ValueError(f"Expected JSON array, got {type(result).__name__}")
            return result

        except (json.JSONDecodeError, ValueError) as exc:
            print(f"    Attempt {attempt + 1}/{max_retries}: bad JSON – {exc}")
            if attempt < max_retries - 1:
                time.sleep(10)
        except Exception as exc:
            is_rate_limit = "429" in str(exc) or "rate_limit_exceeded" in str(exc)
            wait = 60 if is_rate_limit else 10
            print(f"    Attempt {attempt + 1}/{max_retries}: "
                  f"{'rate limited' if is_rate_limit else 'API error'} – "
                  f"waiting {wait}s...")
            if attempt < max_retries - 1:
                time.sleep(wait)

    print("  ⚠ Groq API quota exhausted — create a new key at https://console.groq.com/keys")
    return []


def verify_matches_with_llm(matches: list) -> list:
    if not matches:
        print("  No candidates to verify — skipping Groq.")
        return []

    n_batches = (len(matches) + GROQ_VERIFY_BATCH - 1) // GROQ_VERIFY_BATCH
    print(f"  {len(matches)} candidates → {n_batches} API call(s) "
          f"(batch size {GROQ_VERIFY_BATCH})")

    verified = []
    bs = GROQ_VERIFY_BATCH

    for start in range(0, len(matches), bs):
        batch = matches[start:start + bs]
        batch_num = start // bs + 1

        pairs_text = "\n".join(
            f"Pair {i}:\n"
            f"  Polymarket: {p.get('question') or p.get('title') or ''}\n"
            f"  Kalshi:     {_kalshi_title(k)}"
            for i, (p, k, _shared, _score) in enumerate(batch)
        )

        print(f"  Sending batch {batch_num}/{n_batches} to Groq "
              f"({len(batch)} pairs)...")
        results = _call_groq_verify(pairs_text)
        confirmed_this_batch = 0

        for r in results:
            idx = r.get("pair_index")
            if not isinstance(idx, int) or not (0 <= idx < len(batch)):
                continue
            p, k, old_shared, old_score = batch[idx]
            conf = float(r.get("confidence", old_score))
            if conf < GROQ_CONFIDENCE_CUTOFF:
                continue
            shared = set(r.get("shared_keywords", old_shared))
            verified.append((p, k, shared, conf))
            confirmed_this_batch += 1

        print(f"    Groq confirmed {confirmed_this_batch}/{len(batch)} "
              f"pairs in this batch")

        if start + len(batch) < len(matches):
            time.sleep(15)

    verified.sort(key=lambda x: x[3], reverse=True)
    print(f"  → {len(verified)} verified / {len(matches)} candidates")
    return verified


def _executable_ask(price, size):
    if price is None or price <= 0 or size is None or float(size) <= 0:
        return None, None
    return float(price), float(size)


def _poly_best_ask_from_ob(side):
    if not side:
        return None, None
    return _executable_ask(side[0], side[1])


def _kalshi_best_asks_from_ob(kob):
    if not kob:
        return None, None, None, None
    (ya, yq), (na, nq) = kob
    y = _executable_ask(ya, yq)
    n = _executable_ask(na, nq)
    return (*y, *n)


def find_arbitrage(matches, poly_markets, kalshi_markets, poly_ob, kalshi_ob) -> list:
    """Uses Polymarket CLOB + Kalshi orderbook best asks only (not Gamma / REST listed)."""
    p_ix = {id(m): i for i, m in enumerate(poly_markets)}
    k_ix = {id(m): i for i, m in enumerate(kalshi_markets)}
    arbs = []
    for poly, kalshi, _, _ in matches:
        pi, ki = p_ix.get(id(poly)), k_ix.get(id(kalshi))
        if pi is None or ki is None:
            continue
        pob = poly_ob.get(pi, {})
        yes_ob, no_ob = pob.get("yes"), pob.get("no")
        py, py_sz = _poly_best_ask_from_ob(yes_ob)
        pn, pn_sz = _poly_best_ask_from_ob(no_ob)

        kob = kalshi_ob.get(ki)
        ky, ky_sz, kn, kn_sz = _kalshi_best_asks_from_ob(kob)

        combos = [
            ("Polymarket", "YES", py, py_sz, "Kalshi", "NO", kn, kn_sz),
            ("Polymarket", "NO", pn, pn_sz, "Kalshi", "YES", ky, ky_sz),
        ]
        for a_plat, a_side, a_price, a_sz, b_plat, b_side, b_price, b_sz in combos:
            if a_price is None or b_price is None:
                continue
            # Require that the Polymarket side came from a real CLOB orderbook entry
            poly_yes_real = yes_ob is not None and yes_ob[0] is not None
            poly_no_real  = no_ob  is not None and no_ob[0]  is not None
            if a_plat == "Polymarket" and a_side == "YES" and not poly_yes_real:
                continue
            if a_plat == "Polymarket" and a_side == "NO"  and not poly_no_real:
                continue
            if b_plat == "Polymarket" and b_side == "YES" and not poly_yes_real:
                continue
            if b_plat == "Polymarket" and b_side == "NO"  and not poly_no_real:
                continue
            ask_sum = a_price + b_price
            if ask_sum >= 1.0:
                continue
            max_contracts = None
            if a_sz is not None and b_sz is not None:
                max_contracts = min(a_sz, b_sz)
            arbs.append({
                "poly":    poly,
                "kalshi":  kalshi,
                "a_plat":  a_plat, "a_side":  a_side, "a_price": a_price, "a_ask_size": a_sz,
                "b_plat":  b_plat, "b_side":  b_side, "b_price": b_price, "b_ask_size": b_sz,
                "prob_sum": ask_sum,
                "margin":  1.0 - ask_sum,
                "max_contracts": max_contracts,
                "poly_ob_yes": yes_ob,
                "poly_ob_no":  no_ob,
                "kalshi_ob":   kob,
            })

    arbs.sort(key=lambda x: x["margin"], reverse=True)
    print(f"  {len(arbs)} arb(s) (best asks)")
    return arbs


def _kalshi_title(market: dict) -> str:
    title = market.get("title") or market.get("question") or ""
    sub   = (market.get("yes_sub_title") or market.get("subtitle") or "").strip()
    return f"{title}  [{sub}]" if sub and sub.lower() not in title.lower() else title


def _kalshi_date(market: dict) -> str:
    return (market.get("expected_expiration_time") or
            market.get("expiration_time") or
            market.get("close_time") or "")[:10]


def _poly_outcome_index(market: dict, label: str):
    raw = market.get("outcomes")
    outcomes = json.loads(raw) if isinstance(raw, str) else (raw or [])
    lower = label.lower()
    return next((i for i, o in enumerate(outcomes) if str(o).lower() == lower), None)


def _poly_price_raw(market: dict, label: str, fallback_idx: int):
    try:
        raw = market.get("outcomePrices")
        prices = json.loads(raw) if isinstance(raw, str) else (raw or [])
        idx = _poly_outcome_index(market, label)
        if idx is not None and idx < len(prices):
            return float(prices[idx])
        return float(prices[fallback_idx]) if len(prices) > fallback_idx else None
    except Exception:
        return None


def _kalshi_price_raw(market: dict, key: str):
    try:
        v = market.get(key)
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _cents(v) -> str:
    try:
        return f"{round(float(v) * 100)}¢" if v is not None else "—"
    except (TypeError, ValueError):
        return "—"


def _poly_yes(m): return _cents(_poly_price_raw(m, "Yes", 0))
def _poly_no(m): return _cents(_poly_price_raw(m, "No", 1))
def _kalshi_yes(m): return _cents(_kalshi_price_raw(m, "yes_ask_dollars"))
def _kalshi_no(m): return _cents(_kalshi_price_raw(m, "no_ask_dollars"))


def _vol(val):
    try:
        return f"{round(float(val)):,}"
    except (TypeError, ValueError):
        return "—"


def _poly_spread(market: dict) -> str:
    try:
        s = market.get("spread")
        if s is None:
            s = float(market.get("bestAsk")) - float(market.get("bestBid"))
        return f"{round(float(s) * 100)}¢"
    except (TypeError, ValueError):
        return "—"


def _kalshi_spread(market: dict) -> str:
    try:
        bid, ask = market.get("yes_bid_dollars"), market.get("yes_ask_dollars")
        if bid is None or ask is None:
            return "—"
        return f"{round((float(ask) - float(bid)) * 100)}¢"
    except (TypeError, ValueError):
        return "—"


def _ob_size(v):
    try:
        x = float(v or 0)
        return round(x) if x else "—"
    except (TypeError, ValueError):
        return "—"


def _fmt_poly_asks(yes_ob, no_ob):
    def leg(ob):
        return (_cents(ob[0]), _ob_size(ob[1])) if ob else ("—", "—")
    return (*leg(yes_ob), *leg(no_ob))


def _fmt_kalshi_asks(kob):
    if not kob:
        return ("—", "—", "—", "—")
    (a, aq), (b, bq) = kob
    return _cents(a), _ob_size(aq), _cents(b), _ob_size(bq)


OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "all_market_matches.xlsx")


def export_to_excel(poly_markets, kalshi_markets, matches=None,
                    verified=None, arbs=None, poly_ob=None, kalshi_ob=None,
                    filename=OUTPUT_FILE):
    wb = Workbook()
    ws = wb.active
    ws.title = "Markets"

    HDR_FILL = PatternFill("solid", fgColor="1C1C1C")
    HDR_FONT = Font(bold=True, color="FFFFFF", size=10, name="Calibri")
    BODY_FONT = Font(size=10, name="Calibri")
    LABEL_FONT = Font(bold=True, size=10, name="Calibri", color="444444")
    BOTTOM_BORDER = Border(bottom=Side(style="thin", color="CCCCCC"))

    def hdr(cell, value, align="center"):
        cell.value = value
        cell.fill  = HDR_FILL
        cell.font  = HDR_FONT
        cell.alignment = Alignment(horizontal=align, vertical="center")

    def body(cell, value, align="left"):
        cell.value = value
        cell.font  = BODY_FONT
        cell.alignment = Alignment(horizontal=align, vertical="center", wrap_text=True)
        cell.border = BOTTOM_BORDER

    def setup_headers(sheet, cols, labels, text_cols):
        for key, (col, w) in cols.items():
            sheet.column_dimensions[get_column_letter(col)].width = w
        sheet.row_dimensions[1].height = 22
        for key, (col, _) in cols.items():
            if key == "gap":
                sheet.cell(1, col).fill = HDR_FILL
            else:
                hdr(sheet.cell(1, col), labels[key],
                    align="left" if col in text_cols else "center")

    # Polymarket: 1–12 (best asks only)  |  gap: 13  |  Kalshi: 14–25
    P_NUM, P_TITLE, P_DATE, P_YES, P_NO, P_SPREAD = 1, 2, 3, 4, 5, 6
    P_YES_BA, P_YES_BA_QTY, P_NO_BA, P_NO_BA_QTY = 7, 8, 9, 10
    P_V24, P_VTOT = 11, 12
    GAP = 13
    K_NUM, K_TITLE, K_DATE, K_YES, K_NO, K_SPREAD = 14, 15, 16, 17, 18, 19
    K_YES_BA, K_YES_BA_QTY, K_NO_BA, K_NO_BA_QTY = 20, 21, 22, 23
    K_V24, K_VTOT = 24, 25

    for col, w in {
        P_NUM: 4, P_TITLE: 48, P_DATE: 12, P_YES: 9, P_NO: 9, P_SPREAD: 11,
        P_YES_BA: 9, P_YES_BA_QTY: 9, P_NO_BA: 9, P_NO_BA_QTY: 9,
        P_V24: 14, P_VTOT: 14,
        GAP: 3,
        K_NUM: 4, K_TITLE: 48, K_DATE: 12, K_YES: 9, K_NO: 9, K_SPREAD: 11,
        K_YES_BA: 9, K_YES_BA_QTY: 9, K_NO_BA: 9, K_NO_BA_QTY: 9,
        K_V24: 14, K_VTOT: 14,
    }.items():
        ws.column_dimensions[get_column_letter(col)].width = w

    ws.row_dimensions[1].height = 20
    for col, label in [(P_NUM, f"POLYMARKET  ({len(poly_markets)} markets)"),
                       (K_NUM, f"KALSHI  ({len(kalshi_markets)} markets)")]:
        c = ws.cell(1, col, label)
        c.font = LABEL_FONT
        c.alignment = Alignment(vertical="center")

    ws.row_dimensions[2].height = 22
    for col, label in [
        (P_NUM, "#"), (P_TITLE, "Title"), (P_DATE, "Resolves"),
        (P_YES, "Yes (¢)"), (P_NO, "No (¢)"), (P_SPREAD, "Spread (¢)"),
        (P_YES_BA, "Yes ask (¢)"), (P_YES_BA_QTY, "Yes ask qty"), (P_NO_BA, "No ask (¢)"), (P_NO_BA_QTY, "No ask qty"),
        (P_V24, "Vol 24h ($)"), (P_VTOT, "Vol Total ($)"),
        (K_NUM, "#"), (K_TITLE, "Title"), (K_DATE, "Est. Resolves"),
        (K_YES, "Yes (¢)"), (K_NO, "No (¢)"), (K_SPREAD, "Spread (¢)"),
        (K_YES_BA, "Yes ask (¢)"), (K_YES_BA_QTY, "Yes ask qty"), (K_NO_BA, "No ask (¢)"), (K_NO_BA_QTY, "No ask qty"),
        (K_V24, "Vol 24h ($)"), (K_VTOT, "Vol Total ($)"),
    ]:
        hdr(ws.cell(2, col), label, align="left" if col in (P_TITLE, K_TITLE) else "center")
    ws.cell(2, GAP).fill = HDR_FILL

    poly_ob = poly_ob or {}
    kalshi_ob = kalshi_ob or {}

    poly_idx   = {id(m): i for i, m in enumerate(poly_markets)}
    kalshi_idx = {id(m): i for i, m in enumerate(kalshi_markets)}

    for i in range(max(len(poly_markets), len(kalshi_markets))):
        row = i + 3
        ws.row_dimensions[row].height = 32

        if i < len(poly_markets):
            m = poly_markets[i]
            body(ws.cell(row, P_NUM),    i + 1,                             "center")
            body(ws.cell(row, P_TITLE),  m.get("question") or m.get("title") or "")
            body(ws.cell(row, P_DATE),   (m.get("endDate") or "")[:10],    "center")
            body(ws.cell(row, P_YES),    _poly_yes(m),                     "center")
            body(ws.cell(row, P_NO),     _poly_no(m),                      "center")
            body(ws.cell(row, P_SPREAD), _poly_spread(m),                  "center")
            pob = poly_ob.get(i, {})
            yp, yq, np_, nq = _fmt_poly_asks(pob.get("yes"), pob.get("no"))
            body(ws.cell(row, P_YES_BA), yp, "center")
            body(ws.cell(row, P_YES_BA_QTY), yq, "right")
            body(ws.cell(row, P_NO_BA), np_, "center")
            body(ws.cell(row, P_NO_BA_QTY), nq, "right")
            body(ws.cell(row, P_V24),    _vol(m.get("volume24hr")),        "right")
            body(ws.cell(row, P_VTOT),   _vol(m.get("volume")),            "right")

        if i < len(kalshi_markets):
            m = kalshi_markets[i]
            body(ws.cell(row, K_NUM),    i + 1,              "center")
            body(ws.cell(row, K_TITLE),  _kalshi_title(m))
            body(ws.cell(row, K_DATE),   _kalshi_date(m),    "center")
            body(ws.cell(row, K_YES),    _kalshi_yes(m),     "center")
            body(ws.cell(row, K_NO),     _kalshi_no(m),      "center")
            body(ws.cell(row, K_SPREAD), _kalshi_spread(m),  "center")
            ky, kyq, kn, knq = _fmt_kalshi_asks(kalshi_ob.get(i))
            body(ws.cell(row, K_YES_BA), ky, "center")
            body(ws.cell(row, K_YES_BA_QTY), kyq, "right")
            body(ws.cell(row, K_NO_BA), kn, "center")
            body(ws.cell(row, K_NO_BA_QTY), knq, "right")
            body(ws.cell(row, K_V24),    _vol(m.get("volume_24h_fp") or m.get("volume_fp")), "right")
            body(ws.cell(row, K_VTOT),   _vol(m.get("volume_fp")),         "right")

    ws.freeze_panes = "A3"

    MATCH_COLS = {
        "match": (1, 4), "sim": (2, 8),
        "p_title": (3, 48), "p_date": (4, 12),
        "p_yes": (5, 9), "p_no": (6, 9), "p_spread": (7, 11),
        "p_yes_ba": (8, 9), "p_yes_ba_qty": (9, 9), "p_no_ba": (10, 9), "p_no_ba_qty": (11, 9),
        "p_vtot": (12, 14),
        "gap": (13, 3),
        "k_title": (14, 48), "k_date": (15, 12),
        "k_yes": (16, 9), "k_no": (17, 9), "k_spread": (18, 11),
        "k_yes_ba": (19, 9), "k_yes_ba_qty": (20, 9), "k_no_ba": (21, 9), "k_no_ba_qty": (22, 9),
        "k_vtot": (23, 14),
        "shared": (24, 36),
    }
    MATCH_TEXT = {MATCH_COLS["p_title"][0], MATCH_COLS["k_title"][0], MATCH_COLS["shared"][0]}

    def _write_match_sheet(title, score_label, rows, poly_idx, kalshi_idx):
        if not rows:
            return
        sh = wb.create_sheet(title=title)
        labels = {
            "match": "#", "sim": score_label,
            "p_title": "Polymarket Title", "p_date": "Resolves",
            "p_yes": "Yes (¢)", "p_no": "No (¢)", "p_spread": "Spread (¢)",
            "p_yes_ba": "Yes ask (¢)", "p_yes_ba_qty": "Yes ask qty", "p_no_ba": "No ask (¢)", "p_no_ba_qty": "No ask qty",
            "p_vtot": "Vol Total ($)",
            "k_title": "Kalshi Title", "k_date": "Est. Resolves",
            "k_yes": "Yes (¢)", "k_no": "No (¢)", "k_spread": "Spread (¢)",
            "k_yes_ba": "Yes ask (¢)", "k_yes_ba_qty": "Yes ask qty", "k_no_ba": "No ask (¢)", "k_no_ba_qty": "No ask qty",
            "k_vtot": "Vol Total ($)",
            "shared": "Shared Keywords",
        }
        setup_headers(sh, MATCH_COLS, labels, MATCH_TEXT)
        for i, (p, k, shared, score) in enumerate(rows):
            r = i + 2
            pi = poly_idx.get(id(p), -1)
            ki = kalshi_idx.get(id(k), -1)
            pob = poly_ob.get(pi, {}) if pi >= 0 else {}
            kob = kalshi_ob.get(ki) if ki >= 0 else None
            sh.row_dimensions[r].height = 32
            body(sh.cell(r, 1),  i + 1,                                           "center")
            body(sh.cell(r, 2),  f"{score:.0%}",                                  "center")
            body(sh.cell(r, 3),  p.get("question") or p.get("title") or "")
            body(sh.cell(r, 4),  (p.get("endDate") or "")[:10],                   "center")
            body(sh.cell(r, 5),  _poly_yes(p),                                    "center")
            body(sh.cell(r, 6),  _poly_no(p),                                     "center")
            body(sh.cell(r, 7),  _poly_spread(p),                                 "center")
            yp, yq, np_, nq = _fmt_poly_asks(pob.get("yes"), pob.get("no"))
            body(sh.cell(r, 8), yp, "center")
            body(sh.cell(r, 9), yq, "right")
            body(sh.cell(r, 10), np_, "center")
            body(sh.cell(r, 11), nq, "right")
            body(sh.cell(r, 12), _vol(p.get("volume")),                           "right")
            body(sh.cell(r, 14), _kalshi_title(k))
            body(sh.cell(r, 15), _kalshi_date(k),                                 "center")
            body(sh.cell(r, 16), _kalshi_yes(k),                                  "center")
            body(sh.cell(r, 17), _kalshi_no(k),                                   "center")
            body(sh.cell(r, 18), _kalshi_spread(k),                               "center")
            ky, kyq, kn, knq = _fmt_kalshi_asks(kob)
            body(sh.cell(r, 19), ky, "center")
            body(sh.cell(r, 20), kyq, "right")
            body(sh.cell(r, 21), kn, "center")
            body(sh.cell(r, 22), knq, "right")
            body(sh.cell(r, 23), _vol(k.get("volume_fp")),                         "right")
            body(sh.cell(r, 24), ", ".join(sorted(shared)) if shared else "—")
        sh.freeze_panes = "A2"

    _write_match_sheet("Matches",     "Similarity", matches,  poly_idx, kalshi_idx)
    _write_match_sheet("LLM Checker", "Confidence", verified, poly_idx, kalshi_idx)

    if arbs is not None:
        ab = wb.create_sheet(title="Arbitrage")

        BOLD_BODY = Font(bold=True, size=10, name="Calibri")

        ARB_COLS = {
            "num": (1, 4), "margin": (2, 11),
            "p_title": (3, 44),
            "p_yes": (4, 9), "p_yes_q": (5, 9), "p_no": (6, 9), "p_no_q": (7, 9),
            "p_vtot": (8, 14),
            "gap": (9, 3),
            "k_title": (10, 44),
            "k_yes": (11, 9), "k_yes_q": (12, 9), "k_no": (13, 9), "k_no_q": (14, 9),
            "k_vtot": (15, 14),
            "expiration": (16, 14),
        }
        ARB_LABELS = {
            "num": "#", "margin": "Profit %",
            "p_title": "Polymarket Title",
            "p_yes": "Poly Yes ask (¢)", "p_yes_q": "Poly Yes qty", "p_no": "Poly No ask (¢)", "p_no_q": "Poly No qty",
            "p_vtot": "Vol Total ($)",
            "k_title": "Kalshi Title",
            "k_yes": "Kalshi Yes ask (¢)", "k_yes_q": "Kalshi Yes qty", "k_no": "Kalshi No ask (¢)", "k_no_q": "Kalshi No qty",
            "k_vtot": "Vol Total ($)",
            "expiration": "Expiration Date",
        }
        TEXT_ARB = {ARB_COLS["p_title"][0], ARB_COLS["k_title"][0]}
        setup_headers(ab, ARB_COLS, ARB_LABELS, TEXT_ARB)

        if arbs:
            for i, arb in enumerate(arbs):
                row = i + 2
                ab.row_dimensions[row].height = 32
                p, k = arb["poly"], arb["kalshi"]
                pob = {"yes": arb.get("poly_ob_yes"), "no": arb.get("poly_ob_no")}
                kob = arb.get("kalshi_ob")
                p_bet = arb["a_side"] if arb["a_plat"] == "Polymarket" else arb["b_side"]
                k_bet = arb["b_side"] if arb["b_plat"] == "Kalshi" else arb["a_side"]

                body(ab.cell(row, ARB_COLS["num"][0]), i + 1, "center")
                body(ab.cell(row, ARB_COLS["margin"][0]), f"{arb['margin']:.2%}", "center")
                body(ab.cell(row, ARB_COLS["p_title"][0]), p.get("question") or p.get("title") or "")
                yp, yq, np_, nq = _fmt_poly_asks(pob.get("yes"), pob.get("no"))
                body(ab.cell(row, ARB_COLS["p_yes"][0]), yp, "center")
                body(ab.cell(row, ARB_COLS["p_yes_q"][0]), yq, "right")
                body(ab.cell(row, ARB_COLS["p_no"][0]), np_, "center")
                body(ab.cell(row, ARB_COLS["p_no_q"][0]), nq, "right")
                if p_bet == "YES":
                    ab.cell(row, ARB_COLS["p_yes"][0]).font = BOLD_BODY
                    ab.cell(row, ARB_COLS["p_yes_q"][0]).font = BOLD_BODY
                else:
                    ab.cell(row, ARB_COLS["p_no"][0]).font = BOLD_BODY
                    ab.cell(row, ARB_COLS["p_no_q"][0]).font = BOLD_BODY
                body(ab.cell(row, ARB_COLS["p_vtot"][0]), _vol(p.get("volume")), "right")
                body(ab.cell(row, ARB_COLS["k_title"][0]), _kalshi_title(k))
                ky, kyq, kn, knq = _fmt_kalshi_asks(kob)
                body(ab.cell(row, ARB_COLS["k_yes"][0]), ky, "center")
                body(ab.cell(row, ARB_COLS["k_yes_q"][0]), kyq, "right")
                body(ab.cell(row, ARB_COLS["k_no"][0]), kn, "center")
                body(ab.cell(row, ARB_COLS["k_no_q"][0]), knq, "right")
                if k_bet == "YES":
                    ab.cell(row, ARB_COLS["k_yes"][0]).font = BOLD_BODY
                    ab.cell(row, ARB_COLS["k_yes_q"][0]).font = BOLD_BODY
                else:
                    ab.cell(row, ARB_COLS["k_no"][0]).font = BOLD_BODY
                    ab.cell(row, ARB_COLS["k_no_q"][0]).font = BOLD_BODY
                body(ab.cell(row, ARB_COLS["k_vtot"][0]), _vol(k.get("volume_fp")), "right")
                exp = (_kalshi_date(k) or p.get("endDate") or p.get("end_date_iso") or "—")[:10]
                body(ab.cell(row, ARB_COLS["expiration"][0]), exp, "center")
        else:
            ab.row_dimensions[2].height = 28
            c = ab.cell(2, 1, "No arbitrage opportunities found in current matches.")
            c.font = BODY_FONT
            c.alignment = Alignment(horizontal="left", vertical="center")

        ab.freeze_panes = "A2"
        wb.move_sheet(ab, offset=-wb.index(ab))

    wb.save(filename)
    print(f"\n  Excel file saved → {filename}")


def _step(n, total, label):
    print(f"\n{'=' * 60}\nSTEP {n}/{total} — {label}\n{'=' * 60}")


def main():
    total = 6

    _step(1, total, "Fetching Polymarket + Kalshi markets (concurrent)")
    with ThreadPoolExecutor(max_workers=2) as ex:
        poly_fut   = ex.submit(fetch_poly_markets)
        kalshi_fut = ex.submit(fetch_kalshi_markets)
        poly   = poly_fut.result()
        kalshi = kalshi_fut.result()

    _step(2, total, "TF-IDF pre-filter (local)")
    candidates = find_matches(poly, kalshi)
    matches = [c for c in candidates if c[3] >= MATCH_DISPLAY_THRESHOLD]
    print(f"  {len(matches)} at ≥{MATCH_DISPLAY_THRESHOLD} → 'Matches' sheet")

    _step(3, total, "Groq LLM verification")
    verified = verify_matches_with_llm(matches)

    if verified:
        arb_source, arb_label = verified, "LLM Checker"
    else:
        arb_source, arb_label = matches, "Matches"
        print("  ⚠ Using TF-IDF matches for arbitrage (Groq verified none)")

    _step(4, total, "Fetching orderbooks (best ask + size per outcome)")
    matched_poly   = list({id(p): p for p, _k, _, _ in arb_source}.values())
    matched_kalshi = list({id(k): k for _p, k, _, _ in arb_source}.values())
    print(f"  Fetching orderbooks for {len(matched_poly)} Poly + {len(matched_kalshi)} Kalshi matched markets only")

    poly_ob, kalshi_ob = fetch_all_orderbooks(matched_poly, matched_kalshi)

    _step(5, total, "Arbitrage scan (CLOB / Kalshi best asks)")
    arbs = find_arbitrage(arb_source, matched_poly, matched_kalshi, poly_ob, kalshi_ob)

    _step(6, total, "Excel export")
    export_to_excel(poly, kalshi, matches=matches,
                    verified=verified, arbs=arbs,
                    poly_ob=poly_ob, kalshi_ob=kalshi_ob)

    print(f"\n{'=' * 60}\nDONE\n{'=' * 60}")
    print(f"  Poly: {len(poly)}  |  Kalshi: {len(kalshi)}")
    print(f"  TF-IDF candidates: {len(candidates)}  |  Matches (≥{MATCH_DISPLAY_THRESHOLD}): {len(matches)}")
    print(f"  Groq verified (≥{GROQ_CONFIDENCE_CUTOFF:.0%}): {len(verified)}  |  Arb opps: {len(arbs)}")
    print(f"  Arbitrage source: '{arb_label}' sheet")
    for sheet, count in [("Matches", len(matches)),
                         ("LLM Checker", len(verified)),
                         ("Arbitrage", len(arbs))]:
        if count:
            print(f"  → '{sheet}' sheet: {count}")


if __name__ == "__main__":
    main()

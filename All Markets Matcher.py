import re
import json
import requests
import time
from datetime import datetime, timezone
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_sim
from google import genai

CUTOFF_DATE = datetime(2026, 5, 1, tzinfo=timezone.utc)
MIN_VOLUME  = 10_000

# ---------------------------------------------------------------------------
# Gemini LLM Configuration
# Get a free API key at: https://aistudio.google.com/apikey
# Free tier: 15 requests/min, 1 500 requests/day for gemini-2.0-flash
# ---------------------------------------------------------------------------
GEMINI_API_KEY = "AIzaSyBI0QoXmIM_hiCuzN-NtCexFWilKe3tUN4"
GEMINI_MODEL        = "gemini-2.5-flash"
GEMINI_VERIFY_BATCH = 200

_gemini_client = genai.Client(api_key=GEMINI_API_KEY)


def _parse_iso(s: str):
    try:
        return datetime.fromisoformat((s or "").replace("Z", "+00:00"))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Polymarket
# ---------------------------------------------------------------------------

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
    print(f"  → {len(eligible)} eligible (vol ≥ ${MIN_VOLUME:,}, expires before May 1 2026)")
    return eligible


# ---------------------------------------------------------------------------
# Kalshi
# ---------------------------------------------------------------------------

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
_session    = requests.Session()


def _kalshi_get(url, params=None, attempts=5):
    for attempt in range(attempts):
        try:
            r = _session.get(url, params=params, timeout=15)
            if r.status_code == 429:
                time.sleep(1.5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == attempts - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Gave up after {attempts} attempts: {url}")


def fetch_kalshi_markets() -> list:
    print("Fetching Kalshi markets...")
    seen, all_markets = set(), []
    cursor = None
    while True:
        params = {"status": "open", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        data  = _kalshi_get(f"{KALSHI_BASE}/markets", params=params).json()
        batch = data.get("markets", [])
        for m in batch:
            if (tid := m.get("ticker")) not in seen:
                seen.add(tid)
                all_markets.append(m)
        cursor = data.get("cursor")
        if not batch or not cursor:
            break
    print(f"  {len(all_markets)} raw Kalshi markets fetched")

    eligible = [
        m for m in all_markets
        if m.get("status") in {"open", "active"}
        and float(m.get("volume_fp") or 0) >= MIN_VOLUME
        and (d := _parse_iso(m.get("close_time") or m.get("expiration_time")))
        and d < CUTOFF_DATE
    ]
    eligible.sort(key=lambda m: float(m.get("volume_fp") or 0), reverse=True)
    print(f"  → {len(eligible)} eligible (vol ≥ ${MIN_VOLUME:,}, expires before May 1 2026)")
    return eligible


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

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

TFIDF_THRESHOLD         = 0.30
COMBINED_THRESHOLD      = 0.30
MATCH_DISPLAY_THRESHOLD = 0.55
GEMINI_CONFIDENCE_CUTOFF = 0.50

# Synonym rules: longer/more-specific phrases first so they aren't partially
# consumed by shorter rules that follow.
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
    (re.compile(r'\bcuts?\b',           re.I), 'ratecut'),
    (re.compile(r'\bdecrease[sd]?\b',   re.I), 'ratecut'),
    (re.compile(r'\blowere?d?\b',       re.I), 'ratecut'),
    (re.compile(r'\bhikes?\b',          re.I), 'ratehike'),
    (re.compile(r'\bincrease[sd]?\b',   re.I), 'ratehike'),
    (re.compile(r'\braise[sd]?\b',      re.I), 'ratehike'),
    (re.compile(r'\bunchanged\b',       re.I), 'ratehold'),
    (re.compile(r'\bpaused?\b',         re.I), 'ratehold'),
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


def normalize_for_matching(text: str) -> str:
    for pat, rep in _SYNONYM_SUBS:
        text = pat.sub(rep, text)
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
    """TF-IDF + Jaccard pre-filter.  Low thresholds to cast a wide net;
    Gemini verifies the candidates afterwards."""
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


# ---------------------------------------------------------------------------
# Gemini LLM verification
# ---------------------------------------------------------------------------

def _call_gemini_verify(pairs_text: str, max_retries: int = 3) -> list:
    """Ask Gemini to confirm/reject a batch of candidate pairs."""
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
            response = _gemini_client.models.generate_content(
                model=GEMINI_MODEL, contents=prompt,
            )
            raw = response.text.strip()

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
            is_rate_limit = "429" in str(exc)
            wait = 60 if is_rate_limit else 10
            print(f"    Attempt {attempt + 1}/{max_retries}: "
                  f"{'rate limited' if is_rate_limit else 'API error'} – "
                  f"waiting {wait}s...")
            if attempt < max_retries - 1:
                time.sleep(wait)

    print("  ⚠ Gemini API quota exhausted — create a new key in a new project at:")
    print("    https://aistudio.google.com/apikey")
    return []


def verify_matches_with_llm(matches: list) -> list:
    """Send TF-IDF candidate matches to Gemini for semantic verification."""
    if not matches:
        print("  No candidates to verify — skipping Gemini.")
        return []

    n_batches = (len(matches) + GEMINI_VERIFY_BATCH - 1) // GEMINI_VERIFY_BATCH
    print(f"  {len(matches)} candidates → {n_batches} API call(s) "
          f"(batch size {GEMINI_VERIFY_BATCH})")

    verified = []
    bs = GEMINI_VERIFY_BATCH

    for start in range(0, len(matches), bs):
        batch = matches[start:start + bs]
        batch_num = start // bs + 1

        pairs_text = "\n".join(
            f"Pair {i}:\n"
            f"  Polymarket: {p.get('question') or p.get('title') or ''}\n"
            f"  Kalshi:     {_kalshi_title(k)}"
            for i, (p, k, _shared, _score) in enumerate(batch)
        )

        print(f"  Sending batch {batch_num}/{n_batches} to Gemini "
              f"({len(batch)} pairs)...")
        results = _call_gemini_verify(pairs_text)
        confirmed_this_batch = 0

        for r in results:
            idx = r.get("pair_index")
            if idx is None:
                continue
            if not isinstance(idx, int) or idx < 0 or idx >= len(batch):
                continue

            p, k, old_shared, old_score = batch[idx]
            conf = float(r.get("confidence", old_score))
            if conf < GEMINI_CONFIDENCE_CUTOFF:
                continue
            shared = set(r.get("shared_keywords", old_shared))
            verified.append((p, k, shared, conf))
            confirmed_this_batch += 1

        print(f"    Gemini confirmed {confirmed_this_batch}/{len(batch)} "
              f"pairs in this batch")

        if start + len(batch) < len(matches):
            print(f"    Waiting 5s before next batch (rate limit)...")
            time.sleep(5)

    verified.sort(key=lambda x: x[3], reverse=True)
    print(f"  → {len(verified)} verified matches, "
          f"{len(matches) - len(verified)} rejected by Gemini")
    return verified


def find_arbitrage(matches: list) -> list:
    """Return arb opportunities (prob sum < 1) sorted by profit margin."""
    arbs = []
    for poly, kalshi, shared, cosine in matches:
        py = _poly_price_raw(poly, "Yes", 0)
        pn = _poly_price_raw(poly, "No", 1)
        ky = _kalshi_price_raw(kalshi, "yes_ask_dollars")
        kn = _kalshi_price_raw(kalshi, "no_ask_dollars")

        combos = [
            ("Polymarket", "YES", py, "Kalshi", "NO",  kn),
            ("Polymarket", "NO",  pn, "Kalshi", "YES", ky),
        ]
        for a_plat, a_side, a_price, b_plat, b_side, b_price in combos:
            if a_price is None or b_price is None or a_price <= 0 or b_price <= 0:
                continue
            prob_sum = a_price + b_price
            if prob_sum < 1.0:
                arbs.append({
                    "poly":    poly,
                    "kalshi":  kalshi,
                    "a_plat":  a_plat, "a_side":  a_side, "a_price": a_price,
                    "b_plat":  b_plat, "b_side":  b_side, "b_price": b_price,
                    "prob_sum": prob_sum,
                    "margin":  1.0 - prob_sum,
                })

    arbs.sort(key=lambda x: x["margin"], reverse=True)
    print(f"  {len(arbs)} arbitrage opportunity/ies found")
    return arbs


# ---------------------------------------------------------------------------
# Excel helpers
# ---------------------------------------------------------------------------

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


def _fmt_price(v) -> str:
    return f"{round(v * 100)}¢" if v is not None else "—"


def _poly_yes(m):   return _fmt_price(_poly_price_raw(m, "Yes", 0))
def _poly_no(m):    return _fmt_price(_poly_price_raw(m, "No", 1))
def _kalshi_yes(m): return _fmt_price(_kalshi_price_raw(m, "yes_ask_dollars"))
def _kalshi_no(m):  return _fmt_price(_kalshi_price_raw(m, "no_ask_dollars"))


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


# ---------------------------------------------------------------------------
# Excel export
# ---------------------------------------------------------------------------

OUTPUT_FILE = "/Users/leonardoliu/Downloads/Fetch Market/all_market_matches.xlsx"


def export_to_excel(poly_markets, kalshi_markets, matches=None,
                    verified=None, arbs=None, filename=OUTPUT_FILE):
    wb = Workbook()
    ws = wb.active
    ws.title = "Markets"

    HDR_FILL     = PatternFill("solid", fgColor="1C1C1C")
    HDR_FONT     = Font(bold=True, color="FFFFFF", size=10, name="Calibri")
    BODY_FONT    = Font(size=10, name="Calibri")
    LABEL_FONT   = Font(bold=True, size=10, name="Calibri", color="444444")
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

    # Polymarket: cols 1–8  |  gap: col 9  |  Kalshi: cols 10–17
    P_NUM, P_TITLE, P_DATE, P_YES, P_NO, P_SPREAD, P_V24, P_VTOT = 1, 2, 3, 4, 5, 6, 7, 8
    GAP = 9
    K_NUM, K_TITLE, K_DATE, K_YES, K_NO, K_SPREAD, K_V24, K_VTOT = 10, 11, 12, 13, 14, 15, 16, 17

    for col, w in {
        P_NUM: 4, P_TITLE: 48, P_DATE: 12, P_YES: 9, P_NO: 9, P_SPREAD: 11, P_V24: 14, P_VTOT: 14,
        GAP: 3,
        K_NUM: 4, K_TITLE: 48, K_DATE: 12, K_YES: 9, K_NO: 9, K_SPREAD: 11, K_V24: 14, K_VTOT: 14,
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
        (P_YES, "Yes (¢)"), (P_NO, "No (¢)"), (P_SPREAD, "Spread (¢)"), (P_V24, "Vol 24h ($)"), (P_VTOT, "Vol Total ($)"),
        (K_NUM, "#"), (K_TITLE, "Title"), (K_DATE, "Est. Resolves"),
        (K_YES, "Yes (¢)"), (K_NO, "No (¢)"), (K_SPREAD, "Spread (¢)"), (K_V24, "Vol 24h ($)"), (K_VTOT, "Vol Total ($)"),
    ]:
        hdr(ws.cell(2, col), label, align="left" if col in (P_TITLE, K_TITLE) else "center")
    ws.cell(2, GAP).fill = HDR_FILL

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
            body(ws.cell(row, K_V24),    _vol(m.get("volume_24h_fp") or m.get("volume_fp")), "right")
            body(ws.cell(row, K_VTOT),   _vol(m.get("volume_fp")),         "right")

    ws.freeze_panes = "A3"

    MATCH_COLS = {
        "match": (1, 4), "sim": (2, 8),
        "p_title": (3, 48), "p_date": (4, 12),
        "p_yes": (5, 9), "p_no": (6, 9), "p_spread": (7, 11), "p_vtot": (8, 14),
        "gap": (9, 3),
        "k_title": (10, 48), "k_date": (11, 12),
        "k_yes": (12, 9), "k_no": (13, 9), "k_spread": (14, 11), "k_vtot": (15, 14),
        "shared": (16, 36),
    }
    MATCH_TEXT = {MATCH_COLS["p_title"][0], MATCH_COLS["k_title"][0], MATCH_COLS["shared"][0]}

    def _write_match_sheet(title, score_label, rows):
        if not rows:
            return
        sh = wb.create_sheet(title=title)
        labels = {
            "match": "#", "sim": score_label,
            "p_title": "Polymarket Title", "p_date": "Resolves",
            "p_yes": "Yes (¢)", "p_no": "No (¢)", "p_spread": "Spread (¢)", "p_vtot": "Vol Total ($)",
            "k_title": "Kalshi Title", "k_date": "Est. Resolves",
            "k_yes": "Yes (¢)", "k_no": "No (¢)", "k_spread": "Spread (¢)", "k_vtot": "Vol Total ($)",
            "shared": "Shared Keywords",
        }
        setup_headers(sh, MATCH_COLS, labels, MATCH_TEXT)
        for i, (p, k, shared, score) in enumerate(rows):
            r = i + 2
            sh.row_dimensions[r].height = 32
            body(sh.cell(r, 1),  i + 1,                                           "center")
            body(sh.cell(r, 2),  f"{score:.0%}",                                  "center")
            body(sh.cell(r, 3),  p.get("question") or p.get("title") or "")
            body(sh.cell(r, 4),  (p.get("endDate") or "")[:10],                   "center")
            body(sh.cell(r, 5),  _poly_yes(p),                                    "center")
            body(sh.cell(r, 6),  _poly_no(p),                                     "center")
            body(sh.cell(r, 7),  _poly_spread(p),                                 "center")
            body(sh.cell(r, 8),  _vol(p.get("volume")),                           "right")
            body(sh.cell(r, 10), _kalshi_title(k))
            body(sh.cell(r, 11), _kalshi_date(k),                                 "center")
            body(sh.cell(r, 12), _kalshi_yes(k),                                  "center")
            body(sh.cell(r, 13), _kalshi_no(k),                                   "center")
            body(sh.cell(r, 14), _kalshi_spread(k),                               "center")
            body(sh.cell(r, 15), _vol(k.get("volume_fp")),                        "right")
            body(sh.cell(r, 16), ", ".join(sorted(shared)) if shared else "—")
        sh.freeze_panes = "A2"

    _write_match_sheet("Matches",          "Similarity",  matches)
    _write_match_sheet("LLM Checker",      "Confidence",  verified)

    if arbs is not None:
        ab = wb.create_sheet(title="Arbitrage")

        BOLD_BODY = Font(bold=True, size=10, name="Calibri")

        ARB_COLS = {
            "num":       ( 1,  4),
            "margin":    ( 2, 11),
            "prob_sum":  ( 3, 11),
            "p_title":   ( 4, 44),
            "p_yes":     ( 5,  9),
            "p_no":      ( 6,  9),
            "p_spread":  ( 7, 11),
            "p_vtot":    ( 8, 14),
            "gap":       ( 9,  3),
            "k_title":   (10, 44),
            "k_yes":     (11,  9),
            "k_no":      (12,  9),
            "k_spread":  (13, 11),
            "k_vtot":    (14, 14),
        }
        ARB_LABELS = {
            "num": "#", "margin": "Profit %", "prob_sum": "Prob Sum",
            "p_title": "Polymarket Title", "p_yes": "Yes (¢)", "p_no": "No (¢)",
            "p_spread": "Spread (¢)", "p_vtot": "Vol Total ($)",
            "k_title": "Kalshi Title", "k_yes": "Yes (¢)", "k_no": "No (¢)",
            "k_spread": "Spread (¢)", "k_vtot": "Vol Total ($)",
        }
        TEXT_ARB = {ARB_COLS["p_title"][0], ARB_COLS["k_title"][0]}
        setup_headers(ab, ARB_COLS, ARB_LABELS, TEXT_ARB)

        if arbs:
            for i, arb in enumerate(arbs):
                row = i + 2
                ab.row_dimensions[row].height = 32
                p, k = arb["poly"], arb["kalshi"]

                p_bet = arb["a_side"] if arb["a_plat"] == "Polymarket" else arb["b_side"]
                k_bet = arb["b_side"] if arb["b_plat"] == "Kalshi"     else arb["a_side"]

                body(ab.cell(row, ARB_COLS["num"][0]),      i + 1,                                   "center")
                body(ab.cell(row, ARB_COLS["margin"][0]),   f"{arb['margin']:.2%}",                  "center")
                body(ab.cell(row, ARB_COLS["prob_sum"][0]), f"{arb['prob_sum']:.4f}",                 "center")
                body(ab.cell(row, ARB_COLS["p_title"][0]),  p.get("question") or p.get("title") or "")

                body(ab.cell(row, ARB_COLS["p_yes"][0]),    _poly_yes(p),                            "center")
                body(ab.cell(row, ARB_COLS["p_no"][0]),     _poly_no(p),                             "center")
                if p_bet == "YES":
                    ab.cell(row, ARB_COLS["p_yes"][0]).font = BOLD_BODY
                else:
                    ab.cell(row, ARB_COLS["p_no"][0]).font = BOLD_BODY
                body(ab.cell(row, ARB_COLS["p_spread"][0]), _poly_spread(p),                         "center")
                body(ab.cell(row, ARB_COLS["p_vtot"][0]),   _vol(p.get("volume")),                   "right")

                body(ab.cell(row, ARB_COLS["k_title"][0]),  _kalshi_title(k))

                body(ab.cell(row, ARB_COLS["k_yes"][0]),    _kalshi_yes(k),                          "center")
                body(ab.cell(row, ARB_COLS["k_no"][0]),     _kalshi_no(k),                           "center")
                if k_bet == "YES":
                    ab.cell(row, ARB_COLS["k_yes"][0]).font = BOLD_BODY
                else:
                    ab.cell(row, ARB_COLS["k_no"][0]).font = BOLD_BODY
                body(ab.cell(row, ARB_COLS["k_spread"][0]), _kalshi_spread(k),                       "center")
                body(ab.cell(row, ARB_COLS["k_vtot"][0]),   _vol(k.get("volume_fp")),                "right")
        else:
            ab.row_dimensions[2].height = 28
            c = ab.cell(2, 1, "No arbitrage opportunities found in current matches.")
            c.font = BODY_FONT
            c.alignment = Alignment(horizontal="left", vertical="center")

        ab.freeze_panes = "A2"

    wb.save(filename)
    print(f"\n  Excel file saved → {filename}")


def _step(n, total, label):
    print(f"\n{'=' * 60}\nSTEP {n}/{total} — {label}\n{'=' * 60}")


def main():
    total = 6

    _step(1, total, "Fetching Polymarket markets")
    poly = fetch_poly_markets()

    _step(2, total, "Fetching Kalshi markets")
    kalshi = fetch_kalshi_markets()

    _step(3, total, "TF-IDF pre-filter (local)")
    candidates = find_matches(poly, kalshi)
    matches = [c for c in candidates if c[3] >= MATCH_DISPLAY_THRESHOLD]
    print(f"  {len(matches)} at ≥{MATCH_DISPLAY_THRESHOLD} → 'Matches' sheet")

    _step(4, total, "Gemini LLM verification")
    verified = verify_matches_with_llm(matches)

    if verified:
        arb_source = verified
        arb_label = "LLM Checker"
    else:
        arb_source = matches
        arb_label = "Matches"
        print("  ⚠ No verified matches from Gemini — falling back to TF-IDF Matches for arbitrage")

    _step(5, total, "Arbitrage scan")
    arbs = find_arbitrage(arb_source)

    _step(6, total, "Excel export")
    export_to_excel(poly, kalshi, matches=matches,
                    verified=verified, arbs=arbs)

    print(f"\n{'=' * 60}\nDONE\n{'=' * 60}")
    print(f"  Poly: {len(poly)}  |  Kalshi: {len(kalshi)}")
    print(f"  TF-IDF candidates: {len(candidates)}  |  Matches (≥{MATCH_DISPLAY_THRESHOLD}): {len(matches)}")
    print(f"  Gemini verified (≥{GEMINI_CONFIDENCE_CUTOFF:.0%}): {len(verified)}  |  Arb opps: {len(arbs)}")
    print(f"  Arbitrage source: '{arb_label}' sheet")
    for sheet, count in [("Matches", len(matches)),
                         ("LLM Checker", len(verified)),
                         ("Arbitrage", len(arbs))]:
        if count:
            print(f"  → '{sheet}' sheet: {count}")


if __name__ == "__main__":
    main()

"""Remember LLM verdicts on market pairs so later scans skip repeat calls.

A verdict is reused only while both markets still read the same: the entry
stores a fingerprint of every field that states what a market resolves on
(question, rules, outcomes, deadline, settlement source), plus the model and
the prompt version. Change any of those and the fingerprint no longer matches,
so the pair is verified again.

Both answers are cached. A cached "different markets" keeps a rejected pair
from being re-sent every scan; it can hide an opportunity but never invent one.
Failed or timed-out calls are never written here.

The file lives in `.cache/` next to the project (`/tmp` on Vercel, where the
project directory is read-only). Serverless instances lose `/tmp` between cold
starts, so the web app also hands the entries to the browser and takes them
back as a seed on the next scan.
"""

import hashlib
import json
import os
import tempfile
import threading
import time

CACHE_VERSION = 1
MAX_SEED_ENTRIES = 20_000

# Fields that state what each market resolves on. Prices, volumes and other
# churn are left out on purpose.
POLY_RULE_FIELDS = ("conditionId", "question", "description", "resolutionSource",
                    "outcomes", "endDate")
KALSHI_RULE_FIELDS = ("ticker", "title", "yes_sub_title", "no_sub_title",
                      "rules_primary", "rules_secondary", "close_time",
                      "can_close_early", "early_close_condition",
                      "strike_type", "floor_strike", "cap_strike",
                      "functional_strike", "custom_strike")


def default_path() -> str:
    override = os.environ.get("VERIFY_CACHE_PATH")
    if override:
        return override
    if os.environ.get("VERCEL"):
        return os.path.join(tempfile.gettempdir(), "pmaf-verified-pairs.json")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, ".cache", "verified_pairs.json")


def pair_key(poly: dict, kalshi: dict) -> str:
    return f"{poly.get('conditionId') or poly.get('id') or ''}|{kalshi.get('ticker') or ''}"


def _norm(value):
    return " ".join(value.split()) if isinstance(value, str) else value


def fingerprint(poly: dict, kalshi: dict, model: str, prompt_version: str) -> str:
    payload = {
        "p": {f: _norm(poly.get(f)) for f in POLY_RULE_FIELDS},
        "k": {f: _norm(kalshi.get(f)) for f in KALSHI_RULE_FIELDS},
        "model": model, "prompt": prompt_version,
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def _valid_entry(entry) -> bool:
    return (isinstance(entry, dict)
            and isinstance(entry.get("fp"), str)
            and isinstance(entry.get("match"), bool)
            and isinstance(entry.get("confidence"), (int, float))
            and isinstance(entry.get("ts"), (int, float)))


class VerifyCache:
    def __init__(self, path: str = "", ttl_days: float = 30.0,
                 model: str = "", prompt_version: str = ""):
        self.path = path
        self.ttl = ttl_days * 86400
        self.model = model
        self.prompt_version = prompt_version
        self.entries = {}
        self._lock = threading.Lock()
        self._dirty = False

    @classmethod
    def load(cls, path: str, ttl_days: float, model: str, prompt_version: str,
             seed: dict = None):
        cache = cls(path, ttl_days, model, prompt_version)
        if path:
            try:
                with open(path, encoding="utf-8") as handle:
                    data = json.load(handle)
                if isinstance(data, dict) and data.get("version") == CACHE_VERSION:
                    cache.merge(data.get("entries"))
                    cache._dirty = False
            except (OSError, ValueError):
                pass                      # A missing or damaged cache is just empty.
        if seed:
            cache.merge(seed)
        return cache

    def merge(self, entries) -> int:
        """Add entries (from disk or a browser seed), keeping the newer of each."""
        if not isinstance(entries, dict):
            return 0
        added = 0
        with self._lock:
            for key, entry in list(entries.items())[:MAX_SEED_ENTRIES]:
                if not isinstance(key, str) or not _valid_entry(entry):
                    continue
                current = self.entries.get(key)
                if current is None or entry["ts"] > current["ts"]:
                    self.entries[key] = {
                        "fp": entry["fp"], "match": entry["match"],
                        "confidence": float(entry["confidence"]),
                        "shared": [str(w) for w in (entry.get("shared") or [])][:12],
                        "ts": float(entry["ts"]),
                    }
                    added += 1
                    self._dirty = True
        return added

    def get(self, poly: dict, kalshi: dict):
        """The stored verdict, or None if absent, expired, or the rules changed."""
        entry = self.entries.get(pair_key(poly, kalshi))
        if not entry:
            return None
        if self.ttl and time.time() - entry["ts"] > self.ttl:
            return None
        if entry["fp"] != fingerprint(poly, kalshi, self.model, self.prompt_version):
            return None
        return entry

    def put(self, poly: dict, kalshi: dict, match: bool, confidence: float, shared=()):
        with self._lock:
            self.entries[pair_key(poly, kalshi)] = {
                "fp": fingerprint(poly, kalshi, self.model, self.prompt_version),
                "match": bool(match), "confidence": float(confidence),
                "shared": sorted(str(w) for w in shared)[:12],
                "ts": time.time(),
            }
            self._dirty = True

    def export(self, pairs) -> dict:
        """Entries for these (poly, kalshi) pairs — what the browser keeps."""
        out = {}
        for poly, kalshi in pairs:
            key = pair_key(poly, kalshi)
            if key in self.entries:
                out[key] = self.entries[key]
        return out

    def save(self) -> bool:
        if not self.path or not self._dirty:
            return False
        with self._lock:
            now = time.time()
            if self.ttl:
                self.entries = {k: e for k, e in self.entries.items()
                                if now - e["ts"] <= self.ttl}
            body = json.dumps({"version": CACHE_VERSION, "entries": self.entries})
            try:
                os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
                tmp = f"{self.path}.{os.getpid()}.{threading.get_ident()}.tmp"
                with open(tmp, "w", encoding="utf-8") as handle:
                    handle.write(body)
                os.replace(tmp, self.path)   # Atomic: readers never see half a file.
            except OSError:
                return False                 # Read-only host: the seed still works.
            self._dirty = False
            return True

"""Scan configuration — every knob the web UI can set."""

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

DEFAULT_HORIZON_DAYS = 180


def _load_local_env() -> None:
    """Read a gitignored .env so a local GROQ_API_KEY is used without exporting it."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    try:
        with open(path, encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                if key.startswith("export "):
                    key = key[len("export "):].strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        return


_load_local_env()


def _default_cutoff() -> datetime:
    """A rolling horizon — a hard-coded date silently empties the scan once passed."""
    return datetime.now(timezone.utc) + timedelta(days=DEFAULT_HORIZON_DAYS)


@dataclass
class ScanConfig:
    # Market eligibility
    min_volume: float = 25_000.0
    cutoff_date: datetime = field(default_factory=_default_cutoff)

    # Matching thresholds
    tfidf_threshold: float = 0.45
    combined_threshold: float = 0.45
    match_display_threshold: float = 0.45
    max_date_gap_days: int = 45

    # LLM verification (Groq)
    use_llm: bool = True
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    groq_batch_size: int = 50
    groq_confidence_cutoff: float = 0.65

    # Arbitrage
    min_margin: float = 0.0          # e.g. 0.01 => only show >= 1% edge

    # Fetch tuning
    orderbook_workers: int = 80
    poly_max_markets: int = 20_000

    def __post_init__(self):
        if not self.groq_api_key:
            self.groq_api_key = os.environ.get("GROQ_API_KEY", "")
        if isinstance(self.cutoff_date, str):
            self.cutoff_date = parse_cutoff(self.cutoff_date)

    @property
    def llm_enabled(self) -> bool:
        return bool(self.use_llm and self.groq_api_key)


def parse_cutoff(value: str) -> datetime:
    """Accept 'YYYY-MM-DD' from the UI's date input."""
    d = datetime.strptime(value.strip()[:10], "%Y-%m-%d")
    return d.replace(tzinfo=timezone.utc)

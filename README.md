# Prediction Markets Arbitrage Finder

Finds arbitrage opportunities between [Polymarket](https://polymarket.com) and
[Kalshi](https://kalshi.com) by matching equivalent markets across the two
platforms and comparing orderbook prices.

**Pipeline:** fetch eligible markets from both platforms → TF-IDF similarity
matching with entity/number/date filters → optional Claude LLM verification of
matches → orderbook scan for pairs where `YES ask + NO ask < $1.00` → export to
Excel (`all_market_matches.xlsx`), JSON (`results.json`), and a local web
dashboard.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...   # optional but recommended
```

The API key is read **only** from the `ANTHROPIC_API_KEY` environment variable —
never hardcode keys in source files. Without a key, the scan still runs but
skips LLM verification and uses raw TF-IDF matches for the arbitrage scan.

## Usage

### Dashboard (recommended)

```bash
python3 dashboard.py
```

Opens `http://localhost:8400` in your browser. Press **Run scan** to fetch
markets, watch live progress in the *Scan log* tab, and browse the results:

- **Arbitrage** — profitable pairs with the exact legs to buy (side, platform,
  price, available size), max contracts, expiration, and a LIVE badge for
  markets expiring today
- **LLM verified** — market pairs Claude confirmed resolve identically
- **TF-IDF matches** — all candidate pairs above the similarity threshold

The dashboard supports light/dark mode and has a text filter across all tables.

### Command line only

```bash
python3 matcher.py
```

Writes `all_market_matches.xlsx` and `results.json` next to the script.
`results.json` and the Excel output are gitignored — your scan results stay
local.

## Configuration

Edit the constants at the top of `matcher.py`:

| Constant | Default | Meaning |
|---|---|---|
| `MIN_VOLUME` | `25_000` | minimum market volume ($) |
| `CUTOFF_DATE` | `2026-07-01` | only markets expiring before this date |
| `TFIDF_THRESHOLD` | `0.45` | similarity floor for candidate pairs |
| `GROQ_CONFIDENCE_CUTOFF` | `0.65` | minimum LLM confidence to keep a match |

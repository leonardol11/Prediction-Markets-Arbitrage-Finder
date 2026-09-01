# Prediction Market Arbitrage Finder

Finds the same question listed on **Polymarket** and **Kalshi**, then checks
whether buying YES on one venue and NO on the other costs less than $1.00 — a
locked-in profit, since exactly one leg pays out at resolution.

Runs as a website: live progress, five spreadsheet-style tabs, and a one-click
`.xlsx` export of exactly what you see.

<pre>
Arbitrage   LLM Verified   Matches   Polymarket   Kalshi
</pre>

## Run it

```bash
pip3 install -r requirements.txt
python3 app.py
```

That opens <http://127.0.0.1:5000> in your browser. Press **Run scan**.

`PORT=8080 python3 app.py` moves it; `NO_BROWSER=1` stops it opening a tab. It's
Flask's development server — fine on your own machine, not something to expose
to the internet as-is.

A full scan takes roughly 4–8 minutes; almost all of it is step 1. Kalshi's API
has no volume filter and ignores its own `order` parameter, so the only way to
find the liquid markets is to page through every open one — about 2.1 million.
The scan splits that walk into parallel close-time windows and streams progress
as it goes. Results stay loaded after it finishes, and survive a page refresh.

### LLM verification (optional but recommended)

Text similarity alone will happily pair *"Will Ben Shelton win the match?"* with
*"Will Ben Shelton win set 1?"* — different questions, and the "arbitrage"
between them is imaginary. The LLM pass reads each candidate pair and keeps only
the ones that genuinely resolve alike.

Get a free key at <https://console.groq.com/keys>, then either paste it into
**Settings** in the UI (kept in your browser only) or export it:

```bash
export GROQ_API_KEY=gsk_…
python3 app.py
```

Without a key the scan still runs, and the Arbitrage tab falls back to raw
TF-IDF matches — treat those as leads to check by hand, not as trades. The
**Match** column on each arb row shows how confident the pairing is.

> The key hard-coded in the original `All Markets Matcher.py` is committed to
> this repo's history and now returns 401. Treat it as burned: issue a new one
> and keep it in the environment or the UI field, never in the source.

## Settings

| Setting | Default | What it does |
| --- | --- | --- |
| Min volume | $25,000 | Skips illiquid markets on both venues. |
| Resolves before | today + 180 days | Rolling horizon. Shorter = much faster scan. |
| Similarity threshold | 0.45 | Lower finds more pairs, and more junk. |
| Min edge | 0% | Hides arbs below this profit per $1.00 pair. |
| LLM confidence | 0.65 | Cutoff for keeping an LLM-confirmed pair. |

## Command line

Same pipeline, straight to a workbook:

```bash
python3 cli.py --min-volume 50000 --cutoff 2026-12-31 --out results.xlsx
python3 cli.py --no-llm --min-margin 0.02        # only edges of 2%+
```

## How it works

| Step | What happens |
| --- | --- |
| 1 | Fetch eligible markets from both venues, concurrently. |
| 2 | TF-IDF + Jaccard over the question text, with guards on numbers, months, named entities and resolution dates. |
| 3 | An LLM confirms each surviving pair resolves on the same outcome. |
| 4 | Pull live best asks — Polymarket CLOB books and Kalshi orderbooks — for matched markets only. |
| 5 | Flag every pair whose two legs cost under $1.00 combined. |

## Where results live

Each finished scan is cached to `.cache/last_scan.json`, so a page refresh — or a
server restart — brings the last run straight back. Downloading rebuilds the
workbook from that same data, which is why the sheets and the `.xlsx` can never
disagree.

## Layout

```
app.py              Flask server — routes, SSE progress stream, .xlsx download
cli.py              Same pipeline, no browser
matcher/
  config.py         ScanConfig — every knob the UI exposes
  core.py           Fetching, matching, LLM verification, arbitrage scan
  sheets.py         One sheet model feeding both the web grid and Excel
  excel.py          Styled workbook writer
templates/, static/ The website
```

## Caveats

Edges are **gross**. Before trusting one, account for:

- **Fees** — Kalshi charges on trades and settlement; Polymarket has gas.
- **Depth** — `Ask qty` is the size resting at that price. `Max pairs` is the
  smaller of the two legs; past that the price moves.
- **Staleness** — books move in seconds. Re-check before sending an order.
- **Resolution risk** — two markets can read alike and settle differently on
  source, timezone, or edge-case wording. That risk is the whole game.

Nothing here places trades or touches an exchange account.

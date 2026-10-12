# Prediction Market Arbitrage Finder

Finds the same question listed on **Polymarket** and **Kalshi**, then checks
whether buying YES on one venue and NO on the other costs less than $1.00 — a
locked-in profit, since exactly one leg pays out at resolution.

Runs as a responsive website: live progress, five spreadsheet-style tabs,
automatic 30-second price refresh, and a one-click `.xlsx` export.

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
Flask's development server, which is fine on your own machine. To host it, deploy
to Vercel (below).

A full scan takes roughly 4–8 minutes; almost all of it is step 1. Kalshi's API
has no volume filter and ignores its own `order` parameter, so the only way to
find the liquid markets is to page through every open one — about 2.1 million.
The scan splits that walk into parallel close-time windows and streams progress
as it goes. Keep the tab open while it runs: closing it stops the scan. Results
stay loaded after it finishes, and your browser keeps them across page reloads.

## Deploy to Vercel

The app is ready for Vercel's Python runtime with no extra setup:

```bash
npm i -g vercel
vercel deploy          # or import the Git repo in the Vercel dashboard
```

Vercel finds the Flask `app` in `app.py`, installs `requirements.txt`, and uses
Python 3.12 from `.python-version`. Files in `public/` are served from the CDN.
Set `GROQ_API_KEY` under **Project → Settings → Environment Variables** if you
want the server to supply the LLM key instead of each visitor.

**Time limit.** A full scan with default settings takes 4–8 minutes. Vercel's
Hobby plan stops functions after 300 seconds, so `vercel.json` sets
`maxDuration` to 300 and the app stops scans itself at 270 seconds with a clear
message. To stay under that, use a sooner **Resolves before** date or a higher
**Min volume**. On a Pro plan you can raise both limits:

1. Set `maxDuration` in `vercel.json` to up to `800`.
2. Set the `SCAN_TIME_LIMIT` environment variable about 30 seconds lower.

**How it fits serverless.** The server keeps nothing between requests. A scan
runs inside one streaming request. The final message carries the results and a
compact context for repricing, which the browser stores in IndexedDB. Live
refresh and Excel export send that data back, gzip-compressed to stay under
Vercel's 4.5 MB request limit.

### Live dashboard

After a full scan, **Auto refresh 30s** is on by default. While the page is open,
it fetches fresh orderbooks for matched markets and recalculates arbitrage every
30 seconds. **Refresh now** requests an immediate update; uncheck auto refresh
to pause. Slow requests and active scans skip refresh ticks instead of overlapping.
Background browser tabs may throttle timers.

The dashboard shows separate **Prices updated** and **Markets discovered**
timestamps. Refresh preserves your selected tab, filter, sorting, and scroll
position. Failed refreshes keep the last snapshot visible with a stale-data
warning; the next automatic attempt can recover.

Price refresh does not repeat market discovery or LLM verification. Use **Run
scan** to find newly listed markets or apply changed settings. Raw Polymarket
and Kalshi tabs, plus listed-price metadata in match rows, remain discovery
snapshots; matched best-ask columns and arbitrage are repriced.

Older saved results remain viewable, but require one new full scan to enable
live refresh. Scans without matched markets also require a new discovery scan.

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

The browser saves each finished scan in IndexedDB, so a page reload brings the
last run straight back. The saved snapshot includes the matched markets needed
to reprice it, never API keys. Successful price refreshes update the saved
snapshot. Downloading sends the displayed sheets to the server, which builds the
workbook from the same sheet model. Results are per browser: another device or
browser starts empty.

## Layout

```
app.py              Flask server: streaming scan, repricing, .xlsx export (stateless)
cli.py              Same pipeline, no browser
vercel.json         Function duration and bundle exclusions for Vercel
matcher/
  config.py         ScanConfig — every knob the UI exposes
  core.py           Fetching, matching, LLM verification, arbitrage scan
  sheets.py         One sheet model feeding both the web grid and Excel
  excel.py          Styled workbook writer
templates/, public/ The website (public/ is served by Vercel's CDN)
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

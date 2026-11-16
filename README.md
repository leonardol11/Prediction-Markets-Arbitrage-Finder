# Prediction Market Arbitrage Finder

Finds the same question listed on **Polymarket** and **Kalshi**, then checks
whether buying YES on one venue and NO on the other costs less than $1.00 — a
locked-in profit, since exactly one leg pays out at resolution.

Runs as a responsive website: live progress, five spreadsheet-style tabs,
automatic 30-second price refresh, and a one-click `.xlsx` export.

<pre>
Live Arbitrage   LLM Verified   Matches   Polymarket   Kalshi
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

A scan takes about a minute. Around 25 seconds of that is Kalshi: its market
list has no volume filter, so step 1 reads every open event and keeps the liquid
markets nested in them. The rest is LLM verification of new pairs, capped by the
**LLM time budget**. Verdicts are remembered, so pairs a scan does not reach are
picked up by the next one and nothing is verified twice. Progress streams as those
pages come in. Keep the tab open while it runs: closing it stops the scan.
Results stay loaded after it finishes, and your browser keeps them across page
reloads.

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

**Time limit.** Vercel's Hobby plan stops functions after 300 seconds, so
`vercel.json` sets `maxDuration` to 300 and the app stops scans itself at 270
seconds with a clear message. Event paging normally finishes well inside that.
If a scan still hits the limit, shorten **Resolves before** or raise the
minimum volume. On a Pro plan you can raise both limits:

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

Only LLM-verified pairs can appear in **Live Arbitrage**. Without a key the scan
still runs and lists title matches in **Matches** (marked *Unverified*), but none
of them is reported as arbitrage. A failed or timed-out LLM call leaves a pair
unverified; it is never read as a match.

**What the LLM checks.** Each pair gets a strict check that reads both markets'
resolution rules, end dates and outcomes, and must confirm that Polymarket YES
and Kalshi YES always resolve the same way. Pairs with no edge first pass a
cheap title screen so obviously different questions cost few tokens.

**Order and budget.** Groq keys are rate limited by tokens per minute, and that,
not a fixed pause, is what bounds this step. Requests run concurrently inside
the limit Groq reports in its response headers and back off when told to. Pairs
are checked most-promising first: a small positive edge after fees, then larger
edges (two *different* markets nearly always look mispriced, so a huge edge
usually means a bad match), then near misses, then the rest. Work stops at the
**LLM time budget** and resumes on the next scan.

**Verdict cache.** Verdicts are saved in `.cache/verified_pairs.json` and reused
while both markets' rules, outcomes, deadline and settlement source are
unchanged; any change to those sends the pair back to the LLM. Entries expire
after 30 days. On Vercel, where the disk does not persist, the browser keeps the
verdicts and returns them with the next scan. Delete the file to start over.

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
| Watch within | 2% | Verified pairs this close to break-even are marked *Near*. |
| LLM confidence | 0.65 | Cutoff for keeping an LLM-confirmed pair. |
| LLM time budget | 45 s | Time one scan may spend on new verdicts. 0 = verify everything, which can take many minutes on a fresh cache. |

## Command line

Same pipeline, straight to a workbook:

```bash
python3 cli.py --min-volume 50000 --cutoff 2026-12-31 --out results.xlsx
python3 cli.py --min-margin 0.02                 # only edges of 2%+
python3 cli.py --llm-budget 0                    # verify every candidate pair
```

## How it works

| Step | What happens |
| --- | --- |
| 1 | Fetch eligible markets from both venues, concurrently. Kalshi comes from open events, which leaves combo contracts out. |
| 2 | TF-IDF + Jaccard over the question text, with guards on numbers, months, named entities and resolution dates. |
| 3 | Read Polymarket order books in bulk and work out each candidate's preliminary edge after both venues' fees, using the quote Kalshi listed with the market. |
| 4 | LLM verification, most promising pairs first, reusing cached verdicts. |
| 5 | Read fresh full-depth books on both venues for every verified pair and keep the ones that cost under $1.00 per pair after fees. |

**Live Arbitrage** walks both order books level by level, so *Exec. size* and
*Est. profit* cover everything that can be filled at a profit, not only the best
ask. Kalshi's fee is rounded up to the cent once per order. Quotes older than 30
seconds, and sizes below Polymarket's minimum order, are left out.

Each stage logs its duration, and the scan log ends with the total.

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
  fees.py           Taker fee formulas for both venues
  sheets.py         One sheet model feeding both the web grid and Excel
  excel.py          Styled workbook writer
templates/, public/ The website (public/ is served by Vercel's CDN)
```

## Caveats

Edges are **net of taker fees** (every leg buys at the ask, so it is a taker):

- **Polymarket**: `shares × rate × p × (1 − p)`, where `rate` comes from each
  market's `feeSchedule` (e.g. 0.04 politics, 0.07 crypto, 0 geopolitics).
  [Docs](https://docs.polymarket.com/trading/fees).
- **Kalshi**: `0.07 × contracts × P × (1 − P) × fee_multiplier`, rounded up to
  the cent per order. The multiplier comes from the market's series (1, 0.5, or
  0). [Fee rounding](https://docs.kalshi.com/getting_started/fee_rounding).
- Missing fee data is priced at the highest standard rate and marked `est.`.

Ranking is by annualized net return. Before trusting an edge, still account for:

- **Withdrawal / on-ramp costs**: bank, card or bridge fees outside either venue.
- **Depth** — `Ask qty` is the size resting at that price. `Max pairs` is the
  smaller of the two legs; past that the price moves.
- **Staleness** — books move in seconds. Re-check before sending an order.
- **Resolution risk** — two markets can read alike and settle differently on
  source, timezone, or edge-case wording. That risk is the whole game.

Nothing here places trades or touches an exchange account.

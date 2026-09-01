"""Command-line runner — same pipeline as the website, straight to .xlsx.

    python3 cli.py --min-volume 50000 --no-llm --out results.xlsx
"""

import argparse
import os
import sys

from matcher import ScanConfig, build_sheets, build_summary, parse_cutoff, run_scan, save_workbook


def main():
    ap = argparse.ArgumentParser(description="Polymarket × Kalshi arbitrage scan")
    d = ScanConfig()
    ap.add_argument("--min-volume", type=float, default=d.min_volume)
    ap.add_argument("--cutoff", default=d.cutoff_date.strftime("%Y-%m-%d"),
                    help="only markets resolving before this date (YYYY-MM-DD)")
    ap.add_argument("--similarity", type=float, default=d.match_display_threshold)
    ap.add_argument("--min-margin", type=float, default=0.0,
                    help="minimum edge as a fraction, e.g. 0.01 for 1%%")
    ap.add_argument("--no-llm", action="store_true", help="skip Groq verification")
    ap.add_argument("--out", default="all_market_matches.xlsx")
    args = ap.parse_args()

    cfg = ScanConfig(
        min_volume=args.min_volume,
        cutoff_date=parse_cutoff(args.cutoff),
        tfidf_threshold=min(args.similarity, d.tfidf_threshold),
        combined_threshold=args.similarity,
        match_display_threshold=args.similarity,
        min_margin=args.min_margin,
        use_llm=not args.no_llm,
    )
    if cfg.use_llm and not cfg.groq_api_key:
        print("  note: GROQ_API_KEY is unset — running without LLM verification\n")

    result = run_scan(cfg)
    path = save_workbook(build_sheets(result), os.path.abspath(args.out))
    s = build_summary(result)
    print(f"\n  {s['arb_count']} arbs · best edge {s['best_margin_display']}")
    print(f"  saved → {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

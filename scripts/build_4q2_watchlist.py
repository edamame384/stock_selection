from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a stock_signal watchlist from 4Q-2 selected candidates.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("projects/shikiho_text_parser/output/4q2_selection/4q2_selected_candidates.csv"),
        help="CSV containing selected 4Q-2 candidates with a ticker column.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/tmp_watchlist_4q2_only.csv"),
        help="Output watchlist CSV for stock_signal.py.",
    )
    parser.add_argument(
        "--group-name",
        type=str,
        default="4q2",
        help="Group label to write into the watchlist.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    if "ticker" not in df.columns:
        raise ValueError("input CSV must contain a 'ticker' column")

    out = (
        df.loc[:, ["ticker"]]
        .dropna()
        .rename(columns={"ticker": "symbol"})
        .drop_duplicates()
        .copy()
    )
    out.insert(0, "group", args.group_name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"[WATCHLIST_4Q2] rows={len(out)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

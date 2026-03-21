from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run named method: q3_post_high_vol")
    parser.add_argument("--selected-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-date", type=str, default="2025-07-01")
    parser.add_argument("--end-date", type=str, default="2025-09-30")
    parser.add_argument("--initial-capital", type=float, default=3_000_000.0)
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(ROOT / "projects" / "quarterly_ranker" / "backtest_q3_post_high_vol.py"),
        "--selected-csv",
        str(args.selected_csv),
        "--output-dir",
        str(args.output_dir),
        "--start-date",
        args.start_date,
        "--end-date",
        args.end_date,
        "--initial-capital",
        str(args.initial_capital),
        "--trend-r2-min",
        "0.60",
        "--annual-return-min",
        "25",
        "--quarter-return-min",
        "10",
        "--positive-month-ratio-min",
        "50",
        "--persistence-20d-min",
        "55",
        "--sector-per-score-min",
        "0.35",
        "--ocr-per-max",
        "20",
        "--entry-limit-pct",
        "1.5",
        "--take-profit-pct",
        "8",
        "--stop-loss-pct",
        "5",
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

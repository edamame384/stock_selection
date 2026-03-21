from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run named method: Nikkei regime switch")
    parser.add_argument("--selected-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, required=True)
    parser.add_argument("--initial-capital", type=float, default=3_000_000.0)
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(ROOT / "projects" / "shikiho_text_parser" / "backtest_q4_regime_switch.py"),
        "--selected-csv",
        str(args.selected_csv),
        "--output-dir",
        str(args.output_dir),
        "--condition-name",
        "method_regime_switch_日経切替",
        "--start-date",
        args.start_date,
        "--end-date",
        args.end_date,
        "--initial-capital",
        str(args.initial_capital),
        "--up-ret20-threshold",
        "0.0",
        "--sideways-abs-ret20-threshold",
        "0.07",
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

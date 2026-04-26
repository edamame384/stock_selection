from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BACKTEST = ROOT / "projects" / "shikiho_text_parser" / "backtest_phase_adaptive_practical_v34_rebuild_interval_candidate.py"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run v3.4 rebuild-interval candidate.")
    parser.add_argument("--detail-csv", type=Path)
    parser.add_argument("--selected-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-date", type=str)
    parser.add_argument("--end-date", type=str)
    parser.add_argument("--dataset-name", type=str)
    parser.add_argument("--initial-capital", type=float, default=3_000_000.0)
    parser.add_argument("--rebuild-interval-trading-days", type=int, default=20)
    parser.add_argument("--batch", action="store_true")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(BACKTEST),
        "--output-dir",
        str(args.output_dir),
        "--earnings-pre-days",
        "1",
        "--earnings-post-days",
        "5",
        "--phase-shift-days",
        "1",
        "--phase-proxy-mode",
        "difficult_v11",
        "--rebuild-interval-trading-days",
        str(args.rebuild_interval_trading_days),
    ]
    if args.batch:
        cmd.append("--batch")
    else:
        cmd.extend(
            [
                "--detail-csv",
                str(args.detail_csv),
                "--selected-csv",
                str(args.selected_csv),
                "--start-date",
                str(args.start_date),
                "--end-date",
                str(args.end_date),
                "--dataset-name",
                str(args.dataset_name),
                "--initial-capital",
                str(args.initial_capital),
            ]
        )
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

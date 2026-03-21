from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run named method: phase_adaptive_practical_v3.1_pre1_post5_etf1489_postcrash_highvol_rebound")
    parser.add_argument("--detail-csv", type=Path, required=True)
    parser.add_argument("--selected-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--initial-capital", type=float, default=3_000_000.0)
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(ROOT / "projects" / "shikiho_text_parser" / "backtest_phase_adaptive_practical_v31_pre1_post5_gold_switch.py"),
        "--earnings-pre-days", "1",
        "--earnings-post-days", "5",
        "--gold-ticker", "1489.T",
        "--gold-phases", "high_vol",
        "--gold-allocation-ratio", "1.0",
        "--gold-signal-name", "rebound_open",
        "--gold-require-recent-phase", "crash",
        "--gold-require-recent-phase-days", "5",
        "--detail-csv", str(args.detail_csv),
        "--selected-csv", str(args.selected_csv),
        "--output-dir", str(args.output_dir),
        "--start-date", args.start_date,
        "--end-date", args.end_date,
        "--dataset-name", args.dataset_name,
        "--initial-capital", str(args.initial_capital),
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

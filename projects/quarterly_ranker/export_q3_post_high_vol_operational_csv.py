from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from projects.quarterly_ranker.operational_csv_utils import write_operational_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="Export an operational-only CSV for the q3 post-high-vol selected table.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("projects/quarterly_ranker/output/q3_pre_analysis_20250630_aligned/threshold_search_post_high_vol/best_selected_candidates.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("projects/quarterly_ranker/output/q3_pre_analysis_20250630_aligned/threshold_search_post_high_vol/operational/best_selected_candidates_operational.csv"),
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    write_operational_csv(df, args.output_csv)
    print(f"[OUT] selected_operational={args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

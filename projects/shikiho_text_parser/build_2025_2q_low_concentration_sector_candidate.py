from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.quarterly_ranker.operational_csv_utils import write_operational_csv
from projects.quarterly_ranker.select_2024_q2_pre_jul_candidates import select_candidates


OUT_DIR = ROOT / "projects" / "quarterly_ranker" / "output" / "2025_2q_low_concentration_sector_candidate"
SOURCE_DETAIL = ROOT / "projects" / "quarterly_ranker" / "output" / "2025_2q_pre_analysis_20250331_candidate" / "pre_shikiho_feature_ranking.csv"
SECTOR_SUMMARY = ROOT / "projects" / "shikiho_text_parser" / "output" / "2025_2q_sector_concentration_candidate" / "sector_low_concentration_candidates.csv"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detail_df = pd.read_csv(SOURCE_DETAIL)
    sector_df = pd.read_csv(SECTOR_SUMMARY)
    selected_sectors = sector_df["sector"].astype(str).head(5).tolist()
    filtered = detail_df[detail_df["sector"].astype(str).isin(selected_sectors)].copy()

    filtered_detail_csv = OUT_DIR / "pre_shikiho_feature_ranking_filtered.csv"
    filtered.to_csv(filtered_detail_csv, index=False, encoding="utf-8-sig")

    selected = select_candidates(filtered)
    selected_csv = OUT_DIR / "pre_selected_candidates.csv"
    selected.to_csv(selected_csv, index=False, encoding="utf-8-sig")

    operational_csv = OUT_DIR / "operational" / "pre_selected_candidates_operational.csv"
    write_operational_csv(selected, operational_csv)

    payload = {
        "selected_sectors": selected_sectors,
        "filtered_universe_count": int(len(filtered)),
        "selected_rows": int(len(selected)),
        "selected_unique_tickers": int(selected["ticker"].astype(str).nunique()) if not selected.empty else 0,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

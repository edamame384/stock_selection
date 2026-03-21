from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.shikiho_text_parser.compare_phase_adaptive_practical_v3 import (  # noqa: E402
    TableSpec,
    run_dataset,
)
from projects.shikiho_text_parser.search_phase_method_optimization import (  # noqa: E402
    PHASE_CSV,
    load_phase_map,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run practical phase-adaptive backtest v3.1 with monthly crash-table refresh."
    )
    parser.add_argument("--detail-csv", type=Path, required=True)
    parser.add_argument("--selected-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--initial-capital", type=float, default=3_000_000.0)
    args = parser.parse_args()

    phase_map = load_phase_map(PHASE_CSV)
    spec = TableSpec(
        name=args.dataset_name,
        detail_csv=args.detail_csv,
        selected_csv=args.selected_csv,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
    )
    result = run_dataset(spec, phase_map, monthly=True)
    payload = {
        "mapping_name": "practical_phase_adaptive_v31",
        "mapping_ja": "実務向け局面切替v3.1",
        "crash_table_name": "post_crash_broad",
        "crash_table_ja": "暴落後拡張",
        "crash_table_refresh": "monthly",
        **result,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.shikiho_text_parser.search_phase_method_optimization import (
    DatasetSpec,
    PHASE_CSV,
    PRICE_DIR,
    load_phase_map,
    prepare_dataset,
    run_backtest,
)


def practical_mapping() -> dict[str, str]:
    return {
        "uptrend": "condition2",
        "normal": "condition2",
        "stable": "breakout_1p5",
        "overheated_range": "q3_post_high_vol",
        "reversal_up": "q3_post_high_vol",
        "capitulation_end": "q3_post_high_vol",
        "high_vol": "q3_post_high_vol",
        "settling": "q3_post_high_vol",
        "surge": "no_trade",
        "downtrend": "no_trade",
        "crash": "no_trade",
        "reversal_down": "no_trade",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run practical phase-adaptive backtest.")
    parser.add_argument("--selected-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--initial-capital", type=float, default=3_000_000.0)
    args = parser.parse_args()

    phase_map = load_phase_map(PHASE_CSV)
    spec = DatasetSpec(
        name=args.dataset_name,
        selected_csv=args.selected_csv,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
    )
    prepared = prepare_dataset(spec, PRICE_DIR, phase_map)
    result = run_backtest(prepared, phase_map, practical_mapping())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps({
        "mapping_name": "practical_phase_adaptive",
        "mapping": practical_mapping(),
        **result,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "mapping_name": "practical_phase_adaptive",
        "mapping": practical_mapping(),
        **result,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

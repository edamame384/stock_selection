from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.shikiho_text_parser import backtest_phase_adaptive_practical_v35_alloc50_candidate as base_mod  # noqa: E402
from projects.shikiho_text_parser import compare_alloc50_post_major_multi_etf_expanded_candidate as v36_mod  # noqa: E402
from projects.shikiho_text_parser.compare_phase_adaptive_practical_v3 import TableSpec  # noqa: E402


PHASE_CSV_2020 = (
    ROOT
    / "projects"
    / "shikiho_text_parser"
    / "output"
    / "nikkei_market_phase_map_2019_2020_candidate"
    / "nikkei_market_phase_daily_labels.csv"
)
OUT_DIR = ROOT / "projects" / "shikiho_text_parser" / "output" / "phase_adaptive_practical_v36_2020_batch"


def build_specs() -> list[TableSpec]:
    return [
        TableSpec(
            "q3",
            ROOT / "projects" / "quarterly_ranker" / "output" / "2020_1q_pre_analysis_20191231_candidate" / "pre_shikiho_feature_ranking.csv",
            ROOT / "projects" / "quarterly_ranker" / "output" / "2020_1q_pre_analysis_20191231_candidate" / "operational" / "pre_selected_candidates_operational.csv",
            "2020-01-01",
            "2020-03-31",
            3_000_000.0,
        ),
        TableSpec(
            "q3",
            ROOT / "projects" / "quarterly_ranker" / "output" / "2020_2q_pre_analysis_20200331_candidate" / "pre_shikiho_feature_ranking.csv",
            ROOT / "projects" / "quarterly_ranker" / "output" / "2020_2q_pre_analysis_20200331_candidate" / "operational" / "pre_selected_candidates_operational.csv",
            "2020-04-01",
            "2020-06-30",
            3_000_000.0,
        ),
    ]


def build_earnings_map(specs: list[TableSpec]) -> dict[str, list[pd.Timestamp]]:
    all_tickers: set[str] = set()
    for spec in specs:
        detail_df = base_mod.normalize_static(pd.read_csv(spec.detail_csv)).drop_duplicates(subset=["ticker"], keep="first")
        end_date = pd.Timestamp(spec.end_date)
        price_map_all = base_mod.load_price_map(detail_df["ticker"].astype(str).tolist(), end_date)
        trade_dates = sorted(
            set().union(
                *[
                    set(df[(df.index >= pd.Timestamp(spec.start_date)) & (df.index <= end_date)].index.tolist())
                    for df in price_map_all.values()
                ]
            )
        )
        if not trade_dates:
            continue
        standard_tables = base_mod.build_interval_standard_tables(spec.name, detail_df, price_map_all, trade_dates, 10)
        crash_tables = base_mod.build_interval_crash_tables(detail_df, price_map_all, trade_dates, 10)
        for tbl in standard_tables.values():
            if not tbl.empty:
                all_tickers.update(tbl["ticker"].astype(str).tolist())
        for tbl in crash_tables.values():
            if not tbl.empty:
                all_tickers.update(tbl["ticker"].astype(str).tolist())

    local_df = base_mod.load_local_disclosure_dates(sorted(all_tickers))
    irbank_df = base_mod.fetch_irbank_earnings_cache(sorted(all_tickers)) if all_tickers else pd.DataFrame(columns=["ticker", "earnings_date", "source"])
    merged = pd.concat(
        [
            local_df,
            irbank_df[["ticker", "earnings_date"]] if not irbank_df.empty else pd.DataFrame(columns=["ticker", "earnings_date"]),
        ],
        ignore_index=True,
    )
    if not merged.empty:
        merged = merged.drop_duplicates(subset=["ticker", "earnings_date"]).sort_values(["ticker", "earnings_date"])
    return base_mod.build_earnings_map(merged)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["2020_1q", "2020_2q", "all"], default="all")
    args = parser.parse_args()

    specs = build_specs()
    labels = ["2020_1q", "2020_2q"]
    if args.dataset != "all":
        idx = labels.index(args.dataset)
        specs = [specs[idx]]
        labels = [labels[idx]]
    phase_src = pd.read_csv(PHASE_CSV_2020, skiprows=[1]).rename(columns={"phase": "phase_name"})
    phase_src["Date"] = pd.to_datetime(phase_src["Date"])
    phase_df = v36_mod.build_post_major_state(phase_src)
    earnings_map = build_earnings_map(specs)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    old_phase_csv = base_mod.PHASE_CSV
    try:
        base_mod.PHASE_CSV = PHASE_CSV_2020
        for label, spec in zip(labels, specs):
            result = v36_mod.run_dataset(spec, phase_df, earnings_map)
            out_dir = OUT_DIR / label
            out_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "mapping_name": "phase_adaptive_practical_v36",
                "mapping_ja": "実務向け局面切替v3.6_50万分割_post_major_crash価格集中ETF拡張",
                **{k: v for k, v in result.items() if k not in {"trade_log", "equity_curve", "state_daily"}},
            }
            (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            result["trade_log"].to_csv(out_dir / "trade_log.csv", index=False, encoding="utf-8-sig")
            result["equity_curve"].to_csv(out_dir / "equity_curve.csv", index=False, encoding="utf-8-sig")
            result["state_daily"].to_csv(out_dir / "state_daily.csv", index=False, encoding="utf-8-sig")
            rows.append(
                {
                    "dataset": label,
                    "start_date": spec.start_date,
                    "end_date": spec.end_date,
                    "total_return_pct": payload["total_return_pct"],
                    "final_capital": payload["final_capital"],
                    "num_buys": payload["num_buys"],
                    "max_drawdown_pct": payload["max_drawdown_pct"],
                }
            )
    finally:
        base_mod.PHASE_CSV = old_phase_csv

    pd.DataFrame(rows).to_csv(OUT_DIR / "summary_all.csv", index=False, encoding="utf-8-sig")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

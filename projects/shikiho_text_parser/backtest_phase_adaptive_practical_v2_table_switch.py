from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.shikiho_text_parser.backtest_phase_adaptive_practical_v2 import (  # noqa: E402
    practical_v2_mapping,
)
from projects.shikiho_text_parser.search_phase_method_optimization import (  # noqa: E402
    DatasetSpec,
    PHASE_CSV,
    PRICE_DIR,
    load_phase_map,
    prepare_dataset,
)
from projects.shikiho_text_parser.backtest_phase_adaptive import eval_signal  # noqa: E402
from projects.shikiho_text_parser.search_post_crash_switch_origcount import build_table  # noqa: E402


@dataclass
class TableSpec:
    name: str
    detail_csv: Path
    selected_csv: Path
    start_date: str
    end_date: str
    initial_capital: float = 3_000_000.0


def load_standard_and_high_vol_tables(detail_csv: Path, selected_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_df = pd.read_csv(detail_csv)
    selected_df = pd.read_csv(selected_csv)
    if "ocr_per" not in detail_df.columns:
        if "forecast_per" in detail_df.columns:
            detail_df["ocr_per"] = detail_df["forecast_per"]
        else:
            detail_df["ocr_per"] = pd.NA
    if "avg_monthly_high_low_change_pct" not in detail_df.columns:
        detail_df["avg_monthly_high_low_change_pct"] = pd.NA
    if "end_to_trailing_high_pct" not in detail_df.columns:
        detail_df["end_to_trailing_high_pct"] = pd.NA
    standard_tickers = set(selected_df["ticker"].astype(str))
    standard_table = detail_df[detail_df["ticker"].astype(str).isin(standard_tickers)].copy()
    standard_table = standard_table.drop_duplicates(subset=["ticker"], keep="first").reset_index(drop=True)
    original_count = int(selected_df["ticker"].astype(str).nunique())
    high_vol_table = build_table(detail_df, "defensive_origcount", original_count)
    return ensure_v2_compatible(standard_table), ensure_v2_compatible(high_vol_table)


def ensure_v2_compatible(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "company_name" not in out.columns:
        out["company_name"] = ""
    if "simple_sector" not in out.columns and "sector" in out.columns:
        out["simple_sector"] = out["sector"]
    if "sector" not in out.columns and "simple_sector" in out.columns:
        out["sector"] = out["simple_sector"]
    if "headline_score_raw" not in out.columns:
        out["headline_score_raw"] = 0.0
    if "shikiho_score_overall" not in out.columns:
        out["shikiho_score_overall"] = 3.0
    if "forecast_per" not in out.columns:
        if "ocr_per" in out.columns:
            out["forecast_per"] = pd.to_numeric(out["ocr_per"], errors="coerce")
        else:
            out["forecast_per"] = pd.NA
    if "promising_score" not in out.columns:
        if "score" in out.columns:
            out["promising_score"] = pd.to_numeric(out["score"], errors="coerce")
        elif "overall_score" in out.columns:
            out["promising_score"] = pd.to_numeric(out["overall_score"], errors="coerce")
        else:
            out["promising_score"] = 0.0
    if "sector_adjusted_per_score" not in out.columns:
        out["sector_adjusted_per_score"] = pd.NA
    if "max_drawdown_pct" in out.columns:
        out["max_drawdown_pct"] = -pd.to_numeric(out["max_drawdown_pct"], errors="coerce").abs()
    return out


def run_backtest(prepared_standard: dict, prepared_high_vol: dict, phase_map: pd.Series) -> dict:
    spec: DatasetSpec = prepared_standard["spec"]
    price_map = prepared_standard["price_map"]
    dates = prepared_standard["dates"]
    if not dates:
        return {
            "dataset": spec.name,
            "final_capital": spec.initial_capital,
            "total_return_pct": 0.0,
            "num_buys": 0,
            "phase_pnl": {},
            "max_drawdown_pct": 0.0,
        }

    mapping = practical_v2_mapping()
    std_static = prepared_standard["static_lookup"]
    hv_static = prepared_high_vol["static_lookup"]
    std_metrics = prepared_standard["metrics_cache"]
    hv_metrics = prepared_high_vol["metrics_cache"]

    cash = float(spec.initial_capital)
    positions: dict[str, dict] = {}
    prev_signal = {ticker: False for ticker in price_map}
    equity_rows = []
    buy_count = 0

    for date in dates:
        phase = phase_map.loc[phase_map.index <= date].tail(1)
        phase_name = str(phase.iloc[0]) if not phase.empty else "normal"
        use_high_vol_table = phase_name == "high_vol"
        rule_name = mapping.get(phase_name, "condition2")

        active_static = hv_static if use_high_vol_table else std_static
        active_metrics = hv_metrics if use_high_vol_table else std_metrics
        active_tickers = active_static.keys()

        signal_today: dict[str, bool] = {}
        score_today: dict[str, float] = {}
        basis_date_today: dict[str, pd.Timestamp] = {}

        for ticker in price_map:
            if ticker not in active_tickers:
                signal_today[ticker] = False
                continue
            metrics = active_metrics.get(ticker, {}).get(date, {})
            if not metrics:
                signal_today[ticker] = prev_signal.get(ticker, False) if date not in price_map[ticker].index else False
                continue
            sig, signal_score = eval_signal(rule_name, metrics, active_static[ticker])
            signal_today[ticker] = sig
            score_today[ticker] = signal_score
            basis_date_today[ticker] = metrics["signal_basis_date"]

        for ticker in list(positions.keys()):
            df = price_map[ticker]
            if date not in df.index:
                continue
            day = df.loc[date]
            close_price = float(day["Close"])
            low_price = float(day["Low"]) if "Low" in day.index and not pd.isna(day["Low"]) else close_price
            open_price = float(day["Open"]) if "Open" in day.index and not pd.isna(day["Open"]) else close_price
            entry_price = positions[ticker]["entry_price"]
            ret = close_price / entry_price - 1.0

            exit_reason = None
            exit_price = None
            if phase_name == "crash":
                prev_idx = df.index[df.index < date]
                if len(prev_idx):
                    prev_date = prev_idx[-1]
                    prev_close = float(df.loc[prev_date, "Close"])
                    early_trigger = prev_close * 0.98
                    if low_price <= early_trigger:
                        exit_reason = "crash_early_exit"
                        exit_price = min(open_price, early_trigger)

            if exit_reason is None and ret >= 0.08:
                exit_reason = "take_profit"
                exit_price = close_price
            elif exit_reason is None and ret <= -0.05:
                exit_reason = "stop_loss"
                exit_price = close_price
            elif exit_reason is None and not signal_today.get(ticker, False):
                exit_reason = "sell_signal"
                exit_price = close_price

            if exit_reason is not None:
                cash += positions[ticker]["shares"] * float(exit_price)
                del positions[ticker]

        candidates = []
        for ticker in price_map:
            if ticker in positions:
                continue
            if signal_today.get(ticker, False) and not prev_signal.get(ticker, False):
                candidates.append((ticker, score_today.get(ticker, 0.0)))
        candidates.sort(key=lambda x: x[1], reverse=True)

        remaining = len(candidates)
        for ticker, _score in candidates:
            df = price_map[ticker]
            day = df.loc[date]
            prev_close = float(df.loc[basis_date_today[ticker], "Close"])
            trigger_price = prev_close * 1.015
            day_open = float(day["Open"]) if "Open" in day.index and not pd.isna(day["Open"]) else float(day["Close"])
            day_high = float(day["High"]) if "High" in day.index and not pd.isna(day["High"]) else float(day["Close"])
            if day_high < trigger_price:
                remaining -= 1
                continue
            fill_price = max(day_open, trigger_price)
            alloc = cash / remaining if remaining > 0 else 0.0
            lot_cost = fill_price * 100.0
            shares = int(alloc // lot_cost) * 100
            if shares >= 100 and shares * fill_price <= cash:
                cash -= shares * fill_price
                positions[ticker] = {"shares": shares, "entry_price": fill_price}
                buy_count += 1
            remaining -= 1

        market_value = 0.0
        for ticker, pos in positions.items():
            df = price_map[ticker]
            usable_idx = df.index[df.index <= date]
            if len(usable_idx):
                market_value += pos["shares"] * float(df.loc[usable_idx[-1], "Close"])
        equity_rows.append({"date": date, "nikkei_phase": phase_name, "equity": cash + market_value})
        prev_signal = signal_today

    latest_date = max(dates)
    for ticker in list(positions.keys()):
        df = price_map[ticker]
        usable_idx = df.index[df.index <= latest_date]
        if len(usable_idx):
            cash += positions[ticker]["shares"] * float(df.loc[usable_idx[-1], "Close"])
        del positions[ticker]

    equity_df = pd.DataFrame(equity_rows)
    if not equity_df.empty:
        equity_df["pnl_day"] = equity_df["equity"].diff().fillna(equity_df["equity"] - spec.initial_capital)
        phase_pnl = equity_df.groupby("nikkei_phase")["pnl_day"].sum().to_dict()
        drawdown = equity_df["equity"].astype(float) / equity_df["equity"].astype(float).cummax() - 1.0
        max_drawdown_pct = float(drawdown.min()) * 100.0
    else:
        phase_pnl = {}
        max_drawdown_pct = 0.0

    return {
        "dataset": spec.name,
        "final_capital": float(cash),
        "total_return_pct": (float(cash) / float(spec.initial_capital) - 1.0) * 100.0,
        "num_buys": int(buy_count),
        "phase_pnl": phase_pnl,
        "max_drawdown_pct": max_drawdown_pct,
    }


def run_for_spec(table_spec: TableSpec, output_dir: Path) -> dict:
    phase_map = load_phase_map(PHASE_CSV)
    standard_table, high_vol_table = load_standard_and_high_vol_tables(table_spec.detail_csv, table_spec.selected_csv)

    std_csv = output_dir / "standard_table.csv"
    hv_csv = output_dir / "high_vol_table.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    standard_table.to_csv(std_csv, index=False, encoding="utf-8-sig")
    high_vol_table.to_csv(hv_csv, index=False, encoding="utf-8-sig")

    std_spec = DatasetSpec(
        name=table_spec.name,
        selected_csv=std_csv,
        start_date=table_spec.start_date,
        end_date=table_spec.end_date,
        initial_capital=table_spec.initial_capital,
    )
    hv_spec = DatasetSpec(
        name=table_spec.name,
        selected_csv=hv_csv,
        start_date=table_spec.start_date,
        end_date=table_spec.end_date,
        initial_capital=table_spec.initial_capital,
    )

    prepared_standard = prepare_dataset(std_spec, PRICE_DIR, phase_map)
    prepared_high_vol = prepare_dataset(hv_spec, PRICE_DIR, phase_map)
    result = run_backtest(prepared_standard, prepared_high_vol, phase_map)
    payload = {
        "mapping_name": "practical_phase_adaptive_v2_table_switch",
        "mapping_ja": "実務向け局面切替v2_テーブル切替",
        "table_switch": {
            "high_vol": "defensive_origcount",
            "other": "standard_table",
        },
        "standard_count": int(len(standard_table)),
        "high_vol_count": int(len(high_vol_table)),
        **result,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    out_root = ROOT / "projects" / "shikiho_text_parser" / "output" / "phase_adaptive_practical_v2_table_switch"
    specs = [
        TableSpec(
            "q2_2024",
            ROOT / "projects" / "quarterly_ranker" / "output" / "q2_2024_pre_analysis_20240630_aligned" / "q2_2024_pre_shikiho_feature_ranking.csv",
            ROOT / "projects" / "quarterly_ranker" / "output" / "q2_2024_pre_analysis_20240630_aligned" / "q2_2024_pre_selected_candidates.csv",
            "2024-07-01",
            "2024-09-30",
        ),
        TableSpec(
            "q3",
            ROOT / "projects" / "quarterly_ranker" / "output" / "q3_pre_analysis_20250630_aligned" / "q3_pre_shikiho_feature_ranking.csv",
            ROOT / "projects" / "quarterly_ranker" / "output" / "q3_pre_analysis_20250630_aligned" / "q3_pre_selected_candidates.csv",
            "2025-07-01",
            "2025-09-30",
        ),
        TableSpec(
            "q4",
            ROOT / "projects" / "quarterly_ranker" / "output" / "q4_pre_analysis_20250930_full" / "q4_pre_shikiho_feature_ranking.csv",
            ROOT / "projects" / "quarterly_ranker" / "output" / "q4_pre_analysis_20250930_full" / "q4_pre_selected_candidates.csv",
            "2025-10-01",
            "2025-12-31",
        ),
        TableSpec(
            "4q2",
            ROOT / "projects" / "shikiho_text_parser" / "output" / "4q2_selection" / "4q2_scored_universe.csv",
            ROOT / "projects" / "shikiho_text_parser" / "output" / "4q2_selection" / "4q2_selected_candidates.csv",
            "2026-01-01",
            "2026-03-10",
        ),
    ]

    rows = []
    for spec in specs:
        rows.append(run_for_spec(spec, out_root / spec.name))
    results_df = pd.DataFrame(rows)
    results_df.to_csv(out_root / "summary_all.csv", index=False, encoding="utf-8-sig")
    print(results_df.to_string(index=False))
    print(f"[OUT] {out_root / 'summary_all.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

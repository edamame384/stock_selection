from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.shikiho_text_parser.search_phase_method_optimization import (
    DatasetSpec,
    PHASE_CSV,
    PRICE_DIR,
    load_phase_map,
    prepare_dataset,
)
from projects.shikiho_text_parser.backtest_phase_adaptive import eval_signal

OUT_DIR = ROOT / "projects" / "shikiho_text_parser" / "output" / "crash_exit_rule_compare"


PRACTICAL_MAP = {
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


def run_backtest(prepared: dict, phase_map: pd.Series, crash_mode: str) -> dict:
    spec: DatasetSpec = prepared["spec"]
    static_lookup = prepared["static_lookup"]
    price_map = prepared["price_map"]
    dates = prepared["dates"]
    metrics_cache = prepared["metrics_cache"]
    if not dates:
        return {"dataset": spec.name, "final_capital": spec.initial_capital, "total_return_pct": 0.0, "num_buys": 0}

    cash = float(spec.initial_capital)
    positions: dict[str, dict] = {}
    prev_signal = {ticker: False for ticker in price_map}
    equity_rows = []
    buy_count = 0

    for date in dates:
        phase = phase_map.loc[phase_map.index <= date].tail(1)
        phase_name = str(phase.iloc[0]) if not phase.empty else "normal"
        rule_name = PRACTICAL_MAP.get(phase_name, "condition2")

        signal_today: dict[str, bool] = {}
        score_today: dict[str, float] = {}
        basis_date_today: dict[str, pd.Timestamp] = {}

        for ticker in price_map:
            metrics = metrics_cache.get(ticker, {}).get(date, {})
            if not metrics:
                signal_today[ticker] = prev_signal.get(ticker, False) if date not in price_map[ticker].index else False
                continue
            sig, signal_score = eval_signal(rule_name, metrics, static_lookup[ticker])
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
                if crash_mode == "stop_5":
                    if ret <= -0.05:
                        exit_reason = "crash_stop_5"
                        exit_price = close_price
                elif crash_mode == "stop_8":
                    if ret <= -0.08:
                        exit_reason = "crash_stop_8"
                        exit_price = close_price
                elif crash_mode == "early":
                    prev_idx = df.index[df.index < date]
                    if len(prev_idx):
                        prev_date = prev_idx[-1]
                        prev_close = float(df.loc[prev_date, "Close"])
                        early_trigger = prev_close * 0.98
                        if low_price <= early_trigger:
                            exit_reason = "crash_early_exit"
                            exit_price = min(open_price, early_trigger)
                else:
                    raise ValueError(crash_mode)

            if exit_reason is None and ret >= 0.08:
                exit_reason = "take_profit"
                exit_price = close_price
            elif exit_reason is None and phase_name != "crash" and ret <= -0.05:
                exit_reason = "stop_loss"
                exit_price = close_price
            elif exit_reason is None and not signal_today.get(ticker, False):
                exit_reason = "sell_signal"
                exit_price = close_price

            if exit_reason is not None:
                shares = positions[ticker]["shares"]
                cash += shares * float(exit_price)
                del positions[ticker]

        candidates = []
        for ticker in price_map:
            if ticker in positions:
                continue
            if signal_today.get(ticker, False) and not prev_signal.get(ticker, False):
                candidates.append((ticker, score_today.get(ticker, 0.0)))
        candidates.sort(key=lambda x: x[1], reverse=True)

        remaining = len(candidates)
        for ticker, signal_score in candidates:
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
        equity_rows.append({"date": date, "phase": phase_name, "equity": cash + market_value})
        prev_signal = signal_today

    latest_date = max(dates)
    for ticker in list(positions.keys()):
        df = price_map[ticker]
        usable_idx = df.index[df.index <= latest_date]
        if len(usable_idx):
            cash += positions[ticker]["shares"] * float(df.loc[usable_idx[-1], "Close"])
        del positions[ticker]

    equity_df = pd.DataFrame(equity_rows)
    max_dd_pct = 0.0
    if not equity_df.empty:
        dd = equity_df["equity"].astype(float) / equity_df["equity"].astype(float).cummax() - 1.0
        max_dd_pct = float(dd.min()) * 100.0
    return {
        "dataset": spec.name,
        "crash_mode": crash_mode,
        "final_capital": float(cash),
        "total_return_pct": (float(cash) / float(spec.initial_capital) - 1.0) * 100.0,
        "num_buys": int(buy_count),
        "max_drawdown_pct": max_dd_pct,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase_map = load_phase_map(PHASE_CSV)
    specs = [
        DatasetSpec(
            name="q3",
            selected_csv=ROOT / "projects" / "quarterly_ranker" / "output" / "q3_pre_analysis_20250630_aligned" / "threshold_search_post_high_vol" / "best_selected_candidates_condition2_input.csv",
            start_date="2025-07-01",
            end_date="2025-09-30",
        ),
        DatasetSpec(
            name="q4",
            selected_csv=ROOT / "projects" / "quarterly_ranker" / "output" / "q4_pre_analysis_20250930_full" / "q4_pre_selected_candidates_condition2_input.csv",
            start_date="2025-10-01",
            end_date="2025-12-31",
        ),
        DatasetSpec(
            name="4q2",
            selected_csv=ROOT / "projects" / "shikiho_text_parser" / "output" / "4q2_selection" / "4q2_selected_candidates.csv",
            start_date="2026-01-01",
            end_date="2026-03-10",
        ),
    ]
    prepared_sets = [prepare_dataset(spec, PRICE_DIR, phase_map) for spec in specs]
    modes = ["stop_5", "stop_8", "early"]
    rows = []
    for mode in modes:
        for prepared in prepared_sets:
            rows.append(run_backtest(prepared, phase_map, mode))

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("crash_mode", dropna=False)
        .agg(
            total_final_capital=("final_capital", "sum"),
            total_return_pct=("total_return_pct", "sum"),
            avg_return_pct=("total_return_pct", "mean"),
            avg_max_drawdown_pct=("max_drawdown_pct", "mean"),
        )
        .reset_index()
        .sort_values("total_final_capital", ascending=False)
    )
    df.to_csv(OUT_DIR / "crash_exit_rule_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "crash_exit_rule_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

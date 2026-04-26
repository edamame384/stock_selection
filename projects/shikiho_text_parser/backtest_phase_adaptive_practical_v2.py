from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.shikiho_text_parser.backtest_phase_adaptive import eval_signal
from projects.shikiho_text_parser.search_phase_method_optimization import (
    DatasetSpec,
    PHASE_CSV,
    PRICE_DIR,
    load_phase_map,
    prepare_dataset,
)


def practical_v2_mapping() -> dict[str, str]:
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


def run_backtest_with_crash_early_exit(prepared: dict, phase_map: pd.Series, mapping: dict[str, str]) -> dict:
    spec: DatasetSpec = prepared["spec"]
    static_lookup = prepared["static_lookup"]
    price_map = prepared["price_map"]
    dates = prepared["dates"]
    metrics_cache = prepared["metrics_cache"]
    if not dates:
        return {
            "dataset": spec.name,
            "final_capital": spec.initial_capital,
            "total_return_pct": 0.0,
            "num_buys": 0,
            "phase_pnl": {},
            "max_drawdown_pct": 0.0,
            "crash_exit_mode": "early",
        }

    cash = float(spec.initial_capital)
    positions: dict[str, dict] = {}
    prev_signal = {ticker: False for ticker in price_map}
    equity_rows = []
    buy_count = 0

    for date in dates:
        phase = phase_map.loc[phase_map.index <= date].tail(1)
        phase_name = str(phase.iloc[0]) if not phase.empty else "normal"
        rule_name = mapping.get(phase_name, "condition2")

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
        for ticker, _ in candidates:
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
        "crash_exit_mode": "early",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run practical phase-adaptive backtest v2 with crash early exit.")
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
    result = run_backtest_with_crash_early_exit(prepared, phase_map, practical_v2_mapping())
    payload = {
        "mapping_name": "practical_phase_adaptive_v2",
        "mapping_ja": "実務向け局面切替v2",
        "mapping": practical_v2_mapping(),
        **result,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

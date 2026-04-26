from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.shikiho_text_parser.backtest_phase_adaptive import (
    PHASE_CSV,
    PRICE_DIR,
    compute_metrics,
    eval_signal,
    load_phase_map,
    load_selected,
)

OUT_DIR = ROOT / "projects" / "shikiho_text_parser" / "output" / "phase_method_optimization"


@dataclass
class DatasetSpec:
    name: str
    selected_csv: Path
    start_date: str
    end_date: str
    initial_capital: float = 3_000_000.0


def baseline_mapping() -> dict[str, str]:
    return {
        "uptrend": "condition2",
        "normal": "condition2",
        "stable": "breakout_1p5",
        "overheated_range": "breakout_1p5",
        "reversal_up": "q3_post_high_vol",
        "capitulation_end": "q3_post_high_vol",
        "surge": "no_trade",
        "downtrend": "no_trade",
        "crash": "no_trade",
        "high_vol": "no_trade",
        "settling": "no_trade",
        "reversal_down": "no_trade",
    }


def default_method_for_unknown() -> str:
    return "condition2"


def price_path_for(price_dir: Path, ticker: str) -> Path:
    return price_dir / f"{ticker.replace('.', '_')}.csv"


def prepare_dataset(spec: DatasetSpec, price_dir: Path, phase_map: pd.Series) -> dict:
    selected = load_selected(spec.selected_csv)
    static_lookup = selected.set_index("ticker").to_dict("index")
    start_date = pd.Timestamp(spec.start_date)
    end_date = pd.Timestamp(spec.end_date)

    price_map: dict[str, pd.DataFrame] = {}
    all_dates: set[pd.Timestamp] = set()
    for ticker in selected["ticker"]:
        path = price_path_for(price_dir, ticker)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "Date" not in df.columns or "Close" not in df.columns:
            continue
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").set_index("Date")
        df = df[(df.index >= pd.Timestamp("2024-01-01")) & (df.index <= end_date)].copy()
        if df.empty:
            continue
        price_map[ticker] = df
        all_dates.update(df[(df.index >= start_date) & (df.index <= end_date)].index.tolist())

    dates = sorted(all_dates)
    metrics_cache: dict[str, dict[pd.Timestamp, dict]] = {}
    for ticker, df in price_map.items():
        cache_for_ticker: dict[pd.Timestamp, dict] = {}
        for date in dates:
            if date not in df.index:
                continue
            prev_idx = df.index[df.index < date]
            if len(prev_idx) == 0:
                continue
            signal_date = prev_idx[-1]
            metrics = compute_metrics(df, signal_date)
            if metrics:
                metrics["signal_basis_date"] = signal_date
            cache_for_ticker[date] = metrics
        metrics_cache[ticker] = cache_for_ticker

    observed_phases = set()
    for date in dates:
        phase = phase_map.loc[phase_map.index <= date].tail(1)
        phase_name = str(phase.iloc[0]) if not phase.empty else "normal"
        observed_phases.add(phase_name)

    return {
        "spec": spec,
        "selected": selected,
        "static_lookup": static_lookup,
        "price_map": price_map,
        "dates": dates,
        "metrics_cache": metrics_cache,
        "observed_phases": observed_phases,
    }


def run_backtest(prepared: dict, phase_map: pd.Series, mapping: dict[str, str]) -> dict:
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
        }

    cash = float(spec.initial_capital)
    positions: dict[str, dict] = {}
    prev_signal = {ticker: False for ticker in price_map}
    equity_rows = []
    buy_count = 0

    for date in dates:
        phase = phase_map.loc[phase_map.index <= date].tail(1)
        phase_name = str(phase.iloc[0]) if not phase.empty else "normal"
        rule_name = mapping.get(phase_name, default_method_for_unknown())

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
            price = float(df.loc[date, "Close"])
            ret = price / positions[ticker]["entry_price"] - 1.0
            exit_now = False
            if ret >= 0.08 or ret <= -0.05 or not signal_today.get(ticker, False):
                exit_now = True
            if exit_now:
                shares = positions[ticker]["shares"]
                cash += shares * price
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
        equity_rows.append(
            {
                "date": date,
                "nikkei_phase": phase_name,
                "equity": cash + market_value,
            }
        )
        prev_signal = signal_today

    latest_date = max(dates)
    for ticker in list(positions.keys()):
        df = price_map[ticker]
        usable_idx = df.index[df.index <= latest_date]
        if len(usable_idx):
            price = float(df.loc[usable_idx[-1], "Close"])
            cash += positions[ticker]["shares"] * price
        del positions[ticker]

    equity_df = pd.DataFrame(equity_rows)
    if not equity_df.empty:
        equity_df["pnl_day"] = equity_df["equity"].diff().fillna(equity_df["equity"] - spec.initial_capital)
        phase_pnl = (
            equity_df.groupby("nikkei_phase")["pnl_day"].sum().to_dict()
        )
    else:
        phase_pnl = {}

    return {
        "dataset": spec.name,
        "final_capital": float(cash),
        "total_return_pct": (float(cash) / float(spec.initial_capital) - 1.0) * 100.0,
        "num_buys": int(buy_count),
        "phase_pnl": phase_pnl,
    }


def aggregate_results(results: list[dict]) -> dict:
    total_final = sum(r["final_capital"] for r in results)
    total_initial = 3_000_000.0 * len(results)
    total_pnl = total_final - total_initial
    return {
        "objective_total_pnl": total_pnl,
        "objective_total_return_pct": (total_final / total_initial - 1.0) * 100.0,
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
    observed_phases = sorted(set().union(*[p["observed_phases"] for p in prepared_sets]))
    methods = ["condition2", "breakout_1p5", "q3_post_high_vol", "no_trade"]

    current_map = baseline_mapping()
    baseline_results = [run_backtest(p, phase_map, current_map) for p in prepared_sets]
    baseline_obj = aggregate_results(baseline_results)

    iteration_rows = []
    all_trial_rows = []
    improved = True
    step = 0
    while improved:
        improved = False
        best_candidate = None
        best_candidate_obj = None
        step += 1
        for phase in observed_phases:
            for method in methods:
                trial_map = current_map.copy()
                trial_map[phase] = method
                results = [run_backtest(p, phase_map, trial_map) for p in prepared_sets]
                agg = aggregate_results(results)
                row = {"step": step, "phase": phase, "method": method, **agg}
                for r in results:
                    row[f"{r['dataset']}_final_capital"] = r["final_capital"]
                    row[f"{r['dataset']}_return_pct"] = r["total_return_pct"]
                    row[f"{r['dataset']}_num_buys"] = r["num_buys"]
                all_trial_rows.append(row)
                if best_candidate_obj is None or agg["objective_total_pnl"] > best_candidate_obj["objective_total_pnl"]:
                    best_candidate = (phase, method, results)
                    best_candidate_obj = agg
        if best_candidate_obj and best_candidate_obj["objective_total_pnl"] > baseline_obj["objective_total_pnl"] + 1e-9:
            phase, method, results = best_candidate
            current_map[phase] = method
            baseline_results = results
            baseline_obj = best_candidate_obj
            iteration_rows.append(
                {
                    "step": step,
                    "applied_phase": phase,
                    "applied_method": method,
                    **baseline_obj,
                }
            )
            improved = True

    stop_phases = [p for p, m in baseline_mapping().items() if m == "no_trade" and p in observed_phases]
    stop_phase_rows = []
    for phase in stop_phases:
        base_method = baseline_mapping()[phase]
        for method in methods:
            trial_map = current_map.copy()
            trial_map[phase] = method
            results = [run_backtest(p, phase_map, trial_map) for p in prepared_sets]
            agg = aggregate_results(results)
            stop_phase_rows.append(
                {
                    "phase": phase,
                    "baseline_method": base_method,
                    "trial_method": method,
                    **agg,
                    "delta_vs_best_pnl": agg["objective_total_pnl"] - baseline_obj["objective_total_pnl"],
                }
            )

    best_phase_pnl_rows = []
    for r in baseline_results:
        for phase, pnl in r["phase_pnl"].items():
            best_phase_pnl_rows.append({"dataset": r["dataset"], "phase": phase, "pnl": pnl})

    pd.DataFrame(all_trial_rows).to_csv(OUT_DIR / "phase_method_search_trials.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(iteration_rows).to_csv(OUT_DIR / "phase_method_search_iterations.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(stop_phase_rows).to_csv(OUT_DIR / "stop_phase_method_check.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(best_phase_pnl_rows).to_csv(OUT_DIR / "best_mapping_phase_pnl.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "best_mapping.json").write_text(json.dumps(current_map, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "best_objective.json").write_text(json.dumps(baseline_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "observed_phases": observed_phases,
        "baseline_mapping": baseline_mapping(),
        "best_mapping": current_map,
        "baseline_objective": aggregate_results([run_backtest(p, phase_map, baseline_mapping()) for p in prepared_sets]),
        "best_objective": baseline_obj,
        "datasets": baseline_results,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

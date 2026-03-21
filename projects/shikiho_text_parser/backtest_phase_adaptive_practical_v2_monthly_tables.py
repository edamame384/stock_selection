from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.shikiho_text_parser.backtest_phase_adaptive import compute_metrics, eval_signal  # noqa: E402
from projects.shikiho_text_parser.backtest_phase_adaptive_practical_v2 import practical_v2_mapping  # noqa: E402
from projects.shikiho_text_parser.search_phase_method_optimization import PHASE_CSV, PRICE_DIR, load_phase_map  # noqa: E402
from projects.shikiho_text_parser.search_post_crash_switch_origcount import build_table  # noqa: E402


INTERMEDIATE_PHASES = {"settling", "reversal_up", "capitulation_end"}
HIGH_VOL_PHASES = {"high_vol", "crash"}


@dataclass
class TableSpec:
    name: str
    detail_csv: Path
    selected_csv: Path
    start_date: str
    end_date: str
    initial_capital: float = 3_000_000.0


def normalize_static(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
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
    if "ocr_per" not in out.columns:
        out["ocr_per"] = out["forecast_per"] if "forecast_per" in out.columns else pd.NA
    if "avg_monthly_high_low_change_pct" not in out.columns:
        out["avg_monthly_high_low_change_pct"] = pd.NA
    if "end_to_trailing_high_pct" not in out.columns:
        out["end_to_trailing_high_pct"] = pd.NA
    return out


def load_price_map(tickers: list[str], end_date: pd.Timestamp) -> dict[str, pd.DataFrame]:
    price_map: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        path = PRICE_DIR / f"{ticker.replace('.', '_')}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "Date" not in df.columns or "Close" not in df.columns:
            continue
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").set_index("Date")
        df = df[(df.index >= pd.Timestamp("2023-01-01")) & (df.index <= end_date)].copy()
        if not df.empty:
            price_map[ticker] = df
    return price_map


def build_snapshot(detail_df: pd.DataFrame, price_map: dict[str, pd.DataFrame], basis_date: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for row in detail_df.to_dict("records"):
        ticker = row["ticker"]
        df = price_map.get(ticker)
        if df is None:
            continue
        usable_idx = df.index[df.index <= basis_date]
        if len(usable_idx) == 0:
            continue
        effective_basis = usable_idx[-1]
        metrics = compute_metrics(df, effective_basis)
        if not metrics:
            continue
        close = df.loc[:effective_basis, "Close"].astype(float)
        row = dict(row)
        row.update(metrics)
        row["max_drawdown_pct"] = abs(float(metrics["max_drawdown_pct"]))
        row["end_to_trailing_high_pct"] = float(close.iloc[-1] / close.max() * 100.0) if len(close) else pd.NA
        rows.append(row)
    snap = pd.DataFrame(rows)
    if snap.empty:
        return snap
    return normalize_static(snap)


def build_monthly_tables(detail_df: pd.DataFrame, selected_count: int, price_map: dict[str, pd.DataFrame], trade_dates: list[pd.Timestamp]) -> dict[pd.Timestamp, dict[str, pd.DataFrame]]:
    month_starts = sorted({pd.Timestamp(d.year, d.month, 1) for d in trade_dates})
    monthly: dict[pd.Timestamp, dict[str, pd.DataFrame]] = {}
    for month_start in month_starts:
        prior_trade_dates = [d for d in trade_dates if d < month_start]
        basis_date = prior_trade_dates[-1] if prior_trade_dates else max([d for d in trade_dates if d.month == month_start.month and d.year == month_start.year][0] - pd.Timedelta(days=1), trade_dates[0] - pd.Timedelta(days=1))
        snapshot = build_snapshot(detail_df, price_map, basis_date)
        if snapshot.empty:
            monthly[month_start] = {"standard": snapshot, "intermediate": snapshot, "high_vol": snapshot}
            continue
        monthly[month_start] = {
            "standard": build_table(snapshot, "regular_origcount", selected_count),
            "intermediate": build_table(snapshot, "post_high_vol_origcount", selected_count),
            "high_vol": build_table(snapshot, "defensive_origcount", selected_count),
        }
    return monthly


def table_key_for_phase(phase_name: str) -> str:
    if phase_name in HIGH_VOL_PHASES:
        return "high_vol"
    if phase_name in INTERMEDIATE_PHASES:
        return "intermediate"
    return "standard"


def prepare_metric_cache(price_map: dict[str, pd.DataFrame], dates: list[pd.Timestamp]) -> dict[str, dict[pd.Timestamp, dict]]:
    metrics_cache: dict[str, dict[pd.Timestamp, dict]] = {}
    for ticker, df in price_map.items():
        ticker_cache: dict[pd.Timestamp, dict] = {}
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
            ticker_cache[date] = metrics
        metrics_cache[ticker] = ticker_cache
    return metrics_cache


def run_backtest(spec: TableSpec, phase_map: pd.Series, output_dir: Path) -> dict:
    detail_df = normalize_static(pd.read_csv(spec.detail_csv)).drop_duplicates(subset=["ticker"], keep="first")
    selected_df = pd.read_csv(spec.selected_csv)
    selected_count = int(selected_df["ticker"].astype(str).nunique())
    end_date = pd.Timestamp(spec.end_date)
    start_date = pd.Timestamp(spec.start_date)

    full_price_map = load_price_map(detail_df["ticker"].astype(str).tolist(), end_date)
    trade_dates = sorted(set().union(*[set(df[(df.index >= start_date) & (df.index <= end_date)].index.tolist()) for df in full_price_map.values()]))
    monthly_tables = build_monthly_tables(detail_df, selected_count, full_price_map, trade_dates)

    union_tickers = set()
    for tables in monthly_tables.values():
        for tbl in tables.values():
            if not tbl.empty:
                union_tickers.update(tbl["ticker"].astype(str).tolist())

    price_map = {ticker: df for ticker, df in full_price_map.items() if ticker in union_tickers}
    dates = sorted(set().union(*[set(df[(df.index >= start_date) & (df.index <= end_date)].index.tolist()) for df in price_map.values()]))
    metrics_cache = prepare_metric_cache(price_map, dates)
    mapping = practical_v2_mapping()

    cash = float(spec.initial_capital)
    positions: dict[str, dict] = {}
    prev_signal = {ticker: False for ticker in price_map}
    equity_rows = []
    buy_count = 0

    saved_tables = {}

    for date in dates:
        phase = phase_map.loc[phase_map.index <= date].tail(1)
        phase_name = str(phase.iloc[0]) if not phase.empty else "normal"
        month_key = pd.Timestamp(date.year, date.month, 1)
        month_tables = monthly_tables[month_key]
        table_key = table_key_for_phase(phase_name)
        active_table = month_tables[table_key]
        if month_key not in saved_tables:
            saved_tables[month_key] = month_tables
        active_lookup = active_table.set_index("ticker").to_dict("index") if not active_table.empty else {}
        rule_name = mapping.get(phase_name, "condition2")

        signal_today: dict[str, bool] = {}
        score_today: dict[str, float] = {}
        basis_date_today: dict[str, pd.Timestamp] = {}

        for ticker in price_map:
            if ticker not in active_lookup:
                signal_today[ticker] = False
                continue
            metrics = metrics_cache.get(ticker, {}).get(date, {})
            if not metrics:
                signal_today[ticker] = prev_signal.get(ticker, False) if date not in price_map[ticker].index else False
                continue
            sig, signal_score = eval_signal(rule_name, metrics, active_lookup[ticker])
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
        equity_rows.append({"date": date, "nikkei_phase": phase_name, "table_key": table_key, "equity": cash + market_value})
        prev_signal = signal_today

    latest_date = max(dates)
    for ticker in list(positions.keys()):
        df = price_map[ticker]
        usable_idx = df.index[df.index <= latest_date]
        if len(usable_idx):
            cash += positions[ticker]["shares"] * float(df.loc[usable_idx[-1], "Close"])
        del positions[ticker]

    output_dir.mkdir(parents=True, exist_ok=True)
    for month_key, tables in saved_tables.items():
        stamp = month_key.strftime("%Y%m")
        for name, tbl in tables.items():
            tbl.to_csv(output_dir / f"{stamp}_{name}_table.csv", index=False, encoding="utf-8-sig")

    equity_df = pd.DataFrame(equity_rows)
    equity_df.to_csv(output_dir / "equity_curve.csv", index=False, encoding="utf-8-sig")
    if not equity_df.empty:
        equity_df["pnl_day"] = equity_df["equity"].diff().fillna(equity_df["equity"] - spec.initial_capital)
        phase_pnl = equity_df.groupby("nikkei_phase")["pnl_day"].sum().to_dict()
        drawdown = equity_df["equity"].astype(float) / equity_df["equity"].astype(float).cummax() - 1.0
        max_drawdown_pct = float(drawdown.min()) * 100.0
    else:
        phase_pnl = {}
        max_drawdown_pct = 0.0

    summary = {
        "mapping_name": "practical_phase_adaptive_v2_monthly_3tables",
        "mapping_ja": "実務向け局面切替v2_月次3テーブル",
        "dataset": spec.name,
        "standard_count": selected_count,
        "intermediate_count": selected_count,
        "high_vol_count": selected_count,
        "initial_capital": float(spec.initial_capital),
        "final_capital": float(cash),
        "total_return_pct": (float(cash) / float(spec.initial_capital) - 1.0) * 100.0,
        "num_buys": int(buy_count),
        "phase_pnl": phase_pnl,
        "max_drawdown_pct": max_drawdown_pct,
        "table_assignment": {
            "high_vol": ["high_vol", "crash"],
            "intermediate": ["settling", "reversal_up", "capitulation_end"],
            "standard": "others",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    phase_map = load_phase_map(PHASE_CSV)
    out_root = ROOT / "projects" / "shikiho_text_parser" / "output" / "phase_adaptive_practical_v2_monthly_3tables"
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
        rows.append(run_backtest(spec, phase_map, out_root / spec.name))
    results_df = pd.DataFrame(rows)
    out_root.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_root / "summary_all.csv", index=False, encoding="utf-8-sig")
    print(results_df.to_string(index=False))
    print(f"[OUT] {out_root / 'summary_all.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

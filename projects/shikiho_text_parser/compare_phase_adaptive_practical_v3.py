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
from projects.shikiho_text_parser.methods.method_q2_2024_bad_regime_concentrated import (  # noqa: E402
    SELECTION_RULE,
    TRADING_RULE,
)


OUT_DIR = ROOT / "projects" / "shikiho_text_parser" / "output" / "phase_adaptive_practical_v3_compare"


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
        df = df[df.index <= end_date].copy()
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
        rec = dict(row)
        rec.update(metrics)
        rec["max_drawdown_pct"] = abs(float(metrics["max_drawdown_pct"]))
        rec["end_to_trailing_high_pct"] = float(close.iloc[-1] / close.max() * 100.0) if len(close) else pd.NA
        rows.append(rec)
    snap = pd.DataFrame(rows)
    if snap.empty:
        return snap
    return normalize_static(snap)


def build_q2_defensive_table(snapshot: pd.DataFrame) -> pd.DataFrame:
    if snapshot.empty:
        return snapshot
    x = snapshot.copy()
    x = x[x["trend_r2"].fillna(0) >= SELECTION_RULE["trend_r2_min"]]
    x = x[x["annual_return_pct"].fillna(-999) >= SELECTION_RULE["annual_return_min"]]
    x = x[x["quarter_return_pct"].fillna(-999) >= SELECTION_RULE["quarter_return_min"]]
    x = x[x["positive_month_ratio_pct"].fillna(0) >= SELECTION_RULE["positive_month_ratio_min"]]
    x = x[x["persistence_20d_pct"].fillna(0) >= SELECTION_RULE["persistence_20d_min"]]
    x = x[x["max_drawdown_pct"].fillna(999) <= SELECTION_RULE["max_drawdown_max"]]
    x = x[(x["avg_monthly_high_low_change_pct"].isna()) | (x["avg_monthly_high_low_change_pct"] <= SELECTION_RULE["avg_monthly_range_max"])]
    x = x[(x["end_to_trailing_high_pct"].isna()) | (x["end_to_trailing_high_pct"] >= SELECTION_RULE["end_to_trailing_high_min"])]
    x = x[x["sector_adjusted_per_score"].fillna(0) >= SELECTION_RULE["sector_adjusted_per_score_min"]]
    x = x[x["ocr_per"].fillna(999) <= SELECTION_RULE["ocr_per_max"]]
    x = x.sort_values(SELECTION_RULE["sort_col"], ascending=SELECTION_RULE["sort_ascending"])
    return x.head(SELECTION_RULE["top_n"]).reset_index(drop=True)


def prepare_metric_cache(price_map: dict[str, pd.DataFrame], dates: list[pd.Timestamp]) -> dict[str, dict[pd.Timestamp, dict]]:
    cache: dict[str, dict[pd.Timestamp, dict]] = {}
    for ticker, df in price_map.items():
        per_ticker: dict[pd.Timestamp, dict] = {}
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
            per_ticker[date] = metrics
        cache[ticker] = per_ticker
    return cache


def q2_defensive_signal(metrics: dict, static_row: dict) -> tuple[bool, float]:
    if not metrics:
        return False, 0.0
    signal = (
        metrics["trend_r2"] >= SELECTION_RULE["trend_r2_min"]
        and metrics["annual_return_pct"] >= SELECTION_RULE["annual_return_min"]
        and metrics["quarter_return_pct"] >= SELECTION_RULE["quarter_return_min"]
        and metrics["positive_month_ratio_pct"] >= SELECTION_RULE["positive_month_ratio_min"]
        and metrics["persistence_20d_pct"] >= SELECTION_RULE["persistence_20d_min"]
        and metrics["max_drawdown_pct"] >= -SELECTION_RULE["max_drawdown_max"]
        and float(static_row.get("sector_adjusted_per_score", 0) if pd.notna(static_row.get("sector_adjusted_per_score")) else 0) >= SELECTION_RULE["sector_adjusted_per_score_min"]
        and (float(static_row.get("forecast_per")) <= SELECTION_RULE["ocr_per_max"] if pd.notna(static_row.get("forecast_per")) else False)
    )
    score = (
        0.35 * min(max(static_row.get("sector_adjusted_per_score", 0) or 0, 0.0), 1.0)
        + 0.20 * min(max(metrics["trend_r2"], 0.0), 1.0)
        + 0.15 * min(max(metrics["annual_return_pct"] / 100.0, 0.0), 1.0)
        + 0.10 * min(max(metrics["quarter_return_pct"] / 40.0, 0.0), 1.0)
        + 0.10 * min(max(metrics["persistence_20d_pct"] / 100.0, 0.0), 1.0)
        + 0.10 * min(max(metrics["positive_month_ratio_pct"] / 100.0, 0.0), 1.0)
    )
    return signal, score


def build_monthly_defensive_tables(detail_df: pd.DataFrame, price_map: dict[str, pd.DataFrame], trade_dates: list[pd.Timestamp], monthly: bool) -> dict[pd.Timestamp, pd.DataFrame]:
    month_starts = sorted({pd.Timestamp(d.year, d.month, 1) for d in trade_dates})
    tables: dict[pd.Timestamp, pd.DataFrame] = {}
    if monthly:
        for month_key in month_starts:
            prior_trade_dates = [d for d in trade_dates if d < month_key]
            basis_date = prior_trade_dates[-1] if prior_trade_dates else trade_dates[0] - pd.Timedelta(days=1)
            tables[month_key] = build_q2_defensive_table(build_snapshot(detail_df, price_map, basis_date))
    else:
        basis_date = trade_dates[0] - pd.Timedelta(days=1)
        fixed_table = build_q2_defensive_table(build_snapshot(detail_df, price_map, basis_date))
        for month_key in month_starts:
            tables[month_key] = fixed_table.copy()
    return tables


def run_dataset(spec: TableSpec, phase_map: pd.Series, monthly: bool) -> dict:
    detail_df = normalize_static(pd.read_csv(spec.detail_csv)).drop_duplicates(subset=["ticker"], keep="first")
    selected_df = pd.read_csv(spec.selected_csv)
    standard_tickers = set(selected_df["ticker"].astype(str))
    standard_table = detail_df[detail_df["ticker"].astype(str).isin(standard_tickers)].copy().drop_duplicates(subset=["ticker"], keep="first")

    start_date = pd.Timestamp(spec.start_date)
    end_date = pd.Timestamp(spec.end_date)
    price_map_all = load_price_map(detail_df["ticker"].astype(str).tolist(), end_date)
    trade_dates = sorted(set().union(*[set(df[(df.index >= start_date) & (df.index <= end_date)].index.tolist()) for df in price_map_all.values()]))
    defensive_tables = build_monthly_defensive_tables(detail_df, price_map_all, trade_dates, monthly)

    union_tickers = set(standard_table["ticker"].astype(str))
    for tbl in defensive_tables.values():
        if not tbl.empty:
            union_tickers.update(tbl["ticker"].astype(str).tolist())
    price_map = {ticker: df for ticker, df in price_map_all.items() if ticker in union_tickers}
    dates = sorted(set().union(*[set(df[(df.index >= start_date) & (df.index <= end_date)].index.tolist()) for df in price_map.values()]))
    metrics_cache = prepare_metric_cache(price_map, dates)
    standard_lookup = standard_table.set_index("ticker").to_dict("index")
    mapping = practical_v2_mapping()

    cash = float(spec.initial_capital)
    positions: dict[str, dict] = {}
    prev_signal = {ticker: False for ticker in price_map}
    equity_rows = []
    buy_count = 0

    for date in dates:
        phase = phase_map.loc[phase_map.index <= date].tail(1)
        phase_name = str(phase.iloc[0]) if not phase.empty else "normal"
        month_key = pd.Timestamp(date.year, date.month, 1)
        active_lookup = defensive_tables[month_key].set_index("ticker").to_dict("index") if phase_name == "crash" and not defensive_tables[month_key].empty else standard_lookup
        rule_name = "q2_defensive" if phase_name == "crash" else mapping.get(phase_name, "condition2")

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
            if rule_name == "q2_defensive":
                sig, signal_score = q2_defensive_signal(metrics, active_lookup[ticker])
            else:
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

            tp = TRADING_RULE["take_profit_pct"] / 100.0 if positions[ticker]["rule_name"] == "q2_defensive" else 0.08
            sl = TRADING_RULE["stop_loss_pct"] / 100.0 if positions[ticker]["rule_name"] == "q2_defensive" else 0.05

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
            if exit_reason is None and ret >= tp:
                exit_reason = "take_profit"
                exit_price = close_price
            elif exit_reason is None and ret <= -sl:
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
            entry_limit = TRADING_RULE["entry_limit_pct"] if rule_name == "q2_defensive" else 1.5
            trigger_price = prev_close * (1.0 + entry_limit / 100.0)
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
                positions[ticker] = {"shares": shares, "entry_price": fill_price, "rule_name": rule_name}
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
        "max_drawdown_pct": max_drawdown_pct,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase_map = load_phase_map(PHASE_CSV)
    specs = [
        TableSpec("q2_2024", ROOT / "projects" / "quarterly_ranker" / "output" / "q2_2024_pre_analysis_20240630_aligned" / "q2_2024_pre_shikiho_feature_ranking.csv", ROOT / "projects" / "quarterly_ranker" / "output" / "q2_2024_pre_analysis_20240630_aligned" / "q2_2024_pre_selected_candidates.csv", "2024-07-01", "2024-09-30"),
        TableSpec("q3", ROOT / "projects" / "quarterly_ranker" / "output" / "q3_pre_analysis_20250630_aligned" / "q3_pre_shikiho_feature_ranking.csv", ROOT / "projects" / "quarterly_ranker" / "output" / "q3_pre_analysis_20250630_aligned" / "q3_pre_selected_candidates.csv", "2025-07-01", "2025-09-30"),
        TableSpec("q4", ROOT / "projects" / "quarterly_ranker" / "output" / "q4_pre_analysis_20250930_full" / "q4_pre_shikiho_feature_ranking.csv", ROOT / "projects" / "quarterly_ranker" / "output" / "q4_pre_analysis_20250930_full" / "q4_pre_selected_candidates.csv", "2025-10-01", "2025-12-31"),
        TableSpec("4q2", ROOT / "projects" / "shikiho_text_parser" / "output" / "4q2_selection" / "4q2_scored_universe.csv", ROOT / "projects" / "shikiho_text_parser" / "output" / "4q2_selection" / "4q2_selected_candidates.csv", "2026-01-01", "2026-03-10"),
    ]

    rows = []
    for mode_name, monthly in [("fixed", False), ("monthly", True)]:
        row = {"mode_name": mode_name}
        total = 0.0
        for spec in specs:
            result = run_dataset(spec, phase_map, monthly)
            total += result["final_capital"]
            row[f"{spec.name}_return_pct"] = result["total_return_pct"]
            row[f"{spec.name}_final_capital"] = result["final_capital"]
        row["total_final_capital"] = total
        row["total_return_pct"] = (total / (3_000_000.0 * len(specs)) - 1.0) * 100.0
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("total_final_capital", ascending=False)
    df.to_csv(OUT_DIR / "compare_results.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "best_summary.json").write_text(df.iloc[0].to_json(force_ascii=False, indent=2), encoding="utf-8")
    print(df.to_string(index=False))
    print(f"[OUT] {OUT_DIR / 'compare_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

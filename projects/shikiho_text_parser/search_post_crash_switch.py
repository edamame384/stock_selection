from __future__ import annotations

import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.shikiho_text_parser.backtest_phase_table_switch import (
    PRICE_DIR,
    PHASE_CSV,
    load_candidates,
    load_phase_map,
)


OUT_DIR = ROOT / "projects" / "shikiho_text_parser" / "output" / "post_crash_switch_search"
POST_CRASH_PHASES = {"crash", "high_vol", "settling", "reversal_up", "capitulation_end"}


@dataclass
class DatasetSpec:
    name: str
    selected_csv: Path
    start_date: str
    end_date: str
    initial_capital: float = 3_000_000.0


def build_table(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    x = df.copy()
    score_col = "overall_score" if "overall_score" in x.columns else ("promising_score" if "promising_score" in x.columns else "score")
    if "avg_monthly_high_low_change_pct" not in x.columns:
        x["avg_monthly_high_low_change_pct"] = np.nan
    if "end_to_trailing_high_pct" not in x.columns:
        x["end_to_trailing_high_pct"] = np.nan

    if table_name == "none":
        return x.iloc[0:0].copy()

    if table_name.startswith("regular_top"):
        top_n = int(table_name.replace("regular_top", ""))
        x = x[x["trend_r2"].fillna(0) >= 0.50]
        x = x[x["annual_return_pct"].fillna(-999) >= 15.0]
        x = x[x["quarter_return_pct"].fillna(-999) >= 5.0]
        x = x[x["positive_month_ratio_pct"].fillna(0) >= 50.0]
        x = x[x["persistence_20d_pct"].fillna(0) >= 45.0]
        x = x[x["sector_adjusted_per_score"].fillna(0) >= 0.45]
        x = x[x["ocr_per"].fillna(999) <= 20.0]
        x = x.sort_values(
            ["sector_adjusted_per_score", "trend_r2", "annual_return_pct", "quarter_return_pct"],
            ascending=[False, False, False, False],
        )
        return x.head(top_n).reset_index(drop=True)

    if table_name.startswith("momentum_top"):
        top_n = int(table_name.replace("momentum_top", ""))
        x = x[x["trend_r2"].fillna(0) >= 0.60]
        x = x[x["annual_return_pct"].fillna(-999) >= 30.0]
        x = x[x["quarter_return_pct"].fillna(-999) >= 15.0]
        x = x[x["positive_month_ratio_pct"].fillna(0) >= 60.0]
        x = x[x["persistence_20d_pct"].fillna(0) >= 60.0]
        x = x[x["sector_adjusted_per_score"].fillna(0) >= 0.55]
        x = x[x["ocr_per"].fillna(999) <= 20.0]
        x = x.sort_values(
            ["quarter_return_pct", "trend_r2", "annual_return_pct", "sector_adjusted_per_score"],
            ascending=[False, False, False, False],
        )
        return x.head(top_n).reset_index(drop=True)

    if table_name.startswith("post_high_vol_top"):
        top_n = int(table_name.replace("post_high_vol_top", ""))
        x = x[x["trend_r2"].fillna(0) >= 0.35]
        x = x[x["annual_return_pct"].fillna(-999) >= 0.0]
        x = x[x["quarter_return_pct"].fillna(-999) >= -10.0]
        x = x[x["positive_month_ratio_pct"].fillna(0) >= 35.0]
        x = x[x["persistence_20d_pct"].fillna(0) >= 35.0]
        x = x[x["max_drawdown_pct"].fillna(999) <= 18.0]
        x = x[x["sector_adjusted_per_score"].fillna(0) >= 0.35]
        x = x[x["ocr_per"].fillna(999) <= 18.0]
        x = x.sort_values(
            ["sector_adjusted_per_score", "max_drawdown_pct", "trend_r2", "annual_return_pct"],
            ascending=[False, True, False, False],
        )
        return x.head(top_n).reset_index(drop=True)

    if table_name.startswith("defensive_top"):
        top_n = int(table_name.replace("defensive_top", ""))
        x = x[x["trend_r2"].fillna(0) >= 0.35]
        x = x[x["annual_return_pct"].fillna(-999) >= 0.0]
        x = x[x["quarter_return_pct"].fillna(-999) >= -10.0]
        x = x[x["positive_month_ratio_pct"].fillna(0) >= 35.0]
        x = x[x["persistence_20d_pct"].fillna(0) >= 35.0]
        x = x[x["max_drawdown_pct"].fillna(999) <= 18.0]
        x = x[(x["avg_monthly_high_low_change_pct"].isna()) | (x["avg_monthly_high_low_change_pct"] <= 14.0)]
        x = x[(x["end_to_trailing_high_pct"].isna()) | (x["end_to_trailing_high_pct"] >= 88.0)]
        x = x[x["sector_adjusted_per_score"].fillna(0) >= 0.35]
        x = x[x["ocr_per"].fillna(999) <= 18.0]
        x = x.sort_values(
            ["sector_adjusted_per_score", "end_to_trailing_high_pct", "max_drawdown_pct", score_col],
            ascending=[False, False, True, False],
        )
        return x.head(top_n).reset_index(drop=True)

    raise ValueError(table_name)


def trade_rule(rule_name: str) -> dict[str, float]:
    if rule_name == "defensive":
        return {"entry_limit_pct": 1.0, "take_profit_pct": 5.0, "stop_loss_pct": 4.0}
    if rule_name == "normal":
        return {"entry_limit_pct": 1.5, "take_profit_pct": 8.0, "stop_loss_pct": 5.0}
    raise ValueError(rule_name)


def calc_signal(df: pd.DataFrame, date: pd.Timestamp, static_row: dict) -> tuple[bool, dict]:
    hist = df.loc[:date].copy()
    if len(hist) < 120:
        return False, {}
    trailing = hist.tail(min(252, len(hist))).copy()
    qdf = hist.tail(min(63, len(hist))).copy()
    if len(trailing) < 120 or len(qdf) < 20:
        return False, {}

    close = trailing["Close"].astype(float)
    annual_return_pct = (close.iloc[-1] / close.iloc[0] - 1.0) * 100.0
    quarter_return_pct = (qdf["Close"].iloc[-1] / qdf["Close"].iloc[0] - 1.0) * 100.0
    y = np.log(close.replace(0, np.nan).dropna().values)
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    trend_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    monthly_close = trailing["Close"].resample("ME").last().dropna()
    monthly_ret = monthly_close.pct_change().dropna()
    positive_month_ratio_pct = float((monthly_ret > 0).mean()) * 100.0 if len(monthly_ret) else 0.0
    ma20 = trailing["Close"].rolling(20).mean()
    ma60 = trailing["Close"].rolling(60).mean()
    last20 = trailing.iloc[-20:].copy()
    ma20_last20 = ma20.reindex(last20.index)
    ma60_last20 = ma60.reindex(last20.index)
    persistence_20d_pct = float(((last20["Close"] > ma20_last20) & (ma20_last20 > ma60_last20)).mean()) * 100.0
    max_drawdown_pct = float(((close / close.cummax()) - 1.0).min()) * 100.0

    signal = (
        annual_return_pct >= float(static_row.get("annual_return_pct", -999))
        and quarter_return_pct >= float(static_row.get("quarter_return_pct", -999))
        and trend_r2 >= float(static_row.get("trend_r2", 0))
        and positive_month_ratio_pct >= float(static_row.get("positive_month_ratio_pct", 0))
        and persistence_20d_pct >= float(static_row.get("persistence_20d_pct", 0))
        and max_drawdown_pct >= -28.0
    )
    signal_score = (
        0.30 * min(max(annual_return_pct / 100.0, 0.0), 1.0)
        + 0.20 * min(max(quarter_return_pct / 40.0, 0.0), 1.0)
        + 0.20 * trend_r2
        + 0.10 * min(max(persistence_20d_pct / 100.0, 0.0), 1.0)
        + 0.10 * min(max(positive_month_ratio_pct / 100.0, 0.0), 1.0)
        + 0.10 * min(max((28.0 + max_drawdown_pct) / 28.0, 0.0), 1.0)
    )
    return signal, {"signal_score": signal_score}


def run_dataset(spec: DatasetSpec, phase_map: pd.Series, post_table_name: str, other_table_name: str, post_rule_name: str, other_rule_name: str) -> dict:
    universe = load_candidates(spec.selected_csv)
    tables = {
        "post": build_table(universe, post_table_name),
        "other": build_table(universe, other_table_name),
    }
    tickers = sorted(set().union(*[set(tbl["ticker"]) for tbl in tables.values()]))

    price_map: dict[str, pd.DataFrame] = {}
    all_dates: set[pd.Timestamp] = set()
    start = pd.Timestamp(spec.start_date)
    end = pd.Timestamp(spec.end_date)
    for ticker in tickers:
        path = PRICE_DIR / f"{ticker.replace('.', '_')}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "Date" not in df.columns or "Close" not in df.columns:
            continue
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").set_index("Date")
        df = df[(df.index >= pd.Timestamp("2023-01-01")) & (df.index <= end)].copy()
        if df.empty:
            continue
        price_map[ticker] = df
        all_dates.update(df[(df.index >= start) & (df.index <= end)].index.tolist())

    dates = sorted(all_dates)
    cash = float(spec.initial_capital)
    positions: dict[str, dict] = {}
    prev_signal = {ticker: False for ticker in price_map}
    equity_rows = []
    buy_count = 0

    lookups = {name: tbl.set_index("ticker").to_dict("index") for name, tbl in tables.items()}

    for date in dates:
        phase = phase_map.loc[phase_map.index <= date].tail(1)
        phase_name = str(phase.iloc[0]) if not phase.empty else "normal"
        regime = "post" if phase_name in POST_CRASH_PHASES else "other"
        table_df = tables[regime]
        lookup = lookups[regime]
        rule = trade_rule(post_rule_name if regime == "post" else other_rule_name)

        signal_today: dict[str, bool] = {}
        score_today: dict[str, float] = {}
        basis_date_today: dict[str, pd.Timestamp] = {}

        for ticker in table_df["ticker"]:
            if ticker not in price_map:
                continue
            df = price_map[ticker]
            if date not in df.index:
                signal_today[ticker] = prev_signal.get(ticker, False)
                continue
            prev_idx = df.index[df.index < date]
            if len(prev_idx) == 0:
                signal_today[ticker] = False
                continue
            signal_date = prev_idx[-1]
            sig, metrics = calc_signal(df, signal_date, lookup[ticker])
            signal_today[ticker] = sig
            score_today[ticker] = metrics.get("signal_score", 0.0)
            basis_date_today[ticker] = signal_date

        for ticker in list(positions.keys()):
            df = price_map[ticker]
            if date not in df.index:
                continue
            day = df.loc[date]
            close_price = float(day["Close"])
            ret = close_price / positions[ticker]["entry_price"] - 1.0
            pos_rule = trade_rule(positions[ticker]["rule_name"])
            exit_reason = None
            if ret >= pos_rule["take_profit_pct"] / 100.0:
                exit_reason = "take_profit"
            elif ret <= -pos_rule["stop_loss_pct"] / 100.0:
                exit_reason = "stop_loss"
            elif not signal_today.get(ticker, False):
                exit_reason = "sell_signal"
            if exit_reason is not None:
                cash += positions[ticker]["shares"] * close_price
                del positions[ticker]

        candidates = []
        for ticker, sig in signal_today.items():
            if ticker in positions:
                continue
            if sig and not prev_signal.get(ticker, False):
                candidates.append((ticker, score_today.get(ticker, 0.0)))
        candidates.sort(key=lambda x: x[1], reverse=True)

        remaining = len(candidates)
        for ticker, score in candidates:
            df = price_map[ticker]
            day = df.loc[date]
            prev_close = float(df.loc[basis_date_today[ticker], "Close"])
            trigger_price = prev_close * (1.0 + rule["entry_limit_pct"] / 100.0)
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
                positions[ticker] = {"shares": shares, "entry_price": fill_price, "rule_name": post_rule_name if regime == "post" else other_rule_name}
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

    latest_date = max(dates) if dates else pd.Timestamp(spec.end_date)
    for ticker in list(positions.keys()):
        df = price_map[ticker]
        usable_idx = df.index[df.index <= latest_date]
        if len(usable_idx):
            cash += positions[ticker]["shares"] * float(df.loc[usable_idx[-1], "Close"])
        del positions[ticker]

    equity_df = pd.DataFrame(equity_rows)
    if not equity_df.empty:
        drawdown = equity_df["equity"].astype(float) / equity_df["equity"].astype(float).cummax() - 1.0
        max_dd_pct = float(drawdown.min()) * 100.0
    else:
        max_dd_pct = 0.0

    return {
        "dataset": spec.name,
        "final_capital": float(cash),
        "total_return_pct": (float(cash) / float(spec.initial_capital) - 1.0) * 100.0,
        "num_buys": int(buy_count),
        "max_drawdown_pct": max_dd_pct,
        "post_table_name": post_table_name,
        "other_table_name": other_table_name,
        "post_rule_name": post_rule_name,
        "other_rule_name": other_rule_name,
        "post_count": int(len(tables["post"])),
        "other_count": int(len(tables["other"])),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase_map = load_phase_map(PHASE_CSV)
    specs = [
        DatasetSpec("q2_2024", ROOT / "projects" / "quarterly_ranker" / "output" / "q2_2024_pre_analysis_20240630_full" / "q2_2024_pre_shikiho_feature_ranking.csv", "2024-07-01", "2024-09-30"),
        DatasetSpec("q3", ROOT / "projects" / "quarterly_ranker" / "output" / "q3_pre_analysis_20250630_aligned" / "q3_pre_shikiho_feature_ranking.csv", "2025-07-01", "2025-09-30"),
        DatasetSpec("q4", ROOT / "projects" / "quarterly_ranker" / "output" / "q4_pre_analysis_20250930_full" / "q4_pre_shikiho_feature_ranking.csv", "2025-10-01", "2025-12-31"),
        DatasetSpec("4q2", ROOT / "projects" / "shikiho_text_parser" / "output" / "4q2_selection" / "4q2_scored_universe.csv", "2026-01-01", "2026-03-10"),
    ]

    post_tables = ["none", "post_high_vol_top20", "defensive_top5", "defensive_top10"]
    other_tables = ["regular_top20", "regular_top30", "momentum_top20"]
    post_rules = ["defensive", "normal"]
    other_rules = ["defensive", "normal"]

    rows = []
    best = None
    for post_table_name, other_table_name, post_rule_name, other_rule_name in itertools.product(post_tables, other_tables, post_rules, other_rules):
        results = [run_dataset(spec, phase_map, post_table_name, other_table_name, post_rule_name, other_rule_name) for spec in specs]
        total_final = sum(r["final_capital"] for r in results)
        total_return_pct = (total_final / (3_000_000.0 * len(results)) - 1.0) * 100.0
        avg_dd = float(np.mean([r["max_drawdown_pct"] for r in results]))
        row = {
            "post_table_name": post_table_name,
            "other_table_name": other_table_name,
            "post_rule_name": post_rule_name,
            "other_rule_name": other_rule_name,
            "total_final_capital": total_final,
            "total_return_pct": total_return_pct,
            "avg_max_drawdown_pct": avg_dd,
        }
        for r in results:
            row[f"{r['dataset']}_return_pct"] = r["total_return_pct"]
            row[f"{r['dataset']}_final_capital"] = r["final_capital"]
        rows.append(row)
        if best is None or row["total_final_capital"] > best["total_final_capital"]:
            best = row

    results_df = pd.DataFrame(rows).sort_values(["total_final_capital", "avg_max_drawdown_pct"], ascending=[False, False])
    results_df.to_csv(OUT_DIR / "search_results.csv", index=False, encoding="utf-8-sig")
    if best is not None:
        pd.DataFrame([best]).to_csv(OUT_DIR / "best_summary.csv", index=False, encoding="utf-8-sig")
        print(pd.DataFrame([best]).to_string(index=False))
    print(f"[OUT] {OUT_DIR / 'search_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

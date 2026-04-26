from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.shikiho_text_parser.backtest_phase_table_switch import (  # noqa: E402
    PRICE_DIR,
    PHASE_CSV,
    load_candidates,
    load_phase_map,
)
from projects.shikiho_text_parser.search_post_crash_switch_origcount import (  # noqa: E402
    build_table,
    calc_signal,
)


@dataclass
class DatasetSpec:
    name: str
    detail_csv: Path
    selected_csv: Path
    start_date: str
    end_date: str
    initial_capital: float = 3_000_000.0


def trade_rule(name: str) -> dict[str, float]:
    if name == "normal":
        return {"entry_limit_pct": 1.5, "take_profit_pct": 8.0, "stop_loss_pct": 5.0}
    if name == "defensive":
        return {"entry_limit_pct": 1.0, "take_profit_pct": 5.0, "stop_loss_pct": 4.0}
    raise ValueError(name)


def make_standard_table(detail_df: pd.DataFrame, selected_df: pd.DataFrame) -> pd.DataFrame:
    tickers = set(selected_df["ticker"].astype(str))
    out = detail_df[detail_df["ticker"].astype(str).isin(tickers)].copy()
    return out.drop_duplicates(subset=["ticker"]).reset_index(drop=True)


def run_dataset(
    spec: DatasetSpec,
    phase_map: pd.Series,
    high_vol_table_name: str,
    output_dir: Path,
    standard_rule_name: str = "normal",
    high_vol_rule_name: str = "defensive",
) -> dict:
    detail_df = load_candidates(spec.detail_csv)
    selected_df = pd.read_csv(spec.selected_csv)
    original_count = int(selected_df["ticker"].nunique())
    standard_table = make_standard_table(detail_df, selected_df)
    high_vol_table = build_table(detail_df, high_vol_table_name, original_count)

    tables = {"standard": standard_table, "high_vol": high_vol_table}
    lookups = {name: tbl.set_index("ticker").to_dict("index") for name, tbl in tables.items()}

    tickers = sorted(set(standard_table["ticker"]).union(set(high_vol_table["ticker"])))
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
    trade_log_rows = []
    phase_pnl_rows = []
    buy_count = 0

    for date in dates:
        phase = phase_map.loc[phase_map.index <= date].tail(1)
        phase_name = str(phase.iloc[0]) if not phase.empty else "normal"
        regime = "high_vol" if phase_name == "high_vol" else "standard"
        table_df = tables[regime]
        lookup = lookups[regime]
        rule_name = high_vol_rule_name if regime == "high_vol" else standard_rule_name
        rule = trade_rule(rule_name)

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
                proceeds = positions[ticker]["shares"] * close_price
                pnl = proceeds - positions[ticker]["shares"] * positions[ticker]["entry_price"]
                cash += proceeds
                trade_log_rows.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "action": "SELL",
                        "price": close_price,
                        "shares": positions[ticker]["shares"],
                        "phase": phase_name,
                        "table_regime": positions[ticker]["table_regime"],
                        "rule_name": positions[ticker]["rule_name"],
                        "reason": exit_reason,
                        "pnl": pnl,
                    }
                )
                phase_pnl_rows.append({"phase": phase_name, "pnl": pnl})
                del positions[ticker]

        candidates = []
        for ticker, sig in signal_today.items():
            if ticker in positions:
                continue
            if sig and not prev_signal.get(ticker, False):
                df = price_map[ticker]
                if date not in df.index:
                    continue
                signal_date = basis_date_today.get(ticker)
                if signal_date is None:
                    continue
                prev_close = float(df.loc[signal_date, "Close"])
                entry_trigger = prev_close * (1.0 + rule["entry_limit_pct"] / 100.0)
                high = float(df.loc[date, "High"])
                open_px = float(df.loc[date, "Open"])
                if high >= entry_trigger:
                    entry_px = max(open_px, entry_trigger)
                    candidates.append((ticker, score_today.get(ticker, 0.0), entry_px))

        candidates.sort(key=lambda x: x[1], reverse=True)
        for ticker, _score, entry_px in candidates:
            shares = int(cash // (entry_px * 100.0)) * 100
            if shares < 100:
                continue
            cost = shares * entry_px
            cash -= cost
            positions[ticker] = {
                "shares": shares,
                "entry_price": entry_px,
                "rule_name": rule_name,
                "table_regime": regime,
            }
            buy_count += 1
            trade_log_rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "action": "BUY",
                    "price": entry_px,
                    "shares": shares,
                    "phase": phase_name,
                    "table_regime": regime,
                    "rule_name": rule_name,
                    "reason": "entry",
                    "pnl": 0.0,
                }
            )

        market_value = 0.0
        for ticker, pos in positions.items():
            df = price_map[ticker]
            usable = df.index[df.index <= date]
            if len(usable):
                market_value += pos["shares"] * float(df.loc[usable[-1], "Close"])
        equity_rows.append({"Date": date, "equity": cash + market_value})
        prev_signal = signal_today

    latest_date = max(dates) if dates else pd.Timestamp(spec.end_date)
    for ticker in list(positions.keys()):
        df = price_map[ticker]
        usable_idx = df.index[df.index <= latest_date]
        if len(usable_idx):
            close_price = float(df.loc[usable_idx[-1], "Close"])
            proceeds = positions[ticker]["shares"] * close_price
            pnl = proceeds - positions[ticker]["shares"] * positions[ticker]["entry_price"]
            cash += proceeds
            trade_log_rows.append(
                {
                    "date": usable_idx[-1],
                    "ticker": ticker,
                    "action": "SELL",
                    "price": close_price,
                    "shares": positions[ticker]["shares"],
                    "phase": "end",
                    "table_regime": positions[ticker]["table_regime"],
                    "rule_name": positions[ticker]["rule_name"],
                    "reason": "end_of_backtest",
                    "pnl": pnl,
                }
            )
            phase_pnl_rows.append({"phase": "end", "pnl": pnl})
        del positions[ticker]

    equity_df = pd.DataFrame(equity_rows)
    if not equity_df.empty:
        dd = equity_df["equity"].astype(float) / equity_df["equity"].astype(float).cummax() - 1.0
        max_dd_pct = float(dd.min()) * 100.0
    else:
        max_dd_pct = 0.0

    trade_log_df = pd.DataFrame(trade_log_rows)
    phase_pnl_df = pd.DataFrame(phase_pnl_rows)
    phase_summary = (
        phase_pnl_df.groupby("phase", dropna=False)["pnl"].sum().reset_index().sort_values("pnl", ascending=False)
        if not phase_pnl_df.empty
        else pd.DataFrame(columns=["phase", "pnl"])
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    equity_df.to_csv(output_dir / "equity_curve.csv", index=False, encoding="utf-8-sig")
    trade_log_df.to_csv(output_dir / "trade_log.csv", index=False, encoding="utf-8-sig")
    phase_summary.to_csv(output_dir / "phase_pnl_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "dataset": spec.name,
        "standard_count": int(len(standard_table)),
        "high_vol_count": int(len(high_vol_table)),
        "initial_capital": float(spec.initial_capital),
        "final_capital": float(cash),
        "total_return_pct": (float(cash) / float(spec.initial_capital) - 1.0) * 100.0,
        "num_buys": int(buy_count),
        "max_drawdown_pct": max_dd_pct,
        "standard_rule": standard_rule_name,
        "high_vol_rule": high_vol_rule_name,
        "high_vol_table": high_vol_table_name,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    phase_map = load_phase_map(PHASE_CSV)
    specs = [
        DatasetSpec(
            "q2_2024",
            ROOT / "projects" / "quarterly_ranker" / "output" / "q2_2024_pre_analysis_20240630_aligned" / "q2_2024_pre_shikiho_feature_ranking.csv",
            ROOT / "projects" / "quarterly_ranker" / "output" / "q2_2024_pre_analysis_20240630_aligned" / "q2_2024_pre_selected_candidates.csv",
            "2024-07-01",
            "2024-09-30",
        ),
        DatasetSpec(
            "q3",
            ROOT / "projects" / "quarterly_ranker" / "output" / "q3_pre_analysis_20250630_aligned" / "q3_pre_shikiho_feature_ranking.csv",
            ROOT / "projects" / "quarterly_ranker" / "output" / "q3_pre_analysis_20250630_aligned" / "q3_pre_selected_candidates.csv",
            "2025-07-01",
            "2025-09-30",
        ),
        DatasetSpec(
            "q4",
            ROOT / "projects" / "quarterly_ranker" / "output" / "q4_pre_analysis_20250930_full" / "q4_pre_shikiho_feature_ranking.csv",
            ROOT / "projects" / "quarterly_ranker" / "output" / "q4_pre_analysis_20250930_full" / "q4_pre_selected_candidates.csv",
            "2025-10-01",
            "2025-12-31",
        ),
        DatasetSpec(
            "4q2",
            ROOT / "projects" / "shikiho_text_parser" / "output" / "4q2_selection" / "4q2_scored_universe.csv",
            ROOT / "projects" / "shikiho_text_parser" / "output" / "4q2_selection" / "4q2_selected_candidates.csv",
            "2026-01-01",
            "2026-03-10",
        ),
    ]

    out_root = ROOT / "projects" / "shikiho_text_parser" / "output" / "standard_vs_highvol_switch"
    results = []
    for high_vol_table_name in ["post_high_vol_origcount", "defensive_origcount"]:
        for spec in specs:
            out_dir = out_root / high_vol_table_name / spec.name
            results.append(run_dataset(spec, phase_map, high_vol_table_name, out_dir))
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_root / "summary_all.csv", index=False, encoding="utf-8-sig")
    print(results_df.to_string(index=False))
    print(f"[OUT] {out_root / 'summary_all.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

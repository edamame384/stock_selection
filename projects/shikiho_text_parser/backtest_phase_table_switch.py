from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.shikiho_text_parser.classify_nikkei_market_phases import OUT_DIR as PHASE_OUT_DIR


PRICE_DIR = ROOT / "data" / "prices"
PHASE_CSV = PHASE_OUT_DIR / "nikkei_market_phase_daily_labels.csv"


def load_phase_map(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").set_index("Date")["phase"]


def load_candidates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename_map = {}
    if "forecast_per" in df.columns and "ocr_per" not in df.columns:
        rename_map["forecast_per"] = "ocr_per"
    if "simple_sector" in df.columns and "sector" not in df.columns:
        rename_map["simple_sector"] = "sector"
    if rename_map:
        df = df.rename(columns=rename_map)
    df = df.drop_duplicates(subset=["ticker"], keep="first").copy()
    numeric_cols = [
        "score",
        "ocr_per",
        "annual_return_pct",
        "quarter_return_pct",
        "trend_r2",
        "max_drawdown_pct",
        "positive_month_ratio_pct",
        "persistence_20d_pct",
        "sector_adjusted_per_score",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_selection_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    common = df.copy()

    high_vol = common.copy()
    high_vol = high_vol[high_vol["trend_r2"].fillna(0) >= 0.35]
    high_vol = high_vol[high_vol["annual_return_pct"].fillna(-999) >= 0.0]
    high_vol = high_vol[high_vol["quarter_return_pct"].fillna(-999) >= -10.0]
    high_vol = high_vol[high_vol["positive_month_ratio_pct"].fillna(0) >= 35.0]
    high_vol = high_vol[high_vol["persistence_20d_pct"].fillna(0) >= 35.0]
    high_vol = high_vol[high_vol["max_drawdown_pct"].fillna(999) <= 18.0]
    high_vol = high_vol[high_vol["sector_adjusted_per_score"].fillna(0) >= 0.35]
    high_vol = high_vol[high_vol["ocr_per"].fillna(999) <= 18.0]
    high_vol = high_vol.sort_values(
        ["sector_adjusted_per_score", "max_drawdown_pct", "trend_r2", "annual_return_pct"],
        ascending=[False, True, False, False],
    ).head(20)

    normal = common.copy()
    normal = normal[normal["trend_r2"].fillna(0) >= 0.50]
    normal = normal[normal["annual_return_pct"].fillna(-999) >= 15.0]
    normal = normal[normal["quarter_return_pct"].fillna(-999) >= 5.0]
    normal = normal[normal["positive_month_ratio_pct"].fillna(0) >= 50.0]
    normal = normal[normal["persistence_20d_pct"].fillna(0) >= 45.0]
    normal = normal[normal["sector_adjusted_per_score"].fillna(0) >= 0.45]
    normal = normal[normal["ocr_per"].fillna(999) <= 20.0]
    normal = normal.sort_values(
        ["sector_adjusted_per_score", "trend_r2", "annual_return_pct", "quarter_return_pct"],
        ascending=[False, False, False, False],
    ).head(30)

    uptrend = common.copy()
    uptrend = uptrend[uptrend["trend_r2"].fillna(0) >= 0.60]
    uptrend = uptrend[uptrend["annual_return_pct"].fillna(-999) >= 30.0]
    uptrend = uptrend[uptrend["quarter_return_pct"].fillna(-999) >= 15.0]
    uptrend = uptrend[uptrend["positive_month_ratio_pct"].fillna(0) >= 60.0]
    uptrend = uptrend[uptrend["persistence_20d_pct"].fillna(0) >= 60.0]
    uptrend = uptrend[uptrend["sector_adjusted_per_score"].fillna(0) >= 0.55]
    uptrend = uptrend[uptrend["ocr_per"].fillna(999) <= 20.0]
    uptrend = uptrend.sort_values(
        ["quarter_return_pct", "trend_r2", "annual_return_pct", "sector_adjusted_per_score"],
        ascending=[False, False, False, False],
    ).head(20)

    return {
        "high_vol": high_vol.reset_index(drop=True),
        "normal": normal.reset_index(drop=True),
        "uptrend": uptrend.reset_index(drop=True),
    }


def table_for_phase(phase: str) -> str | None:
    if phase in {"high_vol", "settling", "reversal_up", "capitulation_end"}:
        return "high_vol"
    if phase in {"normal", "stable", "overheated_range"}:
        return "normal"
    if phase in {"uptrend", "surge"}:
        return "uptrend"
    return None


def trade_rule_for_table(table_name: str) -> dict[str, float]:
    if table_name == "high_vol":
        return {"entry_limit_pct": 1.0, "take_profit_pct": 5.0, "stop_loss_pct": 4.0}
    if table_name == "normal":
        return {"entry_limit_pct": 1.5, "take_profit_pct": 8.0, "stop_loss_pct": 5.0}
    if table_name == "uptrend":
        return {"entry_limit_pct": 1.5, "take_profit_pct": 8.0, "stop_loss_pct": 5.0}
    raise ValueError(table_name)


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

    static_annual = float(static_row.get("annual_return_pct", np.nan))
    static_quarter = float(static_row.get("quarter_return_pct", np.nan))
    static_r2 = float(static_row.get("trend_r2", np.nan))
    static_pmr = float(static_row.get("positive_month_ratio_pct", np.nan))
    static_persist = float(static_row.get("persistence_20d_pct", np.nan))
    static_sector = float(static_row.get("sector_adjusted_per_score", np.nan))
    static_per = float(static_row.get("ocr_per", np.nan))

    signal = (
        annual_return_pct >= static_annual
        and quarter_return_pct >= static_quarter
        and trend_r2 >= static_r2
        and positive_month_ratio_pct >= static_pmr
        and persistence_20d_pct >= static_persist
        and max_drawdown_pct >= -28.0
        and not pd.isna(static_sector)
        and not pd.isna(static_per)
    )

    signal_score = (
        0.30 * min(max(annual_return_pct / 100.0, 0.0), 1.0)
        + 0.20 * min(max(quarter_return_pct / 40.0, 0.0), 1.0)
        + 0.20 * trend_r2
        + 0.10 * min(max(persistence_20d_pct / 100.0, 0.0), 1.0)
        + 0.10 * min(max(positive_month_ratio_pct / 100.0, 0.0), 1.0)
        + 0.10 * min(max((28.0 + max_drawdown_pct) / 28.0, 0.0), 1.0)
    )

    return signal, {
        "signal_score": signal_score,
        "annual_return_pct": annual_return_pct,
        "quarter_return_pct": quarter_return_pct,
        "trend_r2": trend_r2,
    }


def run_backtest(selected_csv: Path, output_dir: Path, start_date: str, end_date: str, initial_capital: float) -> dict:
    candidates = load_candidates(selected_csv)
    phase_map = load_phase_map(PHASE_CSV)
    tables = build_selection_tables(candidates)

    tickers = sorted(set().union(*[set(tbl["ticker"]) for tbl in tables.values()]))
    price_map: dict[str, pd.DataFrame] = {}
    all_dates: set[pd.Timestamp] = set()
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
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
    cash = float(initial_capital)
    positions: dict[str, dict] = {}
    prev_signal = {ticker: False for ticker in price_map}
    trades: list[dict] = []
    equity_rows: list[dict] = []
    buy_count = 0

    static_lookups = {name: tbl.set_index("ticker").to_dict("index") for name, tbl in tables.items()}

    for date in dates:
        phase = phase_map.loc[phase_map.index <= date].tail(1)
        phase_name = str(phase.iloc[0]) if not phase.empty else "normal"
        table_name = table_for_phase(phase_name)

        signal_today: dict[str, bool] = {}
        score_today: dict[str, float] = {}
        basis_date_today: dict[str, pd.Timestamp] = {}
        table_used_today: dict[str, str] = {}

        if table_name is not None:
            table_df = tables[table_name]
            lookup = static_lookups[table_name]
            rule = trade_rule_for_table(table_name)
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
                table_used_today[ticker] = table_name

        for ticker in list(positions.keys()):
            df = price_map[ticker]
            if date not in df.index:
                continue
            day = df.loc[date]
            close_price = float(day["Close"])
            ret = close_price / positions[ticker]["entry_price"] - 1.0
            table_name_for_position = positions[ticker]["table_name"]
            rule = trade_rule_for_table(table_name_for_position)
            exit_reason = None
            if ret >= rule["take_profit_pct"] / 100.0:
                exit_reason = "take_profit"
            elif ret <= -rule["stop_loss_pct"] / 100.0:
                exit_reason = "stop_loss"
            elif not signal_today.get(ticker, False):
                exit_reason = "sell_signal"
            if exit_reason is not None:
                shares = positions[ticker]["shares"]
                cash += shares * close_price
                trades.append(
                    {
                        "date": date.date().isoformat(),
                        "ticker": ticker,
                        "sector": positions[ticker]["sector"],
                        "action": "SELL",
                        "price": close_price,
                        "shares": shares,
                        "cash_after": cash,
                        "table_name": table_name_for_position,
                        "phase": phase_name,
                        "reason": exit_reason,
                    }
                )
                del positions[ticker]

        if table_name is not None:
            rule = trade_rule_for_table(table_name)
            candidates_today = []
            for ticker, sig in signal_today.items():
                if ticker in positions:
                    continue
                if sig and not prev_signal.get(ticker, False):
                    candidates_today.append((ticker, score_today.get(ticker, 0.0)))
            candidates_today.sort(key=lambda x: x[1], reverse=True)

            remaining = len(candidates_today)
            for ticker, signal_score in candidates_today:
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
                    positions[ticker] = {
                        "shares": shares,
                        "entry_price": fill_price,
                        "sector": static_lookups[table_name][ticker].get("sector", ""),
                        "table_name": table_name,
                    }
                    trades.append(
                        {
                            "date": date.date().isoformat(),
                            "ticker": ticker,
                            "sector": static_lookups[table_name][ticker].get("sector", ""),
                            "action": "BUY",
                            "price": fill_price,
                            "shares": shares,
                            "cash_after": cash,
                            "table_name": table_name,
                            "phase": phase_name,
                            "reason": "entry",
                            "signal_score": signal_score,
                        }
                    )
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

    latest_date = max(dates) if dates else pd.Timestamp(end_date)
    for ticker in list(positions.keys()):
        df = price_map[ticker]
        usable_idx = df.index[df.index <= latest_date]
        if len(usable_idx):
            cash += positions[ticker]["shares"] * float(df.loc[usable_idx[-1], "Close"])
        del positions[ticker]

    equity_df = pd.DataFrame(equity_rows)
    trade_df = pd.DataFrame(trades)
    if not equity_df.empty:
        drawdown = equity_df["equity"].astype(float) / equity_df["equity"].astype(float).cummax() - 1.0
        max_dd_pct = float(drawdown.min()) * 100.0
        equity_df["pnl_day"] = equity_df["equity"].diff().fillna(equity_df["equity"] - initial_capital)
        phase_pnl = equity_df.groupby("phase")["pnl_day"].sum().to_dict()
    else:
        max_dd_pct = 0.0
        phase_pnl = {}

    summary = {
        "method_name": "phase_table_switch",
        "selected_csv": str(selected_csv),
        "period_start": start_date,
        "period_end": end_date,
        "initial_capital": initial_capital,
        "final_capital": float(cash),
        "total_return_pct": (float(cash) / float(initial_capital) - 1.0) * 100.0,
        "num_buys": int(buy_count),
        "max_drawdown_pct": max_dd_pct,
        "phase_pnl": phase_pnl,
        "table_counts": {name: int(len(tbl)) for name, tbl in tables.items()},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    trade_df.to_csv(output_dir / "trade_log.csv", index=False, encoding="utf-8-sig")
    equity_df.to_csv(output_dir / "equity_curve.csv", index=False, encoding="utf-8-sig")
    for name, tbl in tables.items():
        tbl.to_csv(output_dir / f"selection_table_{name}.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest table-switch method using three selection tables.")
    parser.add_argument("--selected-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, required=True)
    parser.add_argument("--initial-capital", type=float, default=3_000_000.0)
    args = parser.parse_args()
    run_backtest(args.selected_csv, args.output_dir, args.start_date, args.end_date, args.initial_capital)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

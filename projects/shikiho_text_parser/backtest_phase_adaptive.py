from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PRICE_DIR = ROOT / "data" / "prices"
PHASE_CSV = ROOT / "projects" / "shikiho_text_parser" / "output" / "nikkei_market_phase_map" / "nikkei_market_phase_daily_labels.csv"


def load_selected(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    num_cols = [
        "forecast_per",
        "sector_adjusted_per_score",
        "shikiho_score_overall",
        "headline_score_raw",
        "promising_score",
        "annual_return_pct",
        "quarter_return_pct",
        "trend_r2",
        "max_drawdown_pct",
        "positive_month_ratio_pct",
        "persistence_20d_pct",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.drop_duplicates(subset=["ticker"], keep="first").copy()


def compute_metrics(df: pd.DataFrame, date: pd.Timestamp) -> dict:
    hist = df.loc[:date].copy()
    if len(hist) < 180:
        return {}
    trailing = hist.tail(min(252, len(hist))).copy()
    qdf = hist.tail(min(63, len(hist))).copy()
    if len(trailing) < 120 or len(qdf) < 20:
        return {}

    annual_return_pct = (trailing["Close"].iloc[-1] / trailing["Close"].iloc[0] - 1.0) * 100.0
    quarter_return_pct = (qdf["Close"].iloc[-1] / qdf["Close"].iloc[0] - 1.0) * 100.0

    close = trailing["Close"].astype(float)
    y = np.log(close.replace(0, np.nan).dropna().values)
    if len(y) < 20:
        return {}
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    trend_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    running_max = close.cummax()
    max_drawdown_pct = float(((close / running_max) - 1.0).min()) * 100.0
    monthly_close = trailing["Close"].resample("ME").last().dropna()
    monthly_ret = monthly_close.pct_change().dropna()
    positive_month_ratio_pct = float((monthly_ret > 0).mean()) * 100.0 if len(monthly_ret) else 0.0

    ma20 = trailing["Close"].rolling(20).mean()
    ma60 = trailing["Close"].rolling(60).mean()
    last20 = trailing.iloc[-20:].copy()
    ma20_last20 = ma20.reindex(last20.index)
    ma60_last20 = ma60.reindex(last20.index)
    persistence_20d_pct = float(((last20["Close"] > ma20_last20) & (ma20_last20 > ma60_last20)).mean()) * 100.0

    return {
        "annual_return_pct": annual_return_pct,
        "quarter_return_pct": quarter_return_pct,
        "trend_r2": trend_r2,
        "max_drawdown_pct": max_drawdown_pct,
        "positive_month_ratio_pct": positive_month_ratio_pct,
        "persistence_20d_pct": persistence_20d_pct,
    }


def rule_name_for_phase(phase: str) -> str:
    if phase in {"uptrend", "normal"}:
        return "condition2"
    if phase in {"stable", "overheated_range"}:
        return "breakout_1p5"
    if phase in {"reversal_up", "capitulation_end"}:
        return "q3_post_high_vol"
    return "no_trade"


def eval_signal(rule_name: str, metrics: dict, static_row: dict) -> tuple[bool, float]:
    if not metrics or rule_name == "no_trade":
        return False, 0.0

    annual_return_pct = metrics["annual_return_pct"]
    quarter_return_pct = metrics["quarter_return_pct"]
    trend_r2 = metrics["trend_r2"]
    max_drawdown_pct = metrics["max_drawdown_pct"]
    positive_month_ratio_pct = metrics["positive_month_ratio_pct"]
    persistence_20d_pct = metrics["persistence_20d_pct"]

    head_score = static_row.get("headline_score_raw")
    shikiho_score = static_row.get("shikiho_score_overall")
    sector_per_score = static_row.get("sector_adjusted_per_score")
    forecast_per = static_row.get("forecast_per")
    promising_score = static_row.get("promising_score")

    common_signal_score = (
        0.28 * min(max((annual_return_pct - 20.0) / 160.0, 0.0), 1.0)
        + 0.22 * min(max((quarter_return_pct - 10.0) / 50.0, 0.0), 1.0)
        + 0.20 * trend_r2
        + 0.12 * min(max(persistence_20d_pct / 100.0, 0.0), 1.0)
        + 0.08 * min(max(positive_month_ratio_pct / 100.0, 0.0), 1.0)
        + 0.10 * min(max((28.0 + max_drawdown_pct) / 28.0, 0.0), 1.0)
    )

    if rule_name == "condition2":
        signal = (
            ((promising_score if not pd.isna(promising_score) else 0.0) >= 0.66)
            and 20.0 <= annual_return_pct <= 180.0
            and 10.0 <= quarter_return_pct <= 60.0
            and trend_r2 >= 0.50
            and persistence_20d_pct >= 55.0
            and positive_month_ratio_pct >= 60.0
            and max_drawdown_pct >= -28.0
            and (((head_score if not pd.isna(head_score) else 0) >= 0) or ((shikiho_score if not pd.isna(shikiho_score) else 0) >= 3))
            and (((sector_per_score if not pd.isna(sector_per_score) else 0) >= 0.55) or pd.isna(forecast_per))
            and ((forecast_per <= 20.0) if not pd.isna(forecast_per) else True)
        )
        return signal, common_signal_score

    if rule_name == "breakout_1p5":
        signal = (
            ((promising_score if not pd.isna(promising_score) else 0.0) >= 0.72)
            and 20.0 <= annual_return_pct <= 180.0
            and 10.0 <= quarter_return_pct <= 60.0
            and trend_r2 >= 0.60
            and persistence_20d_pct >= 55.0
            and positive_month_ratio_pct >= 60.0
            and max_drawdown_pct >= -28.0
            and (((head_score if not pd.isna(head_score) else 0) >= 0) or ((shikiho_score if not pd.isna(shikiho_score) else 0) >= 3))
            and (((sector_per_score if not pd.isna(sector_per_score) else 0) >= 0.55) or pd.isna(forecast_per))
            and ((forecast_per <= 20.0) if not pd.isna(forecast_per) else True)
        )
        return signal, common_signal_score

    if rule_name == "q3_post_high_vol":
        signal = (
            trend_r2 >= 0.60
            and annual_return_pct >= 25.0
            and quarter_return_pct >= 10.0
            and positive_month_ratio_pct >= 50.0
            and persistence_20d_pct >= 55.0
            and max_drawdown_pct >= -28.0
            and ((sector_per_score >= 0.35) if not pd.isna(sector_per_score) else False)
            and ((forecast_per <= 20.0) if not pd.isna(forecast_per) else False)
        )
        return signal, common_signal_score

    return False, 0.0


def load_phase_map(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.set_index("Date")["phase"].astype(str)


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest selected universe with Nikkei phase-adaptive methods.")
    parser.add_argument("--selected-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, required=True)
    parser.add_argument("--initial-capital", type=float, default=3_000_000.0)
    parser.add_argument("--price-dir", type=Path, default=PRICE_DIR)
    parser.add_argument("--phase-csv", type=Path, default=PHASE_CSV)
    parser.add_argument("--entry-limit-pct", type=float, default=1.5)
    parser.add_argument("--take-profit-pct", type=float, default=8.0)
    parser.add_argument("--stop-loss-pct", type=float, default=5.0)
    parser.add_argument("--method-name", type=str, default="phase_adaptive")
    args = parser.parse_args()

    selected = load_selected(args.selected_csv)
    static_lookup = selected.set_index("ticker").to_dict("index")
    phase_map = load_phase_map(args.phase_csv)

    start_date = pd.Timestamp(args.start_date)
    end_date = pd.Timestamp(args.end_date)
    price_map: dict[str, pd.DataFrame] = {}
    all_dates: set[pd.Timestamp] = set()

    for ticker in selected["ticker"]:
        price_path = args.price_dir / f"{ticker.replace('.', '_')}.csv"
        if not price_path.exists():
            continue
        df = pd.read_csv(price_path)
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
    if not dates:
        raise SystemExit("No tradable dates found.")
    latest_date = max(dates)

    cash = float(args.initial_capital)
    positions: dict[str, dict] = {}
    trades: list[dict] = []
    equity_curve: list[dict] = []
    prev_signal = {ticker: False for ticker in price_map}

    for date in dates:
        phase = phase_map.loc[phase_map.index <= date].tail(1)
        phase_name = str(phase.iloc[0]) if not phase.empty else "normal"
        rule_name = rule_name_for_phase(phase_name)

        signal_today: dict[str, bool] = {}
        metrics_today: dict[str, dict] = {}

        for ticker, df in price_map.items():
            if date not in df.index:
                signal_today[ticker] = prev_signal.get(ticker, False)
                continue
            prev_idx = df.index[df.index < date]
            if len(prev_idx) == 0:
                signal_today[ticker] = False
                metrics_today[ticker] = {}
                continue
            signal_date = prev_idx[-1]
            metrics = compute_metrics(df, signal_date)
            sig, signal_score = eval_signal(rule_name, metrics, static_lookup[ticker])
            if metrics:
                metrics["signal_score"] = signal_score
                metrics["signal_basis_date"] = signal_date.date().isoformat()
                metrics["phase"] = phase_name
                metrics["rule_name"] = rule_name
            signal_today[ticker] = sig
            metrics_today[ticker] = metrics

        for ticker in list(positions.keys()):
            df = price_map[ticker]
            if date not in df.index:
                continue
            price = float(df.loc[date, "Close"])
            ret = price / positions[ticker]["entry_price"] - 1.0
            exit_reason = None
            if ret >= args.take_profit_pct / 100.0:
                exit_reason = "take_profit"
            elif ret <= -args.stop_loss_pct / 100.0:
                exit_reason = "stop_loss"
            elif not signal_today.get(ticker, False):
                exit_reason = "sell_signal"
            if exit_reason is not None:
                shares = positions[ticker]["shares"]
                cash += shares * price
                trades.append(
                    {
                        "date": date.date().isoformat(),
                        "ticker": ticker,
                        "simple_sector": positions[ticker]["simple_sector"],
                        "sector_33": positions[ticker]["sector_33"],
                        "action": "SELL",
                        "price": price,
                        "shares": shares,
                        "cash_after": cash,
                        "signal_score": metrics_today.get(ticker, {}).get("signal_score"),
                        "signal_basis_date": metrics_today.get(ticker, {}).get("signal_basis_date", ""),
                        "reason": exit_reason,
                        "nikkei_phase": phase_name,
                        "rule_name": rule_name,
                    }
                )
                del positions[ticker]

        candidates = []
        for ticker, sig in signal_today.items():
            if ticker in positions:
                continue
            if sig and not prev_signal.get(ticker, False):
                met = metrics_today.get(ticker, {})
                candidates.append(
                    {
                        "ticker": ticker,
                        "simple_sector": static_lookup[ticker].get("simple_sector", static_lookup[ticker].get("sector", "")),
                        "sector_33": static_lookup[ticker].get("sector_33", static_lookup[ticker].get("sector", "")),
                        "signal_score": float(met.get("signal_score", 0.0)),
                        "signal_basis_date": met.get("signal_basis_date", ""),
                    }
                )

        buy_list = sorted(candidates, key=lambda x: x["signal_score"], reverse=True)
        remaining = len(buy_list)
        for c in buy_list:
            df = price_map[c["ticker"]]
            day = df.loc[date]
            basis_date = pd.Timestamp(c["signal_basis_date"])
            prev_close = float(df.loc[basis_date, "Close"])
            trigger_price = prev_close * (1.0 + args.entry_limit_pct / 100.0)
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
                positions[c["ticker"]] = {
                    "shares": shares,
                    "entry_price": fill_price,
                    "entry_date": date,
                    "simple_sector": c["simple_sector"],
                    "sector_33": c["sector_33"],
                }
                trades.append(
                    {
                        "date": date.date().isoformat(),
                        "ticker": c["ticker"],
                        "simple_sector": c["simple_sector"],
                        "sector_33": c["sector_33"],
                        "action": "BUY",
                        "price": fill_price,
                        "shares": shares,
                        "cash_after": cash,
                        "signal_score": c["signal_score"],
                        "signal_basis_date": c["signal_basis_date"],
                        "reason": "buy_breakout_up",
                        "nikkei_phase": phase_name,
                        "rule_name": rule_name,
                    }
                )
            remaining -= 1

        market_value = 0.0
        for ticker, pos in positions.items():
            usable_idx = price_map[ticker].index[price_map[ticker].index <= date]
            if len(usable_idx):
                market_value += pos["shares"] * float(price_map[ticker].loc[usable_idx[-1], "Close"])
        equity_curve.append(
            {
                "date": date.date().isoformat(),
                "cash": cash,
                "market_value": market_value,
                "equity": cash + market_value,
                "positions": len(positions),
                "nikkei_phase": phase_name,
                "rule_name": rule_name,
            }
        )
        prev_signal = signal_today

    for ticker in list(positions.keys()):
        df = price_map[ticker]
        last_idx = df.index[df.index <= latest_date][-1]
        price = float(df.loc[last_idx, "Close"])
        shares = positions[ticker]["shares"]
        cash += shares * price
        phase = phase_map.loc[phase_map.index <= last_idx].tail(1)
        phase_name = str(phase.iloc[0]) if not phase.empty else "normal"
        rule_name = rule_name_for_phase(phase_name)
        trades.append(
            {
                "date": last_idx.date().isoformat(),
                "ticker": ticker,
                "simple_sector": positions[ticker]["simple_sector"],
                "sector_33": positions[ticker]["sector_33"],
                "action": "SELL_END",
                "price": price,
                "shares": shares,
                "cash_after": cash,
                "signal_score": np.nan,
                "signal_basis_date": "",
                "reason": "end_of_backtest",
                "nikkei_phase": phase_name,
                "rule_name": rule_name,
            }
        )
        del positions[ticker]

    trade_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve)
    if not equity_df.empty:
        equity_df["pnl_day"] = equity_df["equity"].astype(float).diff().fillna(equity_df["equity"].astype(float) - args.initial_capital)
    else:
        equity_df["pnl_day"] = pd.Series(dtype=float)

    phase_pnl = (
        equity_df.groupby("nikkei_phase", dropna=False)
        .agg(days=("date", "count"), pnl_total=("pnl_day", "sum"), avg_day_pnl=("pnl_day", "mean"))
        .reset_index()
        .sort_values("pnl_total", ascending=False)
    )

    period_rows = []
    if not equity_df.empty:
        equity_df["prev_phase"] = equity_df["nikkei_phase"].shift(1)
        equity_df["segment_id"] = (equity_df["nikkei_phase"] != equity_df["prev_phase"]).cumsum()
        for seg_id, seg in equity_df.groupby("segment_id"):
            start_eq = float(seg["equity"].iloc[0] - seg["pnl_day"].iloc[0])
            end_eq = float(seg["equity"].iloc[-1])
            period_rows.append(
                {
                    "segment_id": int(seg_id),
                    "nikkei_phase": seg["nikkei_phase"].iloc[0],
                    "rule_name": seg["rule_name"].iloc[0],
                    "start": seg["date"].iloc[0],
                    "end": seg["date"].iloc[-1],
                    "days": int(len(seg)),
                    "pnl": end_eq - start_eq,
                    "return_pct_on_segment_start_equity": (end_eq / start_eq - 1.0) * 100.0 if start_eq else np.nan,
                }
            )
    period_df = pd.DataFrame(period_rows)

    roundtrip = []
    open_buys = {}
    for _, tr in trade_df.iterrows():
        ticker = tr["ticker"]
        if tr["action"] == "BUY":
            open_buys[ticker] = tr
        elif tr["action"] in {"SELL", "SELL_END"} and ticker in open_buys:
            buy = open_buys.pop(ticker)
            pnl = (tr["price"] - buy["price"]) * tr["shares"]
            ret = (tr["price"] / buy["price"] - 1.0) * 100.0
            roundtrip.append(
                {
                    "ticker": ticker,
                    "simple_sector": tr["simple_sector"],
                    "sector_33": tr["sector_33"],
                    "buy_date": buy["date"],
                    "sell_date": tr["date"],
                    "shares": tr["shares"],
                    "buy_price": buy["price"],
                    "sell_price": tr["price"],
                    "pnl": pnl,
                    "return_pct": ret,
                    "exit_reason": tr["reason"],
                    "buy_phase": buy.get("nikkei_phase", ""),
                    "sell_phase": tr.get("nikkei_phase", ""),
                    "buy_rule_name": buy.get("rule_name", ""),
                }
            )
    roundtrip_df = pd.DataFrame(roundtrip)

    max_dd_pct = 0.0
    if not equity_df.empty:
        dd = equity_df["equity"].astype(float) / equity_df["equity"].astype(float).cummax() - 1.0
        max_dd_pct = float(dd.min()) * 100.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trade_df.to_csv(args.output_dir / "trade_log.csv", index=False, encoding="utf-8-sig")
    equity_df.to_csv(args.output_dir / "equity_curve.csv", index=False, encoding="utf-8-sig")
    roundtrip_df.to_csv(args.output_dir / "roundtrip_trades.csv", index=False, encoding="utf-8-sig")
    phase_pnl.to_csv(args.output_dir / "phase_pnl_summary.csv", index=False, encoding="utf-8-sig")
    period_df.to_csv(args.output_dir / "phase_period_pnl.csv", index=False, encoding="utf-8-sig")

    summary = {
        "method_name": args.method_name,
        "selected_csv": str(args.selected_csv),
        "phase_csv": str(args.phase_csv),
        "period_start": str(start_date.date()),
        "period_end": str(latest_date.date()),
        "initial_capital": float(args.initial_capital),
        "final_capital": float(cash),
        "total_return_pct": (float(cash) / float(args.initial_capital) - 1.0) * 100.0,
        "selected_count": int(selected["ticker"].nunique()),
        "num_buys": int((trade_df["action"] == "BUY").sum()) if not trade_df.empty else 0,
        "num_roundtrips": int(len(roundtrip_df)),
        "win_rate_pct": float((roundtrip_df["pnl"] > 0).mean() * 100.0) if len(roundtrip_df) else 0.0,
        "avg_roundtrip_return_pct": float(roundtrip_df["return_pct"].mean()) if len(roundtrip_df) else 0.0,
        "max_drawdown_pct": max_dd_pct,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

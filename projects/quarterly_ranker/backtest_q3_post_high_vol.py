from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
PRICE_DIR = ROOT_DIR / "data" / "prices"


def load_selected(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric_cols = [
        "score",
        "ocr_per",
        "sector_adjusted_per_score",
        "annual_return_pct",
        "quarter_return_pct",
        "trend_r2",
        "max_drawdown_pct",
        "positive_month_ratio_pct",
        "persistence_20d_pct",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.drop_duplicates(subset=["ticker"], keep="first").copy()


def calc_signal(
    df: pd.DataFrame,
    date: pd.Timestamp,
    static_row: dict,
    trend_r2_min: float,
    annual_return_min: float,
    quarter_return_min: float,
    positive_month_ratio_min: float,
    persistence_20d_min: float,
    sector_per_score_min: float,
    ocr_per_max: float,
) -> tuple[bool, dict]:
    hist = df.loc[:date].copy()
    if len(hist) < 180:
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

    sector_per_score = float(static_row.get("sector_adjusted_per_score", np.nan))
    ocr_per = float(static_row.get("ocr_per", np.nan))
    promising_score = float(static_row.get("score", np.nan))

    signal = (
        (not pd.isna(promising_score))
        and trend_r2 >= trend_r2_min
        and annual_return_pct >= annual_return_min
        and quarter_return_pct >= quarter_return_min
        and positive_month_ratio_pct >= positive_month_ratio_min
        and persistence_20d_pct >= persistence_20d_min
        and max_drawdown_pct >= -28.0
        and ((sector_per_score >= sector_per_score_min) if not pd.isna(sector_per_score) else False)
        and ((ocr_per <= ocr_per_max) if not pd.isna(ocr_per) else False)
    )

    signal_score = (
        0.28 * min(max((annual_return_pct - annual_return_min) / 160.0, 0.0), 1.0)
        + 0.22 * min(max((quarter_return_pct - quarter_return_min) / 50.0, 0.0), 1.0)
        + 0.20 * trend_r2
        + 0.12 * min(max(persistence_20d_pct / 100.0, 0.0), 1.0)
        + 0.08 * min(max(positive_month_ratio_pct / 100.0, 0.0), 1.0)
        + 0.10 * min(max((28.0 + max_drawdown_pct) / 28.0, 0.0), 1.0)
    )

    return signal, {
        "annual_return_pct": annual_return_pct,
        "quarter_return_pct": quarter_return_pct,
        "trend_r2": trend_r2,
        "max_drawdown_pct": max_drawdown_pct,
        "positive_month_ratio_pct": positive_month_ratio_pct,
        "persistence_20d_pct": persistence_20d_pct,
        "signal_score": signal_score,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest Q3 post-high-vol selected stocks.")
    parser.add_argument("--selected-csv", type=Path, required=True)
    parser.add_argument("--price-dir", type=Path, default=PRICE_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-date", type=str, default="2025-07-01")
    parser.add_argument("--end-date", type=str, default="2025-09-30")
    parser.add_argument("--initial-capital", type=float, default=3_000_000.0)
    parser.add_argument("--trend-r2-min", type=float, default=0.60)
    parser.add_argument("--annual-return-min", type=float, default=25.0)
    parser.add_argument("--quarter-return-min", type=float, default=10.0)
    parser.add_argument("--positive-month-ratio-min", type=float, default=50.0)
    parser.add_argument("--persistence-20d-min", type=float, default=55.0)
    parser.add_argument("--sector-per-score-min", type=float, default=0.35)
    parser.add_argument("--ocr-per-max", type=float, default=20.0)
    parser.add_argument("--take-profit-pct", type=float, default=8.0)
    parser.add_argument("--stop-loss-pct", type=float, default=5.0)
    parser.add_argument("--entry-limit-pct", type=float, default=1.5)
    args = parser.parse_args()

    selected = load_selected(args.selected_csv)
    static_lookup = selected.set_index("ticker").to_dict("index")

    start_date = pd.Timestamp(args.start_date)
    end_date = pd.Timestamp(args.end_date)
    price_map: dict[str, pd.DataFrame] = {}
    all_dates: set[pd.Timestamp] = set()
    for ticker in selected["ticker"]:
        path = args.price_dir / f"{ticker.replace('.', '_')}.csv"
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
    if not dates:
        raise SystemExit("No tradable dates found.")

    cash = float(args.initial_capital)
    positions: dict[str, dict] = {}
    trades: list[dict] = []
    equity_curve: list[dict] = []
    prev_signal = {ticker: False for ticker in price_map}
    latest_date = max(dates)

    for date in dates:
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
            sig, metrics = calc_signal(
                df=df,
                date=signal_date,
                static_row=static_lookup[ticker],
                trend_r2_min=args.trend_r2_min,
                annual_return_min=args.annual_return_min,
                quarter_return_min=args.quarter_return_min,
                positive_month_ratio_min=args.positive_month_ratio_min,
                persistence_20d_min=args.persistence_20d_min,
                sector_per_score_min=args.sector_per_score_min,
                ocr_per_max=args.ocr_per_max,
            )
            metrics["signal_basis_date"] = signal_date.date().isoformat()
            signal_today[ticker] = sig
            metrics_today[ticker] = metrics

        for ticker in list(positions.keys()):
            df = price_map[ticker]
            if date not in df.index:
                continue
            day = df.loc[date]
            price = float(day["Close"])
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
                        "sector": positions[ticker]["sector"],
                        "action": "SELL",
                        "price": price,
                        "shares": shares,
                        "cash_after": cash,
                        "signal_score": metrics_today.get(ticker, {}).get("signal_score"),
                        "signal_basis_date": metrics_today.get(ticker, {}).get("signal_basis_date", ""),
                        "reason": exit_reason,
                    }
                )
                del positions[ticker]

        candidates = []
        for ticker, sig in signal_today.items():
            if ticker in positions:
                continue
            if sig and not prev_signal.get(ticker, False):
                candidates.append(
                    {
                        "ticker": ticker,
                        "sector": static_lookup[ticker].get("sector", ""),
                        "signal_score": float(metrics_today.get(ticker, {}).get("signal_score", 0.0)),
                        "signal_basis_date": metrics_today.get(ticker, {}).get("signal_basis_date", ""),
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
                    "sector": c["sector"],
                }
                trades.append(
                    {
                        "date": date.date().isoformat(),
                        "ticker": c["ticker"],
                        "sector": c["sector"],
                        "action": "BUY",
                        "price": fill_price,
                        "shares": shares,
                        "cash_after": cash,
                        "signal_score": c["signal_score"],
                        "signal_basis_date": c["signal_basis_date"],
                        "reason": "buy_breakout_up",
                    }
                )
            remaining -= 1

        market_value = 0.0
        for ticker, pos in positions.items():
            if date in price_map[ticker].index:
                market_value += pos["shares"] * float(price_map[ticker].loc[date, "Close"])
        equity_curve.append(
            {
                "date": date.date().isoformat(),
                "cash": cash,
                "market_value": market_value,
                "equity": cash + market_value,
                "positions": len(positions),
            }
        )
        prev_signal = signal_today

    for ticker in list(positions.keys()):
        df = price_map[ticker]
        price = float(df.loc[latest_date, "Close"])
        shares = positions[ticker]["shares"]
        cash += shares * price
        trades.append(
            {
                "date": latest_date.date().isoformat(),
                "ticker": ticker,
                "sector": positions[ticker]["sector"],
                "action": "SELL_END",
                "price": price,
                "shares": shares,
                "cash_after": cash,
                "signal_score": np.nan,
                "signal_basis_date": "",
                "reason": "end_of_backtest",
            }
        )
        del positions[ticker]

    trade_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve)
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
                    "sector": tr["sector"],
                    "buy_date": buy["date"],
                    "sell_date": tr["date"],
                    "shares": tr["shares"],
                    "buy_price": buy["price"],
                    "sell_price": tr["price"],
                    "pnl": pnl,
                    "return_pct": ret,
                    "exit_reason": tr["reason"],
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

    summary = {
        "method_name": "q3_post_high_vol_method",
        "selected_csv": str(args.selected_csv),
        "period_start": str(start_date.date()),
        "period_end": str(latest_date.date()),
        "initial_capital": float(args.initial_capital),
        "final_capital": float(cash),
        "total_return_pct": (float(cash) / float(args.initial_capital) - 1.0) * 100.0,
        "selected_count": int(len(selected)),
        "num_buys": int((trade_df["action"] == "BUY").sum()) if not trade_df.empty else 0,
        "num_roundtrips": int(len(roundtrip_df)),
        "win_rate_pct": float((roundtrip_df["pnl"] > 0).mean() * 100.0) if len(roundtrip_df) else 0.0,
        "avg_roundtrip_return_pct": float(roundtrip_df["return_pct"].mean()) if len(roundtrip_df) else 0.0,
        "max_drawdown_pct": max_dd_pct,
        "trend_r2_min": args.trend_r2_min,
        "annual_return_min": args.annual_return_min,
        "quarter_return_min": args.quarter_return_min,
        "positive_month_ratio_min": args.positive_month_ratio_min,
        "persistence_20d_min": args.persistence_20d_min,
        "sector_per_score_min": args.sector_per_score_min,
        "ocr_per_max": args.ocr_per_max,
        "entry_mode": "breakout_up",
        "entry_limit_pct": args.entry_limit_pct,
        "take_profit_pct": args.take_profit_pct,
        "stop_loss_pct": args.stop_loss_pct,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

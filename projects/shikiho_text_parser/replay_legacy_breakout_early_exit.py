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

from projects.shikiho_text_parser.backtest_4q2_signals import compute_top_signal
from src.stock_signal import (
    download_additional_macro_features,
    download_dow_futures_feature,
    download_index_returns,
    download_nikkei_futures_night_feature,
)

PRICE_DIR = ROOT / "data" / "prices"
DEFAULT_ROUNDTRIPS = ROOT / "projects" / "shikiho_text_parser" / "output" / "4q2_breakout_backtest" / "breakout_1p5" / "roundtrip_trades.csv"
DEFAULT_SUMMARY = ROOT / "projects" / "shikiho_text_parser" / "output" / "4q2_breakout_backtest" / "breakout_1p5" / "summary.json"
DEFAULT_OUTDIR = ROOT / "projects" / "shikiho_text_parser" / "output" / "4q2_legacy_breakout_early_exit"


def load_price_map(roundtrips: pd.DataFrame, price_dir: Path) -> dict[str, pd.DataFrame]:
    price_map: dict[str, pd.DataFrame] = {}
    for ticker in sorted(roundtrips["ticker"].unique()):
        path = price_dir / f"{ticker.replace('.', '_')}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").set_index("Date")
        price_map[ticker] = df
    return price_map


def simulate_roundtrip(
    row: pd.Series,
    price_map: dict[str, pd.DataFrame],
    external_df: pd.DataFrame,
    dd_threshold: float,
    nk_threshold: float,
    ma_window: int,
    early_tp_buffer_pct: float,
    early_sl_buffer_pct: float,
    take_profit_pct: float,
) -> dict:
    ticker = row["ticker"]
    df = price_map[ticker]
    buy_date = pd.Timestamp(row["buy_date"])
    original_sell_date = pd.Timestamp(row["sell_date"])
    entry_price = float(row["buy_price"])
    shares = int(row["shares"])

    active_dates = df.index[(df.index >= buy_date) & (df.index <= original_sell_date)]
    chosen_exit_date = original_sell_date
    chosen_exit_price = float(row["sell_price"])
    chosen_exit_reason = str(row["exit_reason"])

    for date in active_dates[1:]:
        prev_idx = df.index[df.index < date]
        if len(prev_idx) == 0:
            continue
        prev_date = prev_idx[-1]
        prev_hist = df.loc[:prev_date].copy()
        highest_close = float(df.loc[(df.index >= buy_date) & (df.index <= prev_date), "Close"].max())
        top_flag, top_metrics = compute_top_signal(
            prev_hist=prev_hist,
            prev_date=prev_date,
            external_df=external_df,
            highest_close=highest_close,
            dd_threshold=dd_threshold,
            nk_threshold=nk_threshold,
            ma_window=ma_window,
        )
        if not top_flag:
            continue
        prev_close = float(prev_hist["Close"].iloc[-1])
        unrealized_prev_ret = (prev_close / entry_price - 1.0) * 100.0
        if unrealized_prev_ret >= take_profit_pct:
            continue

        day = df.loc[date]
        day_open = float(day["Open"]) if "Open" in day.index and not pd.isna(day["Open"]) else float(day["Close"])
        day_high = float(day["High"]) if "High" in day.index and not pd.isna(day["High"]) else float(day["Close"])
        day_low = float(day["Low"]) if "Low" in day.index and not pd.isna(day["Low"]) else float(day["Close"])

        if unrealized_prev_ret > 0:
            early_tp_price = prev_close * (1.0 + early_tp_buffer_pct / 100.0)
            if day_high >= early_tp_price:
                chosen_exit_date = date
                chosen_exit_price = max(day_open, early_tp_price)
                chosen_exit_reason = "early_take_profit_top"
                break
        else:
            early_sl_price = prev_close * (1.0 - early_sl_buffer_pct / 100.0)
            if day_low <= early_sl_price:
                chosen_exit_date = date
                chosen_exit_price = min(day_open, early_sl_price)
                chosen_exit_reason = "early_stop_top"
                break

    pnl = (chosen_exit_price - entry_price) * shares
    ret_pct = (chosen_exit_price / entry_price - 1.0) * 100.0
    out = row.to_dict()
    out["sell_date"] = chosen_exit_date.date().isoformat()
    out["sell_price"] = chosen_exit_price
    out["exit_reason"] = chosen_exit_reason
    out["pnl"] = pnl
    out["return_pct"] = ret_pct
    return out


def build_trade_log(roundtrips: pd.DataFrame, initial_capital: float) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    events: list[dict] = []
    for _, row in roundtrips.iterrows():
        events.append(
            {
                "date": row["buy_date"],
                "ticker": row["ticker"],
                "simple_sector": row.get("simple_sector", ""),
                "sector_33": row.get("sector_33", ""),
                "action": "BUY",
                "price": float(row["buy_price"]),
                "shares": int(row["shares"]),
                "reason": "buy_breakout_up",
            }
        )
        events.append(
            {
                "date": row["sell_date"],
                "ticker": row["ticker"],
                "simple_sector": row.get("simple_sector", ""),
                "sector_33": row.get("sector_33", ""),
                "action": "SELL",
                "price": float(row["sell_price"]),
                "shares": int(row["shares"]),
                "reason": row["exit_reason"],
            }
        )

    event_df = pd.DataFrame(events)
    event_df["date"] = pd.to_datetime(event_df["date"])
    action_order = {"BUY": 0, "SELL": 1}
    event_df["_ord"] = event_df["action"].map(action_order)
    event_df = event_df.sort_values(["date", "_ord", "ticker"]).drop(columns="_ord")

    cash = float(initial_capital)
    trade_log = []
    for _, ev in event_df.iterrows():
        price = float(ev["price"])
        shares = int(ev["shares"])
        if ev["action"] == "BUY":
            cash -= price * shares
        else:
            cash += price * shares
        rec = ev.to_dict()
        rec["date"] = pd.Timestamp(rec["date"]).date().isoformat()
        rec["cash_after"] = cash
        trade_log.append(rec)

    trade_df = pd.DataFrame(trade_log)
    return trade_df, event_df, cash


def build_equity_curve(
    event_df: pd.DataFrame,
    roundtrips: pd.DataFrame,
    price_map: dict[str, pd.DataFrame],
    initial_capital: float,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    cash = float(initial_capital)
    positions: dict[str, int] = {}
    idx = pd.date_range(start_date, end_date, freq="B")
    event_groups = {d: df.drop(columns=[]).to_dict("records") for d, df in event_df.groupby("date")}
    rows = []
    for date in idx:
        for ev in event_groups.get(date, []):
            ticker = ev["ticker"]
            shares = int(ev["shares"])
            price = float(ev["price"])
            if ev["action"] == "BUY":
                cash -= shares * price
                positions[ticker] = positions.get(ticker, 0) + shares
            else:
                cash += shares * price
                positions[ticker] = positions.get(ticker, 0) - shares
                if positions[ticker] <= 0:
                    positions.pop(ticker, None)
        market_value = 0.0
        for ticker, shares in positions.items():
            df = price_map[ticker]
            usable = df.index[df.index <= date]
            if len(usable) == 0:
                continue
            market_value += shares * float(df.loc[usable[-1], "Close"])
        rows.append(
            {
                "date": date.date().isoformat(),
                "cash": cash,
                "market_value": market_value,
                "equity": cash + market_value,
                "positions": len(positions),
            }
        )
    return pd.DataFrame(rows)


def max_drawdown_pct(equity_df: pd.DataFrame) -> float:
    eq = equity_df["equity"].astype(float)
    running_max = eq.cummax()
    return float(((eq / running_max) - 1.0).min()) * 100.0


def summarize(roundtrips: pd.DataFrame, equity_df: pd.DataFrame, final_capital: float, initial_capital: float, params: dict) -> dict:
    return {
        "initial_capital": float(initial_capital),
        "final_capital": float(final_capital),
        "total_return_pct": (float(final_capital) / float(initial_capital) - 1.0) * 100.0,
        "num_buys": int(len(roundtrips)),
        "num_roundtrips": int(len(roundtrips)),
        "win_rate_pct": float((roundtrips["pnl"] > 0).mean()) * 100.0 if len(roundtrips) else 0.0,
        "avg_roundtrip_return_pct": float(roundtrips["return_pct"].mean()) if len(roundtrips) else 0.0,
        "max_drawdown_pct": max_drawdown_pct(equity_df),
        **params,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay legacy breakout_1.5 trades with optional early exit.")
    parser.add_argument("--roundtrip-csv", type=Path, default=DEFAULT_ROUNDTRIPS)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--price-dir", type=Path, default=PRICE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--dd-threshold", type=float, default=4.0)
    parser.add_argument("--nk-threshold", type=float, default=-0.005)
    parser.add_argument("--ma-window", type=int, default=5)
    parser.add_argument("--early-tp-buffer-pct", type=float, default=1.0)
    parser.add_argument("--early-sl-buffer-pct", type=float, default=2.0)
    parser.add_argument("--disable-early-exit", action="store_true")
    args = parser.parse_args()

    roundtrips = pd.read_csv(args.roundtrip_csv)
    with args.summary_json.open("r", encoding="utf-8") as f:
        base_summary = json.load(f)
    initial_capital = float(base_summary["initial_capital"])
    take_profit_pct = float(base_summary["take_profit_pct"])
    start_date = pd.Timestamp(base_summary["period_start"])
    end_date = pd.Timestamp(base_summary["period_end"])

    price_map = load_price_map(roundtrips, args.price_dir)
    external_df = (
        download_index_returns(period="3y")
        .join(download_nikkei_futures_night_feature(), how="outer")
        .join(download_dow_futures_feature(period="3y"), how="outer")
        .join(download_additional_macro_features(period="3y"), how="outer")
        .sort_index()
        .fillna(0.0)
    )

    if args.disable_early_exit:
        updated = roundtrips.copy()
    else:
        updated = pd.DataFrame(
            [
                simulate_roundtrip(
                    row=row,
                    price_map=price_map,
                    external_df=external_df,
                    dd_threshold=args.dd_threshold,
                    nk_threshold=args.nk_threshold,
                    ma_window=args.ma_window,
                    early_tp_buffer_pct=args.early_tp_buffer_pct,
                    early_sl_buffer_pct=args.early_sl_buffer_pct,
                    take_profit_pct=take_profit_pct,
                )
                for _, row in roundtrips.iterrows()
            ]
        )

    trade_df, event_df, final_capital = build_trade_log(updated, initial_capital)
    equity_df = build_equity_curve(event_df, updated, price_map, initial_capital, start_date, end_date)
    summary = summarize(
        updated,
        equity_df,
        final_capital,
        initial_capital,
        {
            "dd_threshold": args.dd_threshold,
            "nk_threshold": args.nk_threshold,
            "ma_window": args.ma_window,
            "early_tp_buffer_pct": args.early_tp_buffer_pct,
            "early_sl_buffer_pct": args.early_sl_buffer_pct,
            "early_exit_enabled": not args.disable_early_exit,
        },
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    updated.to_csv(args.output_dir / "roundtrip_trades.csv", index=False, encoding="utf-8-sig")
    trade_df.to_csv(args.output_dir / "trade_log.csv", index=False, encoding="utf-8-sig")
    equity_df.to_csv(args.output_dir / "equity_curve.csv", index=False, encoding="utf-8-sig")
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from projects.shikiho_text_parser.backtest_4q2_signals import calc_signal, load_selected
from projects.shikiho_text_parser.paths import OUTPUT_DIR, PRICE_DIR
from src.stock_signal import (
    download_additional_macro_features,
    download_dow_futures_feature,
    download_index_returns,
    download_nikkei_futures_night_feature,
)


def build_external_df() -> pd.DataFrame:
    return (
        download_index_returns(period="3y")
        .join(download_nikkei_futures_night_feature(), how="outer")
        .join(download_dow_futures_feature(period="3y"), how="outer")
        .join(download_additional_macro_features(period="3y"), how="outer")
        .sort_index()
        .fillna(0.0)
    )


def load_price_map(selected: pd.DataFrame, price_dir: Path, end_date: pd.Timestamp) -> tuple[dict[str, pd.DataFrame], list[pd.Timestamp]]:
    price_map: dict[str, pd.DataFrame] = {}
    dates = set()
    for ticker in selected["ticker"]:
        price_path = price_dir / f"{ticker.replace('.', '_')}.csv"
        if not price_path.exists():
            continue
        df = pd.read_csv(price_path)
        if "Date" not in df.columns or "Close" not in df.columns:
            continue
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").set_index("Date")
        df = df[(df.index >= pd.Timestamp("2025-01-01")) & (df.index <= end_date)].copy()
        if df.empty:
            continue
        price_map[ticker] = df
        dates.update(df.index.tolist())
    return price_map, sorted(dates)


def build_signal_cache(
    selected: pd.DataFrame,
    price_map: dict[str, pd.DataFrame],
    all_dates: list[pd.Timestamp],
    external_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    promising_score_min: float,
    trend_r2_min: float,
    sector_per_score_min: float,
) -> dict[str, dict[pd.Timestamp, dict]]:
    static_lookup = selected.set_index("ticker").to_dict("index")
    cache: dict[str, dict[pd.Timestamp, dict]] = {}
    dates = [d for d in all_dates if start_date <= d <= end_date]
    for ticker, df in price_map.items():
        ticker_cache: dict[pd.Timestamp, dict] = {}
        for date in dates:
            if date not in df.index:
                ticker_cache[date] = {"signal": False, "metrics": {}}
                continue
            prev_idx = df.index[df.index < date]
            if len(prev_idx) == 0:
                ticker_cache[date] = {"signal": False, "metrics": {}}
                continue
            signal_date = prev_idx[-1]
            signal, metrics = calc_signal(
                df,
                signal_date,
                static_lookup[ticker],
                external_df,
                promising_score_min,
                trend_r2_min,
                sector_per_score_min,
                -1.0,
                0.0,
            )
            metrics["signal_basis_date"] = signal_date
            ticker_cache[date] = {"signal": signal, "metrics": metrics}
        cache[ticker] = ticker_cache
    return cache


def compute_top_signal(prev_hist: pd.DataFrame, prev_date: pd.Timestamp, external_df: pd.DataFrame, position: dict, dd_threshold: float, nk_threshold: float, ma_window: int) -> tuple[bool, dict]:
    close = prev_hist["Close"].astype(float)
    ma = close.rolling(ma_window).mean().iloc[-1] if len(close) >= ma_window else np.nan
    prev_close = float(close.iloc[-1])
    prev_close_m1 = float(close.iloc[-2]) if len(close) >= 2 else prev_close
    highest_close = max(float(position["highest_close"]), prev_close)
    drawdown_from_peak_pct = (prev_close / highest_close - 1.0) * 100.0
    nk_ret = 0.0
    if prev_date in external_df.index and "ret_n225" in external_df.columns:
        nk_ret = float(external_df.loc[prev_date, "ret_n225"])
    elif prev_date in external_df.index and "fut_nk_night_ret" in external_df.columns:
        nk_ret = float(external_df.loc[prev_date, "fut_nk_night_ret"])

    chart_top = (
        drawdown_from_peak_pct <= -dd_threshold
        and len(close) >= 2
        and prev_close < prev_close_m1
        and (pd.isna(ma) or prev_close < float(ma))
    )
    nikkei_weak = nk_ret <= nk_threshold
    return chart_top or nikkei_weak, {
        "drawdown_from_peak_pct": drawdown_from_peak_pct,
        "nikkei_ret": nk_ret,
        "chart_top": chart_top,
        "nikkei_weak": nikkei_weak,
        "prev_close": prev_close,
    }


def run_backtest(
    selected: pd.DataFrame,
    price_map: dict[str, pd.DataFrame],
    all_dates: list[pd.Timestamp],
    signal_cache: dict[str, dict[pd.Timestamp, dict]],
    external_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    initial_capital: float,
    promising_score_min: float,
    trend_r2_min: float,
    sector_per_score_min: float,
    entry_breakout_pct: float,
    take_profit_pct: float,
    stop_loss_pct: float,
    use_early_exit: bool,
    dd_threshold: float,
    nk_threshold: float,
    ma_window: int,
    early_take_profit_buffer_pct: float,
    early_stop_buffer_pct: float,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    dates = [d for d in all_dates if start_date <= d <= end_date]
    cash = float(initial_capital)
    positions: dict[str, dict] = {}
    prev_signal = {ticker: False for ticker in price_map}
    trades: list[dict] = []
    equity_rows: list[dict] = []

    for date in dates:
        signal_today: dict[str, bool] = {}
        metrics_today: dict[str, dict] = {}
        for ticker in price_map:
            cached = signal_cache.get(ticker, {}).get(date, {"signal": False, "metrics": {}})
            signal_today[ticker] = bool(cached.get("signal", False))
            metrics_today[ticker] = dict(cached.get("metrics", {}))

        for ticker in list(positions.keys()):
            df = price_map[ticker]
            if date not in df.index:
                continue
            day = df.loc[date]
            day_open = float(day["Open"]) if "Open" in day.index and not pd.isna(day["Open"]) else float(day["Close"])
            day_high = float(day["High"]) if "High" in day.index and not pd.isna(day["High"]) else float(day["Close"])
            day_low = float(day["Low"]) if "Low" in day.index and not pd.isna(day["Low"]) else float(day["Close"])
            day_close = float(day["Close"])

            pos = positions[ticker]
            pos["highest_close"] = max(float(pos["highest_close"]), day_close)

            exit_reason = None
            exit_price = None

            base_tp_price = float(pos["entry_price"]) * (1.0 + take_profit_pct / 100.0)
            base_sl_price = float(pos["entry_price"]) * (1.0 - stop_loss_pct / 100.0)

            if day_high >= base_tp_price:
                exit_reason = "take_profit_8pct"
                exit_price = max(day_open, base_tp_price)
            elif day_low <= base_sl_price:
                exit_reason = "stop_loss_5pct"
                exit_price = min(day_open, base_sl_price)
            elif use_early_exit:
                prev_idx = df.index[df.index < date]
                if len(prev_idx) > 0:
                    prev_date = prev_idx[-1]
                    prev_hist = df.loc[:prev_date].copy()
                    top_flag, top_metrics = compute_top_signal(
                        prev_hist=prev_hist,
                        prev_date=prev_date,
                        external_df=external_df,
                        position=pos,
                        dd_threshold=dd_threshold,
                        nk_threshold=nk_threshold,
                        ma_window=ma_window,
                    )
                    prev_close = float(prev_hist["Close"].iloc[-1])
                    unrealized_prev_ret = (prev_close / float(pos["entry_price"]) - 1.0) * 100.0
                    if top_flag and unrealized_prev_ret < take_profit_pct:
                        if unrealized_prev_ret > 0:
                            early_tp_price = prev_close * (1.0 + early_take_profit_buffer_pct / 100.0)
                            if day_high >= early_tp_price:
                                exit_reason = "early_take_profit_top"
                                exit_price = max(day_open, early_tp_price)
                        else:
                            early_sl_price = prev_close * (1.0 - early_stop_buffer_pct / 100.0)
                            if day_low <= early_sl_price:
                                exit_reason = "early_stop_top"
                                exit_price = min(day_open, early_sl_price)
                        if exit_reason is not None:
                            pos["top_metrics"] = top_metrics

            if exit_reason is not None and exit_price is not None:
                shares = pos["shares"]
                cash += shares * exit_price
                trades.append(
                    {
                        "date": date.date().isoformat(),
                        "ticker": ticker,
                        "action": "SELL",
                        "price": exit_price,
                        "shares": shares,
                        "reason": exit_reason,
                        "cash_after": cash,
                    }
                )
                del positions[ticker]

        buy_list = []
        for ticker, sig in signal_today.items():
            if ticker in positions:
                continue
            if sig and not prev_signal.get(ticker, False):
                df = price_map[ticker]
                if date not in df.index:
                    continue
                signal_basis_date = metrics_today[ticker].get("signal_basis_date")
                prev_close = float(df.loc[signal_basis_date, "Close"])
                trigger_price = prev_close * (1.0 + entry_breakout_pct / 100.0)
                day = df.loc[date]
                day_open = float(day["Open"]) if "Open" in day.index and not pd.isna(day["Open"]) else float(day["Close"])
                day_high = float(day["High"]) if "High" in day.index and not pd.isna(day["High"]) else float(day["Close"])
                if day_high < trigger_price:
                    continue
                fill_price = max(day_open, trigger_price)
                buy_list.append((ticker, fill_price, metrics_today[ticker].get("signal_score", 0.0)))

        buy_list.sort(key=lambda x: x[2], reverse=True)
        remaining = len(buy_list)
        for ticker, fill_price, signal_score in buy_list:
            alloc = cash / remaining if remaining > 0 else 0.0
            lot_cost = fill_price * 100
            shares = int(alloc // lot_cost) * 100
            if shares >= 100 and shares * fill_price <= cash:
                cash -= shares * fill_price
                positions[ticker] = {
                    "shares": shares,
                    "entry_price": fill_price,
                    "entry_date": date,
                    "highest_close": float(fill_price),
                }
                trades.append(
                    {
                        "date": date.date().isoformat(),
                        "ticker": ticker,
                        "action": "BUY",
                        "price": fill_price,
                        "shares": shares,
                        "reason": "entry_breakout",
                        "signal_score": signal_score,
                        "cash_after": cash,
                    }
                )
            remaining -= 1

        market_value = 0.0
        for ticker, pos in positions.items():
            df = price_map[ticker]
            if date not in df.index:
                continue
            market_value += pos["shares"] * float(df.loc[date, "Close"])
        equity_rows.append(
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
        last_idx = df.index[df.index <= end_date][-1]
        last_close = float(df.loc[last_idx, "Close"])
        shares = positions[ticker]["shares"]
        cash += shares * last_close
        trades.append(
            {
                "date": last_idx.date().isoformat(),
                "ticker": ticker,
                "action": "SELL_END",
                "price": last_close,
                "shares": shares,
                "reason": "end_of_backtest",
                "cash_after": cash,
            }
        )
        del positions[ticker]

    trade_df = pd.DataFrame(trades)
    roundtrip_rows = []
    open_buys: dict[str, dict] = {}
    for _, tr in trade_df.iterrows():
        if tr["action"] == "BUY":
            open_buys[tr["ticker"]] = tr
        elif tr["action"] in {"SELL", "SELL_END"} and tr["ticker"] in open_buys:
            buy = open_buys.pop(tr["ticker"])
            pnl = (float(tr["price"]) - float(buy["price"])) * int(tr["shares"])
            ret = (float(tr["price"]) / float(buy["price"]) - 1.0) * 100.0
            roundtrip_rows.append(
                {
                    "ticker": tr["ticker"],
                    "buy_date": buy["date"],
                    "sell_date": tr["date"],
                    "buy_price": float(buy["price"]),
                    "sell_price": float(tr["price"]),
                    "shares": int(tr["shares"]),
                    "pnl": pnl,
                    "return_pct": ret,
                    "exit_reason": tr["reason"],
                }
            )
    roundtrip_df = pd.DataFrame(roundtrip_rows)
    equity_df = pd.DataFrame(equity_rows)
    max_dd_pct = 0.0
    if not equity_df.empty:
        eq = equity_df["equity"].astype(float)
        max_dd_pct = float((eq / eq.cummax() - 1.0).min()) * 100.0

    summary = {
        "initial_capital": initial_capital,
        "final_capital": float(cash),
        "total_return_pct": (float(cash) / float(initial_capital) - 1.0) * 100.0,
        "num_buys": int((trade_df["action"] == "BUY").sum()) if not trade_df.empty else 0,
        "num_roundtrips": int(len(roundtrip_df)),
        "win_rate_pct": float((roundtrip_df["pnl"] > 0).mean() * 100.0) if len(roundtrip_df) else 0.0,
        "avg_roundtrip_return_pct": float(roundtrip_df["return_pct"].mean()) if len(roundtrip_df) else 0.0,
        "max_drawdown_pct": max_dd_pct,
    }
    return summary, trade_df, roundtrip_df


def main() -> int:
    parser = argparse.ArgumentParser(description="Search early-exit rules for the 4Q-2 universe.")
    parser.add_argument("--selected-csv", type=Path, default=OUTPUT_DIR / "4q2_selection" / "4q2_selected_candidates.csv")
    parser.add_argument("--price-dir", type=Path, default=PRICE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "4q2_early_exit_search")
    parser.add_argument("--start-date", type=str, default="2026-01-01")
    parser.add_argument("--end-date", type=str, default="2026-03-16")
    parser.add_argument("--initial-capital", type=float, default=3_000_000.0)
    args = parser.parse_args()

    selected = load_selected(args.selected_csv).drop_duplicates(subset=["ticker"], keep="first").copy()
    end_date = pd.Timestamp(args.end_date)
    start_date = pd.Timestamp(args.start_date)
    external_df = build_external_df()
    price_map, all_dates = load_price_map(selected, args.price_dir, end_date)
    signal_cache = build_signal_cache(
        selected=selected,
        price_map=price_map,
        all_dates=all_dates,
        external_df=external_df,
        start_date=start_date,
        end_date=end_date,
        promising_score_min=0.66,
        trend_r2_min=0.50,
        sector_per_score_min=0.55,
    )

    base_summary, base_trades, base_roundtrips = run_backtest(
        selected=selected,
        price_map=price_map,
        all_dates=all_dates,
        signal_cache=signal_cache,
        external_df=external_df,
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.initial_capital,
        promising_score_min=0.66,
        trend_r2_min=0.50,
        sector_per_score_min=0.55,
        entry_breakout_pct=1.5,
        take_profit_pct=8.0,
        stop_loss_pct=5.0,
        use_early_exit=False,
        dd_threshold=0.0,
        nk_threshold=0.0,
        ma_window=5,
        early_take_profit_buffer_pct=0.0,
        early_stop_buffer_pct=0.0,
    )

    results = [
        {
            "rule_name": "baseline_8tp_5sl",
            "use_early_exit": False,
            "dd_threshold": None,
            "nk_threshold": None,
            "ma_window": None,
            "early_take_profit_buffer_pct": None,
            "early_stop_buffer_pct": None,
            **base_summary,
        }
    ]

    best_summary = results[0]
    best_trade_df = base_trades
    best_roundtrip_df = base_roundtrips

    for dd_threshold, nk_threshold, ma_window, tp_buffer, stop_buffer in product(
        [2.0, 3.0, 4.0, 5.0],
        [-0.015, -0.01, -0.005, 0.0],
        [5, 7, 10],
        [0.0, 0.5, 1.0],
        [0.5, 1.0, 1.5, 2.0],
    ):
        summary, trade_df, roundtrip_df = run_backtest(
            selected=selected,
            price_map=price_map,
            all_dates=all_dates,
            signal_cache=signal_cache,
            external_df=external_df,
            start_date=start_date,
            end_date=end_date,
            initial_capital=args.initial_capital,
            promising_score_min=0.66,
            trend_r2_min=0.50,
            sector_per_score_min=0.55,
            entry_breakout_pct=1.5,
            take_profit_pct=8.0,
            stop_loss_pct=5.0,
            use_early_exit=True,
            dd_threshold=dd_threshold,
            nk_threshold=nk_threshold,
            ma_window=ma_window,
            early_take_profit_buffer_pct=tp_buffer,
            early_stop_buffer_pct=stop_buffer,
        )
        row = {
            "rule_name": f"early_dd{dd_threshold}_nk{nk_threshold}_ma{ma_window}_tpb{tp_buffer}_slb{stop_buffer}",
            "use_early_exit": True,
            "dd_threshold": dd_threshold,
            "nk_threshold": nk_threshold,
            "ma_window": ma_window,
            "early_take_profit_buffer_pct": tp_buffer,
            "early_stop_buffer_pct": stop_buffer,
            **summary,
        }
        results.append(row)
        if float(summary["final_capital"]) > float(best_summary["final_capital"]):
            best_summary = row
            best_trade_df = trade_df
            best_roundtrip_df = roundtrip_df

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results).sort_values(["final_capital", "win_rate_pct"], ascending=[False, False])
    results_df.to_csv(out_dir / "early_exit_search_results.csv", index=False, encoding="utf-8-sig")
    best_trade_df.to_csv(out_dir / "best_trade_log.csv", index=False, encoding="utf-8-sig")
    best_roundtrip_df.to_csv(out_dir / "best_roundtrip_trades.csv", index=False, encoding="utf-8-sig")
    (out_dir / "best_summary.json").write_text(json.dumps(best_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"baseline": results[0], "best": best_summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

from projects.shikiho_text_parser.backtest_4q2_signals import load_selected, calc_signal  # noqa: E402
from src.stock_signal import (  # noqa: E402
    download_additional_macro_features,
    download_dow_futures_feature,
    download_index_returns,
    download_nikkei_futures_night_feature,
)


PRICE_DIR = ROOT / "data" / "prices"
OUTPUT_DIR = ROOT / "projects" / "shikiho_text_parser" / "output"
EXTERNAL_MARKET_DIR = ROOT / "data" / "external_market"


def nikkei_regime(n225_close: pd.Series, date: pd.Timestamp) -> str:
    return nikkei_regime_custom(n225_close, date, up_ret20_threshold=0.0, sideways_abs_ret20_threshold=0.05)


def nikkei_regime_custom(
    n225_close: pd.Series,
    date: pd.Timestamp,
    up_ret20_threshold: float,
    sideways_abs_ret20_threshold: float,
) -> str:
    hist = n225_close.loc[:date].dropna()
    if len(hist) < 21:
        return "down"
    close = float(hist.iloc[-1])
    ret20 = close / float(hist.iloc[-21]) - 1.0
    if ret20 > up_ret20_threshold:
        return "up"
    if abs(ret20) <= sideways_abs_ret20_threshold:
        return "sideways"
    return "down"


def build_nikkei_proxy(index_returns: pd.DataFrame, external_df: pd.DataFrame) -> pd.Series:
    if "ret_n225" in index_returns.columns and index_returns["ret_n225"].dropna().shape[0] > 0:
        ret = index_returns["ret_n225"].fillna(0.0).sort_index()
    elif "fut_nk_night_ret" in external_df.columns and external_df["fut_nk_night_ret"].dropna().shape[0] > 0:
        ret = external_df["fut_nk_night_ret"].fillna(0.0).sort_index()
    else:
        raise ValueError("No usable Nikkei proxy series found")
    proxy = (1.0 + ret).cumprod() * 100.0
    proxy.name = "n225_proxy_close"
    return proxy


def load_local_nikkei_close() -> pd.Series:
    for name in ("nikkei225_daily.csv", "nikkei_futures_daily.csv"):
        path = EXTERNAL_MARKET_DIR / name
        if not path.exists():
            continue
        df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
        if "Close" in df.columns and not df["Close"].dropna().empty:
            s = df.set_index("Date")["Close"].astype(float)
            s.name = f"local_{name}"
            return s
    raise ValueError("No local Nikkei market csv found")


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest Q4 universe with Nikkei regime switching.")
    parser.add_argument("--selected-csv", type=Path, required=True)
    parser.add_argument("--price-dir", type=Path, default=PRICE_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition-name", type=str, default="method_regime_switch")
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, required=True)
    parser.add_argument("--initial-capital", type=float, default=3_000_000.0)
    parser.add_argument("--take-profit-pct", type=float, default=8.0)
    parser.add_argument("--stop-loss-pct", type=float, default=5.0)
    parser.add_argument("--entry-limit-pct", type=float, default=1.5)
    parser.add_argument("--entry-mode", type=str, default="breakout_up", choices=["limit_down", "breakout_up"])
    parser.add_argument("--up-ret20-threshold", type=float, default=0.0)
    parser.add_argument("--sideways-abs-ret20-threshold", type=float, default=0.07)
    args = parser.parse_args()

    selected = load_selected(args.selected_csv).drop_duplicates(subset=["ticker"], keep="first").copy()
    static_lookup = selected.set_index("ticker").to_dict("index")

    index_returns = download_index_returns(period="3y")
    nk_futures = download_nikkei_futures_night_feature()
    dow_futures = download_dow_futures_feature(period="3y")
    extra_macro = download_additional_macro_features(period="3y")
    external_df = (
        index_returns.join(nk_futures, how="outer")
        .join(dow_futures, how="outer")
        .join(extra_macro, how="outer")
        .sort_index()
        .fillna(0.0)
    )
    try:
        n225_close = load_local_nikkei_close()
    except Exception:
        n225_close = build_nikkei_proxy(index_returns, external_df)

    start_date = pd.Timestamp(args.start_date)
    end_date = pd.Timestamp(args.end_date)
    price_map: dict[str, pd.DataFrame] = {}
    all_dates = set()
    for ticker in selected["ticker"]:
        path = args.price_dir / f"{ticker.replace('.', '_')}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").set_index("Date")
        df = df[(df.index >= pd.Timestamp("2025-01-01")) & (df.index <= end_date)].copy()
        if df.empty:
            continue
        price_map[ticker] = df
        all_dates.update(df[df.index >= start_date].index.tolist())

    dates = sorted(all_dates)
    if not dates:
        raise SystemExit("No tradable dates found.")

    regime_params = {
        "up": {"promising_score_min": 0.66, "trend_r2_min": 0.50, "sector_per_score_min": 0.55},
        "sideways": {"promising_score_min": 0.72, "trend_r2_min": 0.60, "sector_per_score_min": 0.55},
    }
    regime_counts = {"up": 0, "sideways": 0, "down": 0}

    cash = float(args.initial_capital)
    positions: dict[str, dict] = {}
    trades: list[dict] = []
    equity_curve: list[dict] = []
    prev_signal = {ticker: False for ticker in price_map}

    for date in dates:
        regime = nikkei_regime_custom(
            n225_close,
            date - pd.Timedelta(days=1),
            up_ret20_threshold=args.up_ret20_threshold,
            sideways_abs_ret20_threshold=args.sideways_abs_ret20_threshold,
        )
        regime_counts[regime] += 1

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
            if regime in regime_params:
                p = regime_params[regime]
                sig, metrics = calc_signal(
                    df,
                    signal_date,
                    static_lookup[ticker],
                    external_df,
                    p["promising_score_min"],
                    p["trend_r2_min"],
                    p["sector_per_score_min"],
                    -1.0,
                    0.0,
                )
                metrics["signal_basis_date"] = signal_date.date().isoformat()
            else:
                sig, metrics = False, {"signal_basis_date": signal_date.date().isoformat()}
            signal_today[ticker] = sig
            metrics_today[ticker] = metrics

        for ticker in list(positions.keys()):
            df = price_map[ticker]
            if date not in df.index:
                continue
            day = df.loc[date]
            price = float(day["Close"])
            entry_price = positions[ticker]["entry_price"]
            ret = price / entry_price - 1.0
            exit_reason = None
            exit_price = None
            if ret >= args.take_profit_pct / 100.0:
                exit_reason = "take_profit"
                exit_price = price
            elif ret <= -args.stop_loss_pct / 100.0:
                exit_reason = "stop_loss"
                exit_price = price
            elif not signal_today.get(ticker, False):
                exit_reason = "sell_signal"
                exit_price = price
            if exit_reason is not None:
                shares = positions[ticker]["shares"]
                cash += shares * float(exit_price)
                trades.append(
                    {
                        "date": date.date().isoformat(),
                        "ticker": ticker,
                        "simple_sector": positions[ticker]["simple_sector"],
                        "sector_33": positions[ticker]["sector_33"],
                        "action": "SELL",
                        "price": float(exit_price),
                        "shares": shares,
                        "cash_after": cash,
                        "signal_score": metrics_today.get(ticker, {}).get("signal_score"),
                        "signal_basis_date": metrics_today.get(ticker, {}).get("signal_basis_date"),
                        "reason": exit_reason,
                        "nikkei_regime": regime,
                    }
                )
                del positions[ticker]

        candidates = []
        if regime in regime_params:
            for ticker, sig in signal_today.items():
                if ticker in positions:
                    continue
                if sig and not prev_signal.get(ticker, False):
                    df = price_map[ticker]
                    static = static_lookup[ticker]
                    candidates.append(
                        {
                            "ticker": ticker,
                            "simple_sector": static.get("simple_sector", static.get("sector", "")),
                            "sector_33": static.get("sector_33", "-"),
                            "signal_score": metrics_today.get(ticker, {}).get("signal_score", 0.0),
                            "promising_score": static.get("promising_score", 0.0),
                            "signal_basis_date": metrics_today.get(ticker, {}).get("signal_basis_date", ""),
                        }
                    )

        buy_list = sorted(candidates, key=lambda x: (x["signal_score"], x["promising_score"]), reverse=True)
        remaining = len(buy_list)
        for c in buy_list:
            df = price_map[c["ticker"]]
            day = df.loc[date]
            prev_close = float(df.loc[pd.Timestamp(c["signal_basis_date"]), "Close"])
            trigger_price = prev_close * (1.0 - args.entry_limit_pct / 100.0) if args.entry_mode == "limit_down" else prev_close * (1.0 + args.entry_limit_pct / 100.0)
            day_low = float(day["Low"]) if "Low" in day.index and not pd.isna(day["Low"]) else float(day["Close"])
            day_high = float(day["High"]) if "High" in day.index and not pd.isna(day["High"]) else float(day["Close"])
            day_open = float(day["Open"]) if "Open" in day.index and not pd.isna(day["Open"]) else float(day["Close"])
            if args.entry_mode == "limit_down":
                if day_low > trigger_price:
                    remaining -= 1
                    continue
                fill_price = min(day_open, trigger_price)
            else:
                if day_high < trigger_price:
                    remaining -= 1
                    continue
                fill_price = max(day_open, trigger_price)
            alloc = cash / remaining if remaining > 0 else 0.0
            shares = int((alloc // (fill_price * 100))) * 100
            if shares >= 100 and shares * fill_price <= cash:
                cash -= shares * fill_price
                positions[c["ticker"]] = {
                    "shares": shares,
                    "entry_price": fill_price,
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
                        "reason": f"buy_{args.entry_mode}",
                        "nikkei_regime": regime,
                    }
                )
            remaining -= 1

        market_value = 0.0
        for ticker, pos in positions.items():
            df = price_map[ticker]
            usable_idx = df.index[df.index <= date]
            if len(usable_idx) == 0:
                continue
            market_value += pos["shares"] * float(df.loc[usable_idx[-1], "Close"])
        equity_curve.append(
            {
                "date": date.date().isoformat(),
                "cash": cash,
                "market_value": market_value,
                "equity": cash + market_value,
                "positions": len(positions),
                "nikkei_regime": regime,
            }
        )
        prev_signal = signal_today

    latest_date = max(dates)
    for ticker in list(positions.keys()):
        df = price_map[ticker]
        last_idx = df.index[df.index <= latest_date][-1]
        price = float(df.loc[last_idx, "Close"])
        shares = positions[ticker]["shares"]
        cash += shares * price
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
                "nikkei_regime": nikkei_regime_custom(
                    n225_close,
                    last_idx - pd.Timedelta(days=1),
                    up_ret20_threshold=args.up_ret20_threshold,
                    sideways_abs_ret20_threshold=args.sideways_abs_ret20_threshold,
                ),
            }
        )
        del positions[ticker]

    trade_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve)
    max_dd_pct = 0.0
    if not equity_df.empty:
        eq = equity_df["equity"].astype(float)
        running_max = eq.cummax()
        max_dd_pct = float(((eq / running_max) - 1.0).min()) * 100.0

    roundtrips = []
    buy_map = {}
    for _, tr in trade_df.iterrows():
        key = tr["ticker"]
        action = str(tr["action"]).upper()
        if action == "BUY":
            buy_map.setdefault(key, []).append(tr)
        else:
            buy = buy_map.get(key, []).pop(0)
            pnl = (tr["price"] - buy["price"]) * tr["shares"]
            ret_pct = (tr["price"] / buy["price"] - 1.0) * 100.0
            roundtrips.append(
                {
                    "ticker": key,
                    "simple_sector": buy["simple_sector"],
                    "sector_33": buy["sector_33"],
                    "buy_date": buy["date"],
                    "sell_date": tr["date"],
                    "shares": tr["shares"],
                    "buy_price": buy["price"],
                    "sell_price": tr["price"],
                    "pnl": pnl,
                    "return_pct": ret_pct,
                    "exit_reason": tr["reason"],
                }
            )
    roundtrip_df = pd.DataFrame(roundtrips)

    summary = {
        "condition_name": args.condition_name,
        "period_start": dates[0].date().isoformat(),
        "period_end": latest_date.date().isoformat(),
        "initial_capital": float(args.initial_capital),
        "final_capital": float(cash),
        "total_return_pct": (float(cash) / float(args.initial_capital) - 1.0) * 100.0,
        "num_buys": int((trade_df["action"] == "BUY").sum()) if not trade_df.empty else 0,
        "num_roundtrips": int(len(roundtrip_df)),
        "win_rate_pct": float((roundtrip_df["pnl"] > 0).mean()) * 100.0 if len(roundtrip_df) else 0.0,
        "avg_roundtrip_return_pct": float(roundtrip_df["return_pct"].mean()) if len(roundtrip_df) else 0.0,
        "max_drawdown_pct": max_dd_pct,
        "up_ret20_threshold": args.up_ret20_threshold,
        "sideways_abs_ret20_threshold": args.sideways_abs_ret20_threshold,
        "regime_counts": regime_counts,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trade_df.to_csv(args.output_dir / "trade_log.csv", index=False, encoding="utf-8-sig")
    roundtrip_df.to_csv(args.output_dir / "roundtrip_trades.csv", index=False, encoding="utf-8-sig")
    equity_df.to_csv(args.output_dir / "equity_curve.csv", index=False, encoding="utf-8-sig")
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

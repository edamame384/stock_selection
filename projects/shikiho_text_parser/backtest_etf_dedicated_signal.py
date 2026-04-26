from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PRICES_DIR = ROOT / "data" / "prices"


def load_df(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    path = PRICES_DIR / f"{ticker.replace('.T', '')}_T.csv"
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df[(df.index >= start - pd.Timedelta(days=120)) & (df.index <= end)].copy()
    df["ma5"] = df["Close"].rolling(5).mean()
    df["ma20"] = df["Close"].rolling(20).mean()
    df["ma60"] = df["Close"].rolling(60).mean()
    df["ret5"] = df["Close"].pct_change(5)
    df["ret10"] = df["Close"].pct_change(10)
    df["hh20"] = df["Close"].rolling(20).max()
    df["dd20"] = df["Close"] / df["hh20"] - 1.0
    return df


def signal_current(hist: pd.DataFrame) -> tuple[bool, float | str | None]:
    if len(hist) < 20:
        return False, None
    row = hist.iloc[-1]
    if pd.isna(row["ma5"]) or pd.isna(row["ma20"]):
        return False, None
    sig = row["Close"] > row["ma5"] and row["ma5"] >= row["ma20"]
    return sig, row["Close"] * 1.01


def signal_rebound_open(hist: pd.DataFrame) -> tuple[bool, float | str | None]:
    if len(hist) < 20:
        return False, None
    row = hist.iloc[-1]
    if pd.isna(row["ma5"]) or pd.isna(row["dd20"]) or pd.isna(row["ret5"]):
        return False, None
    sig = row["dd20"] <= -0.05 and row["ret5"] <= -0.02 and row["Close"] > row["ma5"]
    return sig, "OPEN"


def signal_trend_open(hist: pd.DataFrame) -> tuple[bool, float | str | None]:
    if len(hist) < 60:
        return False, None
    row = hist.iloc[-1]
    if pd.isna(row["ma20"]) or pd.isna(row["ma60"]):
        return False, None
    sig = row["Close"] > row["ma20"] and row["ma20"] >= row["ma60"] and row["ret10"] > 0
    return sig, "OPEN"


def signal_crash_reclaim(hist: pd.DataFrame) -> tuple[bool, float | str | None]:
    if len(hist) < 20:
        return False, None
    row = hist.iloc[-1]
    if pd.isna(row["ma20"]) or pd.isna(row["dd20"]):
        return False, None
    sig = row["dd20"] <= -0.04 and row["Close"] > row["ma20"]
    return sig, row["Close"] * 1.005


SIGNALS = {
    "current_breakout": signal_current,
    "rebound_open": signal_rebound_open,
    "trend_open": signal_trend_open,
    "crash_reclaim": signal_crash_reclaim,
}


def run_backtest(
    ticker: str,
    name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    initial_capital: float,
    signal_name: str,
) -> dict:
    df = load_df(ticker, start, end)
    trade_dates = [d for d in df.index if start <= d <= end]
    fn = SIGNALS[signal_name]
    cash = initial_capital
    pos = None
    buys = 0
    equity_rows = []
    trade_rows = []
    prev_sig = False

    for date in trade_dates:
        prev_idx = df.index[df.index < date]
        if len(prev_idx) == 0:
            equity_rows.append({"date": date, "equity": cash})
            continue
        basis = prev_idx[-1]
        hist = df.loc[:basis]
        sig, trigger = fn(hist)
        day = df.loc[date]
        open_p = float(day["Open"]) if not pd.isna(day["Open"]) else float(day["Close"])
        high_p = float(day["High"]) if not pd.isna(day["High"]) else float(day["Close"])
        close_p = float(day["Close"])

        if pos is not None:
            ret = close_p / pos["entry_price"] - 1.0
            reason = None
            exit_p = None
            if ret >= 0.05:
                reason = "take_profit"
                exit_p = close_p
            elif ret <= -0.04:
                reason = "stop_loss"
                exit_p = close_p
            elif not sig:
                reason = "sell_signal"
                exit_p = close_p
            if reason:
                cash += pos["shares"] * exit_p
                trade_rows.append(
                    {"date": date.strftime("%Y-%m-%d"), "ticker": ticker, "name": name, "action": "SELL", "price": exit_p, "shares": pos["shares"], "reason": reason}
                )
                pos = None

        if pos is None and sig and not prev_sig:
            if trigger == "OPEN":
                tradable = True
                fill = open_p
            else:
                tradable = high_p >= float(trigger)
                fill = max(open_p, float(trigger)) if tradable else None
            if tradable:
                shares = int((cash // (fill * 100)) * 100)
                if shares >= 100 and shares * fill <= cash:
                    cash -= shares * fill
                    pos = {"entry_price": fill, "shares": shares}
                    buys += 1
                    trade_rows.append(
                        {"date": date.strftime("%Y-%m-%d"), "ticker": ticker, "name": name, "action": "BUY", "price": fill, "shares": shares, "reason": signal_name}
                    )

        prev_sig = sig
        market_value = 0.0 if pos is None else pos["shares"] * close_p
        equity_rows.append({"date": date, "equity": cash + market_value})

    if pos is not None:
        px = float(df.loc[trade_dates[-1], "Close"])
        cash += pos["shares"] * px
        trade_rows.append(
            {"date": trade_dates[-1].strftime("%Y-%m-%d"), "ticker": ticker, "name": name, "action": "SELL", "price": px, "shares": pos["shares"], "reason": "end_of_backtest"}
        )

    equity_df = pd.DataFrame(equity_rows)
    max_dd_pct = (equity_df["equity"] / equity_df["equity"].cummax() - 1.0).min() * 100 if not equity_df.empty else 0.0
    return {
        "ticker": ticker,
        "name": name,
        "signal_name": signal_name,
        "final_capital": cash,
        "total_return_pct": (cash / initial_capital - 1.0) * 100.0,
        "num_buys": buys,
        "max_drawdown_pct": max_dd_pct,
        "trade_log": pd.DataFrame(trade_rows),
        "equity_curve": equity_df,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run dedicated ETF signal backtest.")
    parser.add_argument("--ticker", type=str, required=True)
    parser.add_argument("--name", type=str, default="ETF")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, required=True)
    parser.add_argument("--initial-capital", type=float, default=3_000_000.0)
    parser.add_argument("--signal-name", type=str, default="rebound_open", choices=sorted(SIGNALS.keys()))
    args = parser.parse_args()

    result = run_backtest(
        ticker=args.ticker,
        name=args.name,
        start=pd.Timestamp(args.start_date),
        end=pd.Timestamp(args.end_date),
        initial_capital=args.initial_capital,
        signal_name=args.signal_name,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in result.items() if k not in {"trade_log", "equity_curve"}}
    (args.output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    result["trade_log"].to_csv(args.output_dir / "trade_log.csv", index=False, encoding="utf-8-sig")
    result["equity_curve"].to_csv(args.output_dir / "equity_curve.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

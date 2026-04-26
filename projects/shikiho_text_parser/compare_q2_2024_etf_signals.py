from __future__ import annotations

from pathlib import Path
import math

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PRICES_DIR = ROOT / "data" / "prices"
OUT_DIR = ROOT / "projects" / "shikiho_text_parser" / "output" / "q2_2024_etf_signal_reverify"

CANDIDATES = {
    "1306.T": "TOPIX ETF",
    "1321.T": "Nikkei225 ETF",
    "1489.T": "Nikkei High Dividend 50",
    "1328.T": "Gold Price ETF",
    "1540.T": "Physical Gold ETF",
    "1326.T": "SPDR Gold",
    "1672.T": "WisdomTree Gold",
    "1478.T": "MSCI Japan High Dividend",
    "1399.T": "Japan High Dividend Low Vol",
}

START = pd.Timestamp("2024-07-01")
END = pd.Timestamp("2024-09-30")
INITIAL_CAPITAL = 3_000_000.0


def load_df(ticker: str) -> pd.DataFrame | None:
    path = PRICES_DIR / f"{ticker.replace('.T', '')}_T.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df[(df.index >= START - pd.Timedelta(days=120)) & (df.index <= END)].copy()
    if df.empty:
        return None
    df["ma5"] = df["Close"].rolling(5).mean()
    df["ma20"] = df["Close"].rolling(20).mean()
    df["ma60"] = df["Close"].rolling(60).mean()
    df["ret5"] = df["Close"].pct_change(5)
    df["ret10"] = df["Close"].pct_change(10)
    df["hh20"] = df["Close"].rolling(20).max()
    df["dd20"] = df["Close"] / df["hh20"] - 1.0
    return df


def signal_current(hist: pd.DataFrame) -> tuple[bool, float | str | None, float]:
    if len(hist) < 20:
        return False, None, 0.0
    row = hist.iloc[-1]
    if pd.isna(row["ma5"]) or pd.isna(row["ma20"]):
        return False, None, 0.0
    sig = row["Close"] > row["ma5"] and row["ma5"] >= row["ma20"]
    trig = row["Close"] * 1.01
    score = max(row["Close"] / row["ma5"] - 1.0, 0.0)
    return sig, trig, float(score)


def signal_rebound_open(hist: pd.DataFrame) -> tuple[bool, float | str | None, float]:
    if len(hist) < 20:
        return False, None, 0.0
    row = hist.iloc[-1]
    if pd.isna(row["ma5"]) or pd.isna(row["dd20"]) or pd.isna(row["ret5"]):
        return False, None, 0.0
    sig = row["dd20"] <= -0.05 and row["ret5"] <= -0.02 and row["Close"] > row["ma5"]
    score = max(-row["dd20"], 0.0)
    return sig, "OPEN", float(score)


def signal_trend_open(hist: pd.DataFrame) -> tuple[bool, float | str | None, float]:
    if len(hist) < 60:
        return False, None, 0.0
    row = hist.iloc[-1]
    if pd.isna(row["ma20"]) or pd.isna(row["ma60"]):
        return False, None, 0.0
    sig = row["Close"] > row["ma20"] and row["ma20"] >= row["ma60"] and row["ret10"] > 0
    return sig, "OPEN", float(row["ret10"])


def signal_crash_reclaim(hist: pd.DataFrame) -> tuple[bool, float | str | None, float]:
    if len(hist) < 20:
        return False, None, 0.0
    row = hist.iloc[-1]
    if pd.isna(row["ma20"]) or pd.isna(row["dd20"]):
        return False, None, 0.0
    sig = row["dd20"] <= -0.04 and row["Close"] > row["ma20"]
    return sig, row["Close"] * 1.005, max(-row["dd20"], 0.0)


SIGNALS = {
    "current_breakout": signal_current,
    "rebound_open": signal_rebound_open,
    "trend_open": signal_trend_open,
    "crash_reclaim": signal_crash_reclaim,
}


def run_one(ticker: str, name: str, signal_name: str, fn) -> tuple[dict, list[dict]]:
    df = load_df(ticker)
    if df is None:
        return {}, []
    trade_dates = [d for d in df.index if START <= d <= END]
    cash = INITIAL_CAPITAL
    pos = None
    buys = 0
    equity = []
    trade_rows = []
    prev_sig = False

    for date in trade_dates:
        prev_idx = df.index[df.index < date]
        if len(prev_idx) == 0:
            equity.append({"date": date, "equity": cash})
            continue
        basis = prev_idx[-1]
        hist = df.loc[:basis]
        sig, trigger, _score = fn(hist)
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
                    {
                        "ticker": ticker,
                        "name": name,
                        "signal": signal_name,
                        "date": date.strftime("%Y-%m-%d"),
                        "action": "SELL",
                        "price": exit_p,
                        "shares": pos["shares"],
                        "reason": reason,
                    }
                )
                pos = None

        if pos is None and sig and not prev_sig:
            if trigger == "OPEN":
                fill = open_p
                tradable = True
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
                        {
                            "ticker": ticker,
                            "name": name,
                            "signal": signal_name,
                            "date": date.strftime("%Y-%m-%d"),
                            "action": "BUY",
                            "price": fill,
                            "shares": shares,
                            "reason": "entry",
                        }
                    )

        prev_sig = sig
        market_value = 0.0 if pos is None else pos["shares"] * close_p
        equity.append({"date": date, "equity": cash + market_value})

    if pos is not None:
        px = float(df.loc[trade_dates[-1], "Close"])
        cash += pos["shares"] * px
        trade_rows.append(
            {
                "ticker": ticker,
                "name": name,
                "signal": signal_name,
                "date": trade_dates[-1].strftime("%Y-%m-%d"),
                "action": "SELL",
                "price": px,
                "shares": pos["shares"],
                "reason": "end_of_backtest",
            }
        )

    eq = pd.DataFrame(equity)
    max_dd_pct = (eq["equity"] / eq["equity"].cummax() - 1.0).min() * 100 if not eq.empty else 0.0
    summary = {
        "ticker": ticker,
        "name": name,
        "signal": signal_name,
        "final_capital": cash,
        "return_pct": (cash / INITIAL_CAPITAL - 1.0) * 100.0,
        "num_buys": buys,
        "max_dd_pct": max_dd_pct,
    }
    return summary, trade_rows


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    trades = []
    for ticker, name in CANDIDATES.items():
        for signal_name, fn in SIGNALS.items():
            summary, trade_rows = run_one(ticker, name, signal_name, fn)
            if summary:
                summaries.append(summary)
                trades.extend(trade_rows)
    summary_df = pd.DataFrame(summaries).sort_values(["signal", "return_pct"], ascending=[True, False])
    trade_df = pd.DataFrame(trades)
    summary_df.to_csv(OUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    trade_df.to_csv(OUT_DIR / "trade_log.csv", index=False, encoding="utf-8-sig")
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PRICES_DIR = ROOT / "data" / "prices"
PHASE_CSV = ROOT / "projects" / "shikiho_text_parser" / "output" / "nikkei_market_phase_map" / "nikkei_market_phase_daily_labels.csv"
OUT_DIR = ROOT / "projects" / "shikiho_text_parser" / "output" / "post_major_crash_etf_price_reversal_candidate"

DATASETS = {
    "q2_2024": ("2024-07-01", "2024-09-30"),
    "2025_2q": ("2025-04-01", "2025-06-30"),
}

ETF_CANDIDATES = {
    "1306.T": "TOPIX ETF",
    "1321.T": "Nikkei225 ETF",
}

EXIT_VARIANTS = [
    {"name": "tp6_sl3", "take_profit_pct": 6.0, "stop_loss_pct": 3.0, "max_hold_days": None},
    {"name": "tp6_sl4", "take_profit_pct": 6.0, "stop_loss_pct": 4.0, "max_hold_days": None},
    {"name": "tp5_sl4", "take_profit_pct": 5.0, "stop_loss_pct": 4.0, "max_hold_days": None},
]


def load_phase_df() -> pd.DataFrame:
    df = pd.read_csv(PHASE_CSV, skiprows=[1])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.rename(columns={"phase": "phase_name"})
    for col in ["ret5", "dd20", "vol10"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["major_crash"] = (df["phase_name"].astype(str) == "crash") & ((df["ret5"] <= -0.08) | (df["dd20"] <= -0.14))
    return df[["Date", "phase_name", "ret5", "dd20", "vol10", "major_crash"]].copy().sort_values("Date")


def build_post_major_state(phase_df: pd.DataFrame) -> pd.DataFrame:
    x = phase_df.copy().reset_index(drop=True)
    mode = False
    recovery_streak = 0
    states = []
    for _, row in x.iterrows():
        phase = str(row["phase_name"])
        dd20 = float(row["dd20"]) if pd.notna(row["dd20"]) else None
        if bool(row["major_crash"]):
            mode = True
            recovery_streak = 0
        elif mode:
            if phase in {"stable", "uptrend"} and dd20 is not None and dd20 >= -0.05:
                recovery_streak += 1
            else:
                recovery_streak = 0
            if recovery_streak >= 5:
                mode = False
        states.append(bool(mode))
    x["post_major_crash_mode"] = states
    return x


def load_price_df(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    path = PRICES_DIR / f"{ticker.replace('.T', '')}_T.csv"
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df[(df.index >= start - pd.Timedelta(days=160)) & (df.index <= end)].copy()
    df["ret3"] = df["Close"].pct_change(3)
    df["ret5"] = df["Close"].pct_change(5)
    df["hh3"] = df["High"].rolling(3).max()
    df["ll3"] = df["Low"].rolling(3).min()
    df["hh5"] = df["High"].rolling(5).max()
    df["ll5"] = df["Low"].rolling(5).min()
    df["hh20"] = df["Close"].rolling(20).max()
    df["dd20"] = df["Close"] / df["hh20"] - 1.0
    return df


def signal_prev_high_break(hist: pd.DataFrame) -> tuple[bool, float | None]:
    if len(hist) < 3:
        return False, None
    prev_high = float(hist.iloc[-1]["High"])
    return True, prev_high * 1.002


def signal_reclaim_3d_low(hist: pd.DataFrame) -> tuple[bool, float | None]:
    if len(hist) < 5:
        return False, None
    row = hist.iloc[-1]
    if pd.isna(row["ll3"]) or pd.isna(row["ret3"]):
        return False, None
    sig = (row["Close"] / row["ll3"] - 1.0) >= 0.035 and row["ret3"] > 0
    return bool(sig), None


def signal_crash_low_reclaim(hist: pd.DataFrame) -> tuple[bool, float | None]:
    if len(hist) < 10:
        return False, None
    row = hist.iloc[-1]
    if pd.isna(row["ll5"]) or pd.isna(row["dd20"]):
        return False, None
    crash_low = float(hist.tail(10)["Low"].min())
    sig = row["Close"] >= crash_low * 1.05 and row["dd20"] <= -0.03
    return bool(sig), None


SIGNALS = {
    "prev_high_break": signal_prev_high_break,
    "reclaim_3d_low": signal_reclaim_3d_low,
    "crash_low_reclaim": signal_crash_low_reclaim,
}


def regime_allows_entry(date: pd.Timestamp, phase_df: pd.DataFrame) -> bool:
    hist = phase_df[phase_df["Date"] < date].copy()
    if hist.empty:
        return False
    last = hist.iloc[-1]
    if not bool(last["post_major_crash_mode"]):
        return False
    return str(last["phase_name"]) in {"high_vol", "capitulation_end", "settling", "normal"}


def run_one(dataset: str, ticker: str, name: str, signal_name: str, exit_variant: dict, initial_capital: float = 3_000_000.0) -> tuple[dict, pd.DataFrame]:
    start = pd.Timestamp(DATASETS[dataset][0])
    end = pd.Timestamp(DATASETS[dataset][1])
    phase_df = build_post_major_state(load_phase_df())
    price_df = load_price_df(ticker, start, end)
    trade_dates = [d for d in price_df.index if start <= d <= end]
    signal_fn = SIGNALS[signal_name]

    cash = initial_capital
    pos = None
    buys = 0
    equity_rows = []
    trade_rows = []
    for date in trade_dates:
        prev_idx = price_df.index[price_df.index < date]
        if len(prev_idx) == 0:
            equity_rows.append({"date": date, "equity": cash})
            continue
        basis = prev_idx[-1]
        hist = price_df.loc[:basis]
        sig, trigger = signal_fn(hist)
        allow = regime_allows_entry(date, phase_df)
        day = price_df.loc[date]
        open_p = float(day["Open"]) if not pd.isna(day["Open"]) else float(day["Close"])
        high_p = float(day["High"]) if not pd.isna(day["High"]) else float(day["Close"])
        close_p = float(day["Close"])

        if pos is not None:
            ret = close_p / pos["entry_price"] - 1.0
            reason = None
            if ret >= exit_variant["take_profit_pct"] / 100.0:
                reason = "take_profit"
            elif ret <= -exit_variant["stop_loss_pct"] / 100.0:
                reason = "stop_loss"
            elif not sig:
                reason = "sell_signal"
            if reason:
                cash += pos["shares"] * close_p
                trade_rows.append({"date": date.strftime("%Y-%m-%d"), "ticker": ticker, "name": name, "dataset": dataset, "signal": signal_name, "exit_variant": exit_variant["name"], "action": "SELL", "price": close_p, "shares": pos["shares"], "reason": reason})
                pos = None

        if pos is None and allow and sig:
            tradable = True
            fill = open_p
            if trigger is not None:
                tradable = high_p >= float(trigger)
                fill = max(open_p, float(trigger)) if tradable else None
            if tradable:
                shares = int((cash // (fill * 100)) * 100)
                if shares >= 100 and shares * fill <= cash:
                    cash -= shares * fill
                    pos = {"entry_price": fill, "shares": shares}
                    buys += 1
                    trade_rows.append({"date": date.strftime("%Y-%m-%d"), "ticker": ticker, "name": name, "dataset": dataset, "signal": signal_name, "exit_variant": exit_variant["name"], "action": "BUY", "price": fill, "shares": shares, "reason": "entry"})

        market_value = 0.0 if pos is None else pos["shares"] * close_p
        equity_rows.append({"date": date, "equity": cash + market_value})

    if pos is not None:
        px = float(price_df.loc[trade_dates[-1], "Close"])
        cash += pos["shares"] * px
        trade_rows.append({"date": trade_dates[-1].strftime("%Y-%m-%d"), "ticker": ticker, "name": name, "dataset": dataset, "signal": signal_name, "exit_variant": exit_variant["name"], "action": "SELL", "price": px, "shares": pos["shares"], "reason": "end_of_backtest"})

    equity_df = pd.DataFrame(equity_rows)
    max_dd_pct = (equity_df["equity"] / equity_df["equity"].cummax() - 1.0).min() * 100 if not equity_df.empty else 0.0
    return {
        "dataset": dataset,
        "ticker": ticker,
        "name": name,
        "signal": signal_name,
        "exit_variant": exit_variant["name"],
        "take_profit_pct": exit_variant["take_profit_pct"],
        "stop_loss_pct": exit_variant["stop_loss_pct"],
        "max_hold_days": exit_variant["max_hold_days"],
        "final_capital": cash,
        "total_return_pct": (cash / initial_capital - 1.0) * 100.0,
        "num_buys": buys,
        "max_drawdown_pct": max_dd_pct,
    }, pd.DataFrame(trade_rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    trades = []
    phase_df = build_post_major_state(load_phase_df())
    phase_df.to_csv(OUT_DIR / "phase_state_daily.csv", index=False, encoding="utf-8-sig")
    for dataset in DATASETS:
        for ticker, name in ETF_CANDIDATES.items():
            for signal_name in SIGNALS:
                for exit_variant in EXIT_VARIANTS:
                    summary, trade_df = run_one(dataset, ticker, name, signal_name, exit_variant)
                    summaries.append(summary)
                    if not trade_df.empty:
                        trades.append(trade_df)
    summary_df = pd.DataFrame(summaries).sort_values(["dataset", "total_return_pct"], ascending=[True, False])
    trade_df = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    summary_df.to_csv(OUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    trade_df.to_csv(OUT_DIR / "trade_log.csv", index=False, encoding="utf-8-sig")
    best = summary_df.groupby("dataset").head(10).reset_index(drop=True)
    (OUT_DIR / "best_by_dataset.json").write_text(best.to_json(force_ascii=False, orient="records", indent=2), encoding="utf-8")
    print(best.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

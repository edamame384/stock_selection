from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.shikiho_text_parser.compare_phase_adaptive_practical_v3 import load_price_map, normalize_static
from projects.shikiho_text_parser.search_phase_method_optimization import PHASE_CSV, load_phase_map
from projects.shikiho_text_parser.backtest_phase_adaptive_practical_v31_pre1_post5_except_crash import projected_phase_name


DETAIL_CSV = ROOT / "projects" / "quarterly_ranker" / "output" / "q2_2024_pre_analysis_20240630_aligned" / "q2_2024_pre_shikiho_feature_ranking.csv"
OUT_DIR = ROOT / "projects" / "shikiho_text_parser" / "output" / "q2_2024_highvol_sector_deploy_candidate"
START = pd.Timestamp("2024-07-01")
END = pd.Timestamp("2024-09-30")
INITIAL_CAPITAL = 3_000_000.0
PHASE_SHIFT_DAYS = 1
PHASE_PROXY_MODE = "difficult_v11"
TOP4 = {"サービス業", "不動産業", "陸運業", "情報・通信業"}


def select_top4_balanced(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x = x[
        x["sector"].isin(TOP4)
        & (x["trend_r2"].fillna(0) >= 0.74)
        & (x["annual_return_pct"].fillna(-999) >= 8.0)
        & (x["quarter_return_pct"].fillna(-999) >= 2.0)
        & (x["positive_month_ratio_pct"].fillna(0) >= 60.0)
        & (x["persistence_20d_pct"].fillna(0) >= 64.0)
        & (x["end_to_trailing_high_pct"].fillna(0) >= 87.0)
        & (x["sector_adjusted_per_score"].fillna(0) >= 0.18)
        & (x["ocr_per"].fillna(999) <= 29.0)
    ]
    return x.sort_values(
        ["trend_r2", "persistence_20d_pct", "positive_month_ratio_pct", "annual_return_pct", "sector_adjusted_per_score"],
        ascending=[False, False, False, False, False],
    ).drop_duplicates(subset=["ticker"], keep="first").reset_index(drop=True)


def run_variant(
    tickers: list[str],
    price_map: dict[str, pd.DataFrame],
    phase_map: pd.Series,
    variant: str,
) -> tuple[dict, pd.DataFrame]:
    trade_dates = sorted(set().union(*[set(df[(df.index >= START) & (df.index <= END)].index.tolist()) for df in price_map.values()]))
    cash = INITIAL_CAPITAL
    positions: dict[str, dict] = {}
    trade_rows: list[dict] = []
    equity_rows: list[dict] = []
    prev_sig = {ticker: False for ticker in tickers}
    buys = 0

    for date in trade_dates:
        phase_name = projected_phase_name(date, phase_map, PHASE_SHIFT_DAYS, PHASE_PROXY_MODE)
        signal_today: dict[str, bool] = {}
        trigger_today: dict[str, float] = {}

        active_entry = True
        if variant in {"high_vol_entry_only_hold", "high_vol_only_phase_exit"}:
            active_entry = phase_name == "high_vol"

        for ticker in tickers:
            df = price_map[ticker]
            if date not in df.index:
                signal_today[ticker] = False
                continue
            prev_idx = df.index[df.index < date]
            if len(prev_idx) == 0:
                signal_today[ticker] = False
                continue
            prev_close = float(df.loc[prev_idx[-1], "Close"])
            signal_today[ticker] = active_entry
            trigger_today[ticker] = prev_close * 1.01

        for ticker in list(positions.keys()):
            df = price_map[ticker]
            if date not in df.index:
                continue
            day = df.loc[date]
            close_p = float(day["Close"])
            low_p = float(day["Low"]) if "Low" in day.index and not pd.isna(day["Low"]) else close_p
            open_p = float(day["Open"]) if "Open" in day.index and not pd.isna(day["Open"]) else close_p
            entry = positions[ticker]["entry_price"]
            prev_idx = df.index[df.index < date]
            prev_close = float(df.loc[prev_idx[-1], "Close"]) if len(prev_idx) else close_p
            exit_reason = None
            exit_price = None
            if variant == "high_vol_only_phase_exit" and phase_name != "high_vol":
                exit_reason = "phase_exit"
                exit_price = close_p
            elif low_p <= prev_close * 0.98:
                exit_reason = "crash_early_exit"
                exit_price = min(open_p, prev_close * 0.98)
            else:
                ret = close_p / entry - 1.0
                if ret >= 0.05:
                    exit_reason = "take_profit"
                    exit_price = close_p
                elif ret <= -0.04:
                    exit_reason = "stop_loss"
                    exit_price = close_p
            if exit_reason is not None:
                shares = positions[ticker]["shares"]
                cash += shares * float(exit_price)
                trade_rows.append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "ticker": ticker,
                        "action": "SELL",
                        "price": float(exit_price),
                        "shares": int(shares),
                        "reason": exit_reason,
                        "phase": phase_name,
                    }
                )
                del positions[ticker]

        candidates = [ticker for ticker in tickers if ticker not in positions and signal_today.get(ticker, False) and not prev_sig.get(ticker, False)]
        remaining = len(candidates)
        for ticker in candidates:
            df = price_map[ticker]
            day = df.loc[date]
            high_p = float(day["High"]) if "High" in day.index and not pd.isna(day["High"]) else float(day["Close"])
            open_p = float(day["Open"]) if "Open" in day.index and not pd.isna(day["Open"]) else float(day["Close"])
            trigger = trigger_today[ticker]
            if high_p < trigger:
                remaining -= 1
                continue
            fill = max(open_p, trigger)
            alloc = cash / remaining if remaining > 0 else 0.0
            shares = int((alloc // (fill * 100.0)) * 100)
            if shares >= 100 and shares * fill <= cash:
                cash -= shares * fill
                positions[ticker] = {"entry_price": fill, "shares": shares}
                trade_rows.append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "ticker": ticker,
                        "action": "BUY",
                        "price": float(fill),
                        "shares": int(shares),
                        "reason": "defensive_entry",
                        "phase": phase_name,
                    }
                )
                buys += 1
            remaining -= 1

        market_value = 0.0
        for ticker, pos in positions.items():
            df = price_map[ticker]
            usable = df.index[df.index <= date]
            if len(usable):
                market_value += pos["shares"] * float(df.loc[usable[-1], "Close"])
        equity_rows.append({"date": date, "phase": phase_name, "equity": cash + market_value})
        prev_sig = signal_today

    if trade_dates:
        last_date = trade_dates[-1]
        last_phase = projected_phase_name(last_date, phase_map, PHASE_SHIFT_DAYS, PHASE_PROXY_MODE)
        for ticker in list(positions.keys()):
            px = float(price_map[ticker].loc[last_date, "Close"])
            cash += positions[ticker]["shares"] * px
            trade_rows.append(
                {
                    "date": last_date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "action": "SELL",
                    "price": float(px),
                    "shares": int(positions[ticker]["shares"]),
                    "reason": "end_of_backtest",
                    "phase": last_phase,
                }
            )
            del positions[ticker]

    equity_df = pd.DataFrame(equity_rows)
    max_dd = float((equity_df["equity"] / equity_df["equity"].cummax() - 1.0).min() * 100.0) if not equity_df.empty else 0.0
    phase_pnl = {}
    if not equity_df.empty:
        equity_df["pnl_day"] = equity_df["equity"].diff().fillna(equity_df["equity"] - INITIAL_CAPITAL)
        phase_pnl = equity_df.groupby("phase")["pnl_day"].sum().to_dict()
    summary = {
        "variant": variant,
        "final_capital": float(cash),
        "return_pct": (float(cash) / INITIAL_CAPITAL - 1.0) * 100.0,
        "num_buys": int(buys),
        "max_dd_pct": max_dd,
        "phase_pnl": phase_pnl,
    }
    return summary, pd.DataFrame(trade_rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detail_df = normalize_static(pd.read_csv(DETAIL_CSV)).drop_duplicates(subset=["ticker"], keep="first")
    selected = select_top4_balanced(detail_df)
    selected.to_csv(OUT_DIR / "top4_chart_v2_balanced_selected.csv", index=False, encoding="utf-8-sig")
    tickers = selected["ticker"].astype(str).tolist()
    price_map = load_price_map(tickers, END)
    phase_map = load_phase_map(PHASE_CSV)

    variants = ["full_period", "high_vol_entry_only_hold", "high_vol_only_phase_exit"]
    rows = []
    for variant in variants:
        result, trade_df = run_variant(tickers, price_map, phase_map, variant)
        rows.append(result)
        out = OUT_DIR / variant
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        trade_df.to_csv(out / "trade_log.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(rows).to_csv(OUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

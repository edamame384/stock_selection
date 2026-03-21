from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from projects.shikiho_text_parser.paths import OUTPUT_DIR, PRICE_DIR
from projects.market_news_automation.paths import DAILY_FEATURES_CSV
from src.stock_signal import (
    download_additional_macro_features,
    download_dow_futures_feature,
    download_index_returns,
    download_nikkei_futures_night_feature,
)


def load_live_news_signal(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_historical_news_signal(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["news_sentiment_score", "news_headline_count"])
    df = pd.read_csv(path)
    if "date" not in df.columns:
        return pd.DataFrame(columns=["news_sentiment_score", "news_headline_count"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    if "aggregate_sentiment_score" in df.columns:
        df["news_sentiment_score"] = pd.to_numeric(df["aggregate_sentiment_score"], errors="coerce").fillna(0.0)
    elif "sentiment_mean" in df.columns:
        df["news_sentiment_score"] = pd.to_numeric(df["sentiment_mean"], errors="coerce").fillna(0.0)
    elif "sentiment_score" in df.columns:
        df["news_sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce").fillna(0.0)
    else:
        df["news_sentiment_score"] = 0.0
    if "aggregate_headline_count" in df.columns:
        df["news_headline_count"] = pd.to_numeric(df["aggregate_headline_count"], errors="coerce").fillna(0.0)
    elif "headline_count_sum" in df.columns:
        df["news_headline_count"] = pd.to_numeric(df["headline_count_sum"], errors="coerce").fillna(0.0)
    elif "headline_count" in df.columns:
        df["news_headline_count"] = pd.to_numeric(df["headline_count"], errors="coerce").fillna(0.0)
    else:
        df["news_headline_count"] = 0.0
    return df.set_index("date")[["news_sentiment_score", "news_headline_count"]].sort_index()


def load_selected(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    num_cols = [
        "forecast_per",
        "sector_adjusted_per_score",
        "shikiho_score_overall",
        "headline_score_raw",
        "promising_score",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def calc_signal(
    df: pd.DataFrame,
    date: pd.Timestamp,
    static_row: dict,
    external_df: pd.DataFrame,
    promising_score_min: float,
    trend_r2_min: float,
    sector_per_score_min: float,
    macro_confirm_min: float,
    macro_score_weight: float,
) -> tuple[bool, dict]:
    hist = df.loc[:date].copy()
    if len(hist) < 180:
        return False, {}
    trailing = hist.tail(min(252, len(hist))).copy()
    qdf = hist.tail(min(63, len(hist))).copy()
    if len(trailing) < 120 or len(qdf) < 20:
        return False, {}

    annual_return_pct = (trailing["Close"].iloc[-1] / trailing["Close"].iloc[0] - 1.0) * 100.0
    quarter_return_pct = (qdf["Close"].iloc[-1] / qdf["Close"].iloc[0] - 1.0) * 100.0

    close = trailing["Close"].astype(float)
    y = np.log(close.replace(0, np.nan).dropna().values)
    if len(y) < 20:
        return False, {}
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

    external_hist = external_df.reindex(trailing.index).fillna(0.0)
    ret_1_series = trailing["Close"].astype(float).pct_change()
    corr_dji_20 = float(ret_1_series.rolling(20).corr(external_hist["ret_dji"]).iloc[-1]) if "ret_dji" in external_hist.columns else 0.0
    corr_dow_fut_20 = (
        float(ret_1_series.rolling(20).corr(external_hist["ret_dow_fut"]).iloc[-1])
        if "ret_dow_fut" in external_hist.columns
        else 0.0
    )
    corr_nk_night_20 = (
        float(ret_1_series.rolling(20).corr(external_hist["fut_nk_night_ret"]).iloc[-1])
        if "fut_nk_night_ret" in external_hist.columns
        else 0.0
    )
    corr_vix_20 = float(ret_1_series.rolling(20).corr(external_hist["ret_vix"]).iloc[-1]) if "ret_vix" in external_hist.columns else 0.0
    corr_ixic_20 = float(ret_1_series.rolling(20).corr(external_hist["ret_ixic"]).iloc[-1]) if "ret_ixic" in external_hist.columns else 0.0
    corr_usdjpy_20 = (
        float(ret_1_series.rolling(20).corr(external_hist["ret_usdjpy"]).iloc[-1]) if "ret_usdjpy" in external_hist.columns else 0.0
    )
    corr_dji_20 = 0.0 if pd.isna(corr_dji_20) else corr_dji_20
    corr_dow_fut_20 = 0.0 if pd.isna(corr_dow_fut_20) else corr_dow_fut_20
    corr_nk_night_20 = 0.0 if pd.isna(corr_nk_night_20) else corr_nk_night_20
    corr_vix_20 = 0.0 if pd.isna(corr_vix_20) else corr_vix_20
    corr_ixic_20 = 0.0 if pd.isna(corr_ixic_20) else corr_ixic_20
    corr_usdjpy_20 = 0.0 if pd.isna(corr_usdjpy_20) else corr_usdjpy_20

    latest_external = external_df.loc[:date].tail(1)
    ret_dji = float(latest_external["ret_dji"].iloc[0]) if not latest_external.empty and "ret_dji" in latest_external.columns else 0.0
    ret_dow_fut = (
        float(latest_external["ret_dow_fut"].iloc[0])
        if not latest_external.empty and "ret_dow_fut" in latest_external.columns
        else 0.0
    )
    fut_nk_night_ret = (
        float(latest_external["fut_nk_night_ret"].iloc[0])
        if not latest_external.empty and "fut_nk_night_ret" in latest_external.columns
        else 0.0
    )
    ret_vix = float(latest_external["ret_vix"].iloc[0]) if not latest_external.empty and "ret_vix" in latest_external.columns else 0.0
    ret_ixic = float(latest_external["ret_ixic"].iloc[0]) if not latest_external.empty and "ret_ixic" in latest_external.columns else 0.0
    ret_usdjpy = (
        float(latest_external["ret_usdjpy"].iloc[0]) if not latest_external.empty and "ret_usdjpy" in latest_external.columns else 0.0
    )

    macro_confirm_score = (
        0.20 * np.tanh(fut_nk_night_ret / 0.01)
        + 0.15 * np.tanh(ret_dow_fut / 0.01)
        + 0.15 * np.tanh(ret_dji / 0.01)
        + 0.20 * np.tanh(ret_ixic / 0.01)
        + 0.10 * np.tanh(ret_usdjpy / 0.01)
        - 0.20 * np.tanh(ret_vix / 0.05)
    )
    macro_corr_score = (
        0.20 * np.tanh(fut_nk_night_ret / 0.01) * np.clip(corr_nk_night_20, -1.0, 1.0)
        + 0.15 * np.tanh(ret_dow_fut / 0.01) * np.clip(corr_dow_fut_20, -1.0, 1.0)
        + 0.15 * np.tanh(ret_dji / 0.01) * np.clip(corr_dji_20, -1.0, 1.0)
        + 0.20 * np.tanh(ret_ixic / 0.01) * np.clip(corr_ixic_20, -1.0, 1.0)
        + 0.10 * np.tanh(ret_usdjpy / 0.01) * np.clip(corr_usdjpy_20, -1.0, 1.0)
        - 0.20 * np.tanh(ret_vix / 0.05) * np.clip(corr_vix_20, -1.0, 1.0)
    )

    head_score = static_row.get("headline_score_raw")
    shikiho_score = static_row.get("shikiho_score_overall")
    sector_per_score = static_row.get("sector_adjusted_per_score")
    forecast_per = static_row.get("forecast_per")
    promising_score = static_row.get("promising_score")

    signal = (
        ((promising_score if not pd.isna(promising_score) else 0.0) >= promising_score_min)
        and
        20.0 <= annual_return_pct <= 180.0
        and 10.0 <= quarter_return_pct <= 60.0
        and trend_r2 >= trend_r2_min
        and persistence_20d_pct >= 55.0
        and positive_month_ratio_pct >= 60.0
        and max_drawdown_pct >= -28.0
        and (((head_score if not pd.isna(head_score) else 0) >= 0) or ((shikiho_score if not pd.isna(shikiho_score) else 0) >= 3))
        and (((sector_per_score if not pd.isna(sector_per_score) else 0) >= sector_per_score_min) or pd.isna(forecast_per))
        and ((forecast_per <= 20.0) if not pd.isna(forecast_per) else True)
        and macro_confirm_score >= macro_confirm_min
    )

    signal_score = (
        0.28 * min(max((annual_return_pct - 20.0) / 160.0, 0.0), 1.0)
        + 0.22 * min(max((quarter_return_pct - 10.0) / 50.0, 0.0), 1.0)
        + 0.20 * trend_r2
        + 0.12 * min(max(persistence_20d_pct / 100.0, 0.0), 1.0)
        + 0.08 * min(max(positive_month_ratio_pct / 100.0, 0.0), 1.0)
        + 0.10 * min(max((28.0 + max_drawdown_pct) / 28.0, 0.0), 1.0)
        + macro_score_weight * (0.6 * macro_confirm_score + 0.4 * macro_corr_score)
    )
    return signal, {
        "annual_return_pct": annual_return_pct,
        "quarter_return_pct": quarter_return_pct,
        "trend_r2": trend_r2,
        "max_drawdown_pct": max_drawdown_pct,
        "positive_month_ratio_pct": positive_month_ratio_pct,
        "persistence_20d_pct": persistence_20d_pct,
        "ret_dji": ret_dji,
        "ret_dow_fut": ret_dow_fut,
        "fut_nk_night_ret": fut_nk_night_ret,
        "ret_vix": ret_vix,
        "ret_ixic": ret_ixic,
        "ret_usdjpy": ret_usdjpy,
        "corr_dji_20": corr_dji_20,
        "corr_dow_fut_20": corr_dow_fut_20,
        "corr_nk_night_20": corr_nk_night_20,
        "corr_vix_20": corr_vix_20,
        "corr_ixic_20": corr_ixic_20,
        "corr_usdjpy_20": corr_usdjpy_20,
        "macro_confirm_score": float(macro_confirm_score),
        "macro_corr_score": float(macro_corr_score),
        "signal_score": signal_score,
    }


def compute_top_signal(
    prev_hist: pd.DataFrame,
    prev_date: pd.Timestamp,
    external_df: pd.DataFrame,
    highest_close: float,
    dd_threshold: float,
    nk_threshold: float,
    ma_window: int,
) -> tuple[bool, dict]:
    close = prev_hist["Close"].astype(float)
    prev_close = float(close.iloc[-1])
    prev_close_m1 = float(close.iloc[-2]) if len(close) >= 2 else prev_close
    ma = close.rolling(ma_window).mean().iloc[-1] if len(close) >= ma_window else np.nan
    drawdown_from_peak_pct = (prev_close / float(highest_close) - 1.0) * 100.0 if highest_close > 0 else 0.0
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest 4Q-2 selected stocks with breakout entry.")
    parser.add_argument("--selected-csv", type=Path, default=OUTPUT_DIR / "4q2_selection" / "4q2_selected_candidates.csv")
    parser.add_argument("--price-dir", type=Path, default=PRICE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "4q2_signal_backtest")
    parser.add_argument("--initial-capital", type=float, default=3_000_000.0)
    parser.add_argument("--start-date", type=str, default="2026-01-01")
    parser.add_argument("--end-date", type=str, default="")
    parser.add_argument("--promising-score-min", type=float, default=0.66)
    parser.add_argument("--trend-r2-min", type=float, default=0.50)
    parser.add_argument("--sector-per-score-min", type=float, default=0.55)
    parser.add_argument("--take-profit-pct", type=float, default=8.0)
    parser.add_argument("--stop-loss-pct", type=float, default=5.0)
    parser.add_argument("--entry-limit-pct", type=float, default=1.5)
    parser.add_argument("--entry-mode", type=str, default="breakout_up", choices=["limit_down", "breakout_up"])
    parser.add_argument("--condition-name", type=str, default="condition2")
    parser.add_argument("--macro-confirm-min", type=float, default=-1.0)
    parser.add_argument("--macro-score-weight", type=float, default=0.0)
    parser.add_argument("--macro-size-weight", type=float, default=0.0)
    parser.add_argument("--macro-size-floor", type=float, default=0.5)
    parser.add_argument("--macro-exit-min", type=float, default=-9.0)
    parser.add_argument("--live-news-json", type=Path, default=None)
    parser.add_argument("--live-news-stop-threshold", type=float, default=-9.0)
    parser.add_argument("--historical-news-csv", type=Path, default=DAILY_FEATURES_CSV)
    parser.add_argument("--historical-news-stop-threshold", type=float, default=-9.0)
    parser.add_argument("--early-exit-enable", action="store_true")
    parser.add_argument("--early-dd-threshold", type=float, default=2.0)
    parser.add_argument("--early-nk-threshold", type=float, default=0.0)
    parser.add_argument("--early-ma-window", type=int, default=5)
    parser.add_argument("--early-tp-buffer-pct", type=float, default=1.0)
    parser.add_argument("--early-sl-buffer-pct", type=float, default=2.0)
    args = parser.parse_args()

    selected = load_selected(args.selected_csv)
    selected = selected.drop_duplicates(subset=["ticker"], keep="first").copy()
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
    live_news_signal = load_live_news_signal(args.live_news_json)
    historical_news_df = load_historical_news_signal(args.historical_news_csv)

    price_map = {}
    all_dates = set()
    start_date = pd.Timestamp(args.start_date)
    end_date_limit = pd.Timestamp(args.end_date) if str(args.end_date).strip() else None
    for ticker in selected["ticker"]:
        price_path = args.price_dir / f"{ticker.replace('.', '_')}.csv"
        if not price_path.exists():
            continue
        df = pd.read_csv(price_path)
        if "Date" not in df.columns or "Close" not in df.columns:
            continue
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").set_index("Date")
        df = df[df.index >= pd.Timestamp("2025-01-01")].copy()
        if end_date_limit is not None:
            df = df[df.index <= end_date_limit].copy()
        if df.empty:
            continue
        price_map[ticker] = df
        all_dates.update(df[df.index >= start_date].index.tolist())

    dates = sorted(all_dates)
    if not dates:
        raise SystemExit("No tradable dates found for the selected universe and date range.")
    latest_date = max(dates)

    cash = float(args.initial_capital)
    positions: dict[str, dict] = {}
    trades = []
    equity_curve = []
    prev_signal = {ticker: False for ticker in price_map}

    for date in dates:
        signal_today = {}
        metrics_today = {}
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
                df,
                signal_date,
                static_lookup[ticker],
                external_df,
                args.promising_score_min,
                args.trend_r2_min,
                args.sector_per_score_min,
                args.macro_confirm_min,
                args.macro_score_weight,
            )
            metrics["signal_basis_date"] = signal_date.date().isoformat()
            signal_today[ticker] = sig
            metrics_today[ticker] = metrics

        for ticker in list(positions.keys()):
            df = price_map[ticker]
            if date not in df.index:
                continue
            day = df.loc[date]
            day_open = float(day["Open"]) if "Open" in day.index and not pd.isna(day["Open"]) else float(day["Close"])
            day_high = float(day["High"]) if "High" in day.index and not pd.isna(day["High"]) else float(day["Close"])
            day_low = float(day["Low"]) if "Low" in day.index and not pd.isna(day["Low"]) else float(day["Close"])
            price = float(df.loc[date, "Close"])
            entry_price = positions[ticker]["entry_price"]
            ret = price / entry_price - 1.0
            take_profit = args.take_profit_pct / 100.0 if args.take_profit_pct >= 0 else None
            stop_loss = args.stop_loss_pct / 100.0 if args.stop_loss_pct >= 0 else None
            exit_reason = None
            exit_price = None

            if args.early_exit_enable:
                prev_idx = df.index[df.index < date]
                if len(prev_idx) > 0:
                    prev_date = prev_idx[-1]
                    prev_hist = df.loc[:prev_date].copy()
                    top_flag, top_metrics = compute_top_signal(
                        prev_hist=prev_hist,
                        prev_date=prev_date,
                        external_df=external_df,
                        highest_close=float(positions[ticker].get("highest_close", entry_price)),
                        dd_threshold=args.early_dd_threshold,
                        nk_threshold=args.early_nk_threshold,
                        ma_window=args.early_ma_window,
                    )
                    prev_close = float(prev_hist["Close"].iloc[-1])
                    unrealized_prev_ret = (prev_close / entry_price - 1.0) * 100.0
                    if top_flag and unrealized_prev_ret < args.take_profit_pct:
                        if unrealized_prev_ret > 0:
                            early_tp_price = prev_close * (1.0 + args.early_tp_buffer_pct / 100.0)
                            if day_high >= early_tp_price:
                                exit_reason = "early_take_profit_top"
                                exit_price = max(day_open, early_tp_price)
                        else:
                            early_sl_price = prev_close * (1.0 - args.early_sl_buffer_pct / 100.0)
                            if day_low <= early_sl_price:
                                exit_reason = "early_stop_top"
                                exit_price = min(day_open, early_sl_price)
                        if exit_reason is not None:
                            positions[ticker]["top_metrics"] = top_metrics

            if exit_reason is None and take_profit is not None and ret >= take_profit:
                exit_reason = "take_profit"
                exit_price = price
            elif exit_reason is None and stop_loss is not None and ret <= -stop_loss:
                exit_reason = "stop_loss"
                exit_price = price
            elif (
                exit_reason is None
                and
                args.macro_exit_min > -9.0
                and metrics_today.get(ticker, {}).get("macro_confirm_score") is not None
                and float(metrics_today.get(ticker, {}).get("macro_confirm_score", 0.0)) <= args.macro_exit_min
            ):
                exit_reason = "macro_exit"
                exit_price = price
            elif exit_reason is None and not signal_today.get(ticker, False):
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
                    }
                )
                del positions[ticker]
                continue

            positions[ticker]["highest_close"] = max(float(positions[ticker].get("highest_close", entry_price)), price)

        candidates = []
        live_news_block = False
        if live_news_signal is not None and args.live_news_stop_threshold > -9.0:
            agg = live_news_signal.get("aggregate", {})
            live_sent = float(agg.get("sentiment_score", 0.0))
            live_news_block = live_sent <= args.live_news_stop_threshold
        historical_news_block = False
        if not historical_news_df.empty and args.historical_news_stop_threshold > -9.0:
            hist_row = historical_news_df.loc[historical_news_df.index <= date].tail(1)
            if not hist_row.empty:
                historical_news_block = float(hist_row["news_sentiment_score"].iloc[0]) <= args.historical_news_stop_threshold
        for ticker, sig in signal_today.items():
            if ticker in positions:
                continue
            if sig and not prev_signal.get(ticker, False) and not live_news_block and not historical_news_block:
                df = price_map[ticker]
                if date not in df.index:
                    continue
                static = static_lookup[ticker]
                candidates.append(
                    {
                        "ticker": ticker,
                        "simple_sector": static.get("simple_sector", static.get("sector", "")),
                        "sector_33": static.get("sector_33", "-"),
                        "signal_score": metrics_today.get(ticker, {}).get("signal_score", 0.0),
                        "macro_confirm_score": metrics_today.get(ticker, {}).get("macro_confirm_score", 0.0),
                        "macro_corr_score": metrics_today.get(ticker, {}).get("macro_corr_score", 0.0),
                        "promising_score": static.get("promising_score", 0.0),
                        "price": float(df.loc[date, "Close"]),
                        "signal_basis_date": metrics_today.get(ticker, {}).get("signal_basis_date", ""),
                    }
                )

        buy_list = sorted(candidates, key=lambda x: (x["signal_score"], x["promising_score"]), reverse=True)

        remaining = len(buy_list)
        if buy_list and args.macro_size_weight > 0:
            raw_weights = []
            for c in buy_list:
                macro_size_score = 0.6 * float(c.get("macro_confirm_score", 0.0)) + 0.4 * float(c.get("macro_corr_score", 0.0))
                scaled = 1.0 + args.macro_size_weight * macro_size_score
                raw_weights.append(max(args.macro_size_floor, scaled))
            total_weight = float(sum(raw_weights))
        else:
            raw_weights = [1.0] * len(buy_list)
            total_weight = float(len(raw_weights)) if raw_weights else 0.0
        for idx, c in enumerate(buy_list):
            df = price_map[c["ticker"]]
            day = df.loc[date]
            prev_close = float(df.loc[pd.Timestamp(c["signal_basis_date"]), "Close"]) if c["signal_basis_date"] else float(day["Close"])
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
            if total_weight > 0:
                alloc = cash * (raw_weights[idx] / total_weight)
                total_weight -= raw_weights[idx]
            else:
                alloc = cash / remaining if remaining > 0 else 0.0
            lot_cost = fill_price * 100
            shares = int(alloc // lot_cost) * 100
            if shares >= 100 and shares * fill_price <= cash:
                cash -= shares * fill_price
                positions[c["ticker"]] = {
                    "shares": shares,
                    "entry_price": fill_price,
                    "entry_date": date,
                    "simple_sector": c["simple_sector"],
                    "sector_33": c["sector_33"],
                    "highest_close": float(fill_price),
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
                        "macro_confirm_score": c["macro_confirm_score"],
                        "macro_corr_score": c["macro_corr_score"],
                        "signal_basis_date": metrics_today.get(c["ticker"], {}).get("signal_basis_date"),
                        "reason": f"buy_{args.entry_mode}",
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
            }
        )
        prev_signal = signal_today

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
            }
        )
        del positions[ticker]

    trade_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve)
    max_dd_pct = 0.0
    if not equity_df.empty:
        eq = equity_df["equity"].astype(float)
        dd = eq / eq.cummax() - 1.0
        max_dd_pct = float(dd.min()) * 100.0

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
                }
            )
    roundtrip_df = pd.DataFrame(roundtrip)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trade_df.to_csv(args.output_dir / "trade_log.csv", index=False, encoding="utf-8-sig")
    equity_df.to_csv(args.output_dir / "equity_curve.csv", index=False, encoding="utf-8-sig")
    roundtrip_df.to_csv(args.output_dir / "roundtrip_trades.csv", index=False, encoding="utf-8-sig")

    summary = {
        "condition_name": args.condition_name,
        "promising_score_min": args.promising_score_min,
        "trend_r2_min": args.trend_r2_min,
        "sector_per_score_min": args.sector_per_score_min,
        "take_profit_pct": args.take_profit_pct,
        "stop_loss_pct": args.stop_loss_pct,
        "entry_limit_pct": args.entry_limit_pct,
        "entry_mode": args.entry_mode,
        "macro_confirm_min": args.macro_confirm_min,
        "macro_score_weight": args.macro_score_weight,
        "macro_size_weight": args.macro_size_weight,
        "macro_size_floor": args.macro_size_floor,
        "macro_exit_min": args.macro_exit_min,
        "live_news_json": str(args.live_news_json) if args.live_news_json else "",
        "live_news_stop_threshold": args.live_news_stop_threshold,
        "historical_news_csv": str(args.historical_news_csv) if args.historical_news_csv else "",
        "historical_news_stop_threshold": args.historical_news_stop_threshold,
        "early_exit_enable": args.early_exit_enable,
        "early_dd_threshold": args.early_dd_threshold,
        "early_nk_threshold": args.early_nk_threshold,
        "early_ma_window": args.early_ma_window,
        "early_tp_buffer_pct": args.early_tp_buffer_pct,
        "early_sl_buffer_pct": args.early_sl_buffer_pct,
        "period_start": str(start_date.date()),
        "period_end": str(latest_date.date()),
        "initial_capital": float(args.initial_capital),
        "final_capital": float(cash),
        "total_return_pct": (float(cash) / float(args.initial_capital) - 1.0) * 100.0,
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

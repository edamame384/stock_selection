from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from projects.market_news_automation.paths import DAILY_FEATURES_CSV
from projects.shikiho_text_parser.backtest_4q2_signals import (
    calc_signal,
    load_historical_news_signal,
    load_live_news_signal,
    load_selected,
)
from src.stock_signal import (
    download_additional_macro_features,
    download_dow_futures_feature,
    download_index_returns,
    download_nikkei_futures_night_feature,
)


def calc_fail_flags(static_row: dict, metrics: dict, promising_score_min: float, trend_r2_min: float, sector_per_score_min: float, macro_confirm_min: float) -> dict[str, bool]:
    head_score = static_row.get("headline_score_raw")
    shikiho_score = static_row.get("shikiho_score_overall")
    sector_per_score = static_row.get("sector_adjusted_per_score")
    forecast_per = static_row.get("forecast_per")
    promising_score = static_row.get("promising_score")

    return {
        "promising_score": (promising_score if not pd.isna(promising_score) else 0.0) >= promising_score_min,
        "annual_return_range": 20.0 <= metrics.get("annual_return_pct", -9e9) <= 180.0,
        "quarter_return_range": 10.0 <= metrics.get("quarter_return_pct", -9e9) <= 60.0,
        "trend_r2": metrics.get("trend_r2", -9e9) >= trend_r2_min,
        "persistence_20d_pct": metrics.get("persistence_20d_pct", -9e9) >= 55.0,
        "positive_month_ratio_pct": metrics.get("positive_month_ratio_pct", -9e9) >= 60.0,
        "max_drawdown_pct": metrics.get("max_drawdown_pct", 9e9) >= -28.0,
        "headline_or_shikiho": ((head_score if not pd.isna(head_score) else 0) >= 0)
        or ((shikiho_score if not pd.isna(shikiho_score) else 0) >= 3),
        "sector_adjusted_per_score": (((sector_per_score if not pd.isna(sector_per_score) else 0) >= sector_per_score_min) or pd.isna(forecast_per)),
        "forecast_per": ((forecast_per <= 20.0) if not pd.isna(forecast_per) else True),
        "macro_confirm_score": metrics.get("macro_confirm_score", -9e9) >= macro_confirm_min,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Explain why condition2 produced no trades for the selected Q4 universe.")
    parser.add_argument("--selected-csv", type=Path, required=True)
    parser.add_argument("--price-dir", type=Path, default=Path("data/prices"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-date", type=str, default="2025-10-01")
    parser.add_argument("--end-date", type=str, default="2025-12-31")
    parser.add_argument("--promising-score-min", type=float, default=0.66)
    parser.add_argument("--trend-r2-min", type=float, default=0.50)
    parser.add_argument("--sector-per-score-min", type=float, default=0.55)
    parser.add_argument("--entry-limit-pct", type=float, default=1.5)
    parser.add_argument("--macro-confirm-min", type=float, default=-1.0)
    parser.add_argument("--macro-score-weight", type=float, default=0.0)
    parser.add_argument("--live-news-json", type=Path, default=None)
    parser.add_argument("--historical-news-csv", type=Path, default=DAILY_FEATURES_CSV)
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
    live_news_signal = load_live_news_signal(args.live_news_json)
    historical_news_df = load_historical_news_signal(args.historical_news_csv)

    start_date = pd.Timestamp(args.start_date)
    end_date = pd.Timestamp(args.end_date)
    output_rows = []
    reason_rows = []
    overall_fail_counter = Counter()

    for ticker in selected["ticker"]:
        price_path = args.price_dir / f"{ticker.replace('.', '_')}.csv"
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

        prev_signal = False
        signal_true_days = 0
        turn_on_days = 0
        breakout_fill_days = 0
        max_signal_score = -np.inf
        fail_counter = Counter()
        last_fail_flags = None
        last_metrics = {}

        live_news_block = False
        if live_news_signal is not None:
            agg = live_news_signal.get("aggregate", {})
            live_news_block = float(agg.get("sentiment_score", 0.0)) <= -9.0

        tradable_dates = [d for d in df.index if d >= start_date]
        for date in tradable_dates:
            prev_idx = df.index[df.index < date]
            if len(prev_idx) == 0:
                prev_signal = False
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

            historical_news_block = False
            if not historical_news_df.empty:
                hist_row = historical_news_df.loc[historical_news_df.index <= date].tail(1)
                if not hist_row.empty:
                    historical_news_block = float(hist_row["news_sentiment_score"].iloc[0]) <= -9.0

            if sig:
                signal_true_days += 1
                max_signal_score = max(max_signal_score, float(metrics.get("signal_score", 0.0)))
                if not prev_signal and not live_news_block and not historical_news_block:
                    turn_on_days += 1
                    day = df.loc[date]
                    prev_close = float(df.loc[signal_date, "Close"])
                    trigger_price = prev_close * (1.0 + args.entry_limit_pct / 100.0)
                    day_high = float(day["High"]) if "High" in day.index and not pd.isna(day["High"]) else float(day["Close"])
                    if day_high >= trigger_price:
                        breakout_fill_days += 1
            else:
                flags = calc_fail_flags(
                    static_lookup[ticker],
                    metrics,
                    args.promising_score_min,
                    args.trend_r2_min,
                    args.sector_per_score_min,
                    args.macro_confirm_min,
                )
                last_fail_flags = flags
                last_metrics = metrics
                for name, passed in flags.items():
                    if not passed:
                        fail_counter[name] += 1
                        overall_fail_counter[name] += 1

            prev_signal = sig

        top_fail_reason = fail_counter.most_common(1)[0][0] if fail_counter else ""
        output_rows.append(
            {
                "ticker": ticker,
                "company_name": static_lookup[ticker].get("company_name", ""),
                "sector": static_lookup[ticker].get("sector", ""),
                "learning_date": static_lookup[ticker].get("learning_date", ""),
                "signal_true_days": signal_true_days,
                "turn_on_days": turn_on_days,
                "breakout_fill_days": breakout_fill_days,
                "max_signal_score": None if max_signal_score == -np.inf else max_signal_score,
                "top_fail_reason": top_fail_reason,
                "fail_promising_score": fail_counter["promising_score"],
                "fail_annual_return_range": fail_counter["annual_return_range"],
                "fail_quarter_return_range": fail_counter["quarter_return_range"],
                "fail_trend_r2": fail_counter["trend_r2"],
                "fail_persistence_20d_pct": fail_counter["persistence_20d_pct"],
                "fail_positive_month_ratio_pct": fail_counter["positive_month_ratio_pct"],
                "fail_max_drawdown_pct": fail_counter["max_drawdown_pct"],
                "fail_headline_or_shikiho": fail_counter["headline_or_shikiho"],
                "fail_sector_adjusted_per_score": fail_counter["sector_adjusted_per_score"],
                "fail_forecast_per": fail_counter["forecast_per"],
                "fail_macro_confirm_score": fail_counter["macro_confirm_score"],
                "last_signal_basis_date": last_metrics.get("signal_basis_date", ""),
                "last_annual_return_pct": last_metrics.get("annual_return_pct"),
                "last_quarter_return_pct": last_metrics.get("quarter_return_pct"),
                "last_trend_r2": last_metrics.get("trend_r2"),
                "last_persistence_20d_pct": last_metrics.get("persistence_20d_pct"),
                "last_positive_month_ratio_pct": last_metrics.get("positive_month_ratio_pct"),
                "last_max_drawdown_pct": last_metrics.get("max_drawdown_pct"),
                "last_macro_confirm_score": last_metrics.get("macro_confirm_score"),
            }
        )
        if fail_counter:
            for reason, count in fail_counter.most_common():
                reason_rows.append(
                    {
                        "ticker": ticker,
                        "company_name": static_lookup[ticker].get("company_name", ""),
                        "reason": reason,
                        "count": count,
                    }
                )

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    analysis_df = pd.DataFrame(output_rows).sort_values(
        ["signal_true_days", "turn_on_days", "breakout_fill_days", "max_signal_score"],
        ascending=[True, True, True, False],
    )
    analysis_df.to_csv(out_dir / "no_signal_analysis.csv", index=False, encoding="utf-8-sig")

    reason_df = pd.DataFrame(reason_rows).sort_values(["ticker", "count"], ascending=[True, False])
    reason_df.to_csv(out_dir / "no_signal_reason_counts.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(
        [{"reason": reason, "count": count} for reason, count in overall_fail_counter.most_common()]
    )
    summary.to_csv(out_dir / "no_signal_reason_summary.csv", index=False, encoding="utf-8-sig")

    print(json.dumps(
        {
            "tickers": int(len(analysis_df)),
            "signal_true_total": int(analysis_df["signal_true_days"].sum()) if not analysis_df.empty else 0,
            "turn_on_total": int(analysis_df["turn_on_days"].sum()) if not analysis_df.empty else 0,
            "breakout_fill_total": int(analysis_df["breakout_fill_days"].sum()) if not analysis_df.empty else 0,
            "top_reasons": summary.head(10).to_dict("records"),
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

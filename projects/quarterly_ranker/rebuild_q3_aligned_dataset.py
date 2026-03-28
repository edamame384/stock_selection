from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from projects.quarterly_ranker.rank_quarterly_promising_stocks import calc_max_drawdown, calc_r2_log_trend, percentile_score
from projects.quarterly_ranker.operational_csv_utils import write_operational_csv
from projects.quarterly_ranker.select_q3_pre_jul_candidates import (
    attach_realized_returns,
    enrich_with_numeric_ocr,
    prepare_mapped_image_dir,
    select_candidates,
)


def add_learning_date(df: pd.DataFrame, learning_date: str) -> pd.DataFrame:
    out = df.copy()
    out["learning_date"] = learning_date
    out["training_cutoff_date"] = learning_date
    return out


def calc_pre_jul_metrics_aligned(symbol: str, price_path: Path) -> dict[str, float] | None:
    df = pd.read_csv(price_path)
    if "Date" not in df.columns or "Close" not in df.columns:
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")

    end = pd.Timestamp("2025-06-30")
    hist = df[df.index <= end].copy()
    if len(hist) < 120:
        return None

    trailing = hist.tail(min(252, len(hist))).copy()
    qdf = hist.tail(min(63, len(hist))).copy()
    if len(trailing) < 120 or len(qdf) < 20:
        return None

    monthly = qdf.resample("ME").agg({"High": "max", "Low": "min"}).dropna()
    if monthly.empty:
        return None

    annual_return = float(trailing["Close"].iloc[-1] / trailing["Close"].iloc[0] - 1.0)
    trailing_return = annual_return
    quarter_return = float(qdf["Close"].iloc[-1] / qdf["Close"].iloc[0] - 1.0)
    avg_monthly_range = float(((monthly["High"] / monthly["Low"]) - 1.0).mean())
    max_monthly_range = float(((monthly["High"] / monthly["Low"]) - 1.0).max())
    trend_r2 = calc_r2_log_trend(trailing["Close"])
    max_drawdown = calc_max_drawdown(trailing["Close"])
    monthly_close = trailing["Close"].resample("ME").last().dropna()
    monthly_ret = monthly_close.pct_change().dropna()
    positive_month_ratio = float((monthly_ret > 0).mean()) if len(monthly_ret) > 0 else np.nan
    ma20 = trailing["Close"].rolling(20).mean()
    ma60 = trailing["Close"].rolling(60).mean()
    last20 = trailing.iloc[-20:].copy()
    ma20_last20 = ma20.reindex(last20.index)
    ma60_last20 = ma60.reindex(last20.index)
    persistence_20d = float(((last20["Close"] > ma20_last20) & (ma20_last20 > ma60_last20)).mean())
    end_to_trailing_high = float(trailing["Close"].iloc[-1] / trailing["High"].max())

    return {
        "ticker": symbol,
        "annual_return_pct": annual_return * 100.0,
        "trailing_return_pct": trailing_return * 100.0,
        "quarter_return_pct": quarter_return * 100.0,
        "avg_monthly_high_low_change_pct": avg_monthly_range * 100.0,
        "max_monthly_high_low_change_pct": max_monthly_range * 100.0,
        "trend_r2": trend_r2,
        "max_drawdown_pct": max_drawdown * 100.0,
        "positive_month_ratio_pct": positive_month_ratio * 100.0,
        "end_to_trailing_high_pct": end_to_trailing_high * 100.0,
        "persistence_20d_pct": persistence_20d * 100.0,
        "quarter_end_close": float(hist["Close"].iloc[-1]),
        "price_rows": int(len(df)),
    }


def build_pre_jul_base_aligned(library: pd.DataFrame, price_dir: Path) -> pd.DataFrame:
    rows = []
    for _, row in library[library["resolved"] == True].iterrows():
        ticker = str(row["ticker"])
        price_path = price_dir / f"{ticker.replace('.', '_')}.csv"
        if not price_path.exists():
            continue
        metrics = calc_pre_jul_metrics_aligned(ticker, price_path)
        if metrics is None:
            continue
        metrics["company_name"] = row["company_name"]
        metrics["sector"] = row["sector"]
        rows.append(metrics)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["score"] = (
        0.20 * percentile_score(df["annual_return_pct"], ascending=True)
        + 0.20 * percentile_score(df["trailing_return_pct"], ascending=True)
        + 0.14 * percentile_score(df["quarter_return_pct"], ascending=True)
        + 0.14 * percentile_score(df["trend_r2"], ascending=True)
        + 0.12 * percentile_score(df["positive_month_ratio_pct"], ascending=True)
        + 0.12 * percentile_score(df["end_to_trailing_high_pct"], ascending=True)
        + 0.04 * percentile_score(df["persistence_20d_pct"], ascending=True)
        + 0.02 * percentile_score(df["max_drawdown_pct"], ascending=False)
        + 0.02 * percentile_score(df["avg_monthly_high_low_change_pct"], ascending=False)
    )
    df = df.sort_values(["score", "annual_return_pct", "quarter_return_pct", "trend_r2"], ascending=[False, False, False, False]).reset_index(drop=True)
    df["quarter"] = "2025Q3_PRE_ALIGNED"
    df["rank"] = np.arange(1, len(df) + 1)
    return df[
        [
            "quarter", "rank", "ticker", "company_name", "sector", "score",
            "annual_return_pct", "quarter_return_pct", "avg_monthly_high_low_change_pct",
            "max_monthly_high_low_change_pct", "trend_r2", "max_drawdown_pct",
            "positive_month_ratio_pct", "end_to_trailing_high_pct", "persistence_20d_pct",
            "quarter_end_close", "price_rows", "trailing_return_pct",
        ]
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild 3Q candidates with aligned 252/63-day price windows up to 2025-06-30.")
    parser.add_argument("--library-csv", type=Path, default=Path("projects/quarterly_ranker/output/q3_pre_analysis_20250630_full/q3_image_library.csv"))
    parser.add_argument("--image-dir", type=Path, default=Path(r"C:\Users\mitsu\OneDrive\ドキュメント\四季報DB2025\3Q"))
    parser.add_argument("--price-dir", type=Path, default=Path("data/prices"))
    parser.add_argument("--out-dir", type=Path, default=Path("projects/quarterly_ranker/output/q3_pre_analysis_20250630_aligned"))
    parser.add_argument("--learning-date", default="2025-06-30")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    library = pd.read_csv(args.library_csv)
    library.to_csv(args.out_dir / "q3_image_library.csv", index=False, encoding="utf-8-sig")

    base_df = build_pre_jul_base_aligned(library, args.price_dir)
    base_df = add_learning_date(base_df, args.learning_date)
    base_csv = args.out_dir / "q3_pre_base_candidates.csv"
    base_df.to_csv(base_csv, index=False, encoding="utf-8-sig")

    prepare_mapped_image_dir(library, args.image_dir, args.out_dir / "mapped_images")
    detail = enrich_with_numeric_ocr(base_df, library)
    detail = add_learning_date(detail, args.learning_date)
    detail_csv = args.out_dir / "q3_pre_shikiho_feature_ranking.csv"
    detail.to_csv(detail_csv, index=False, encoding="utf-8-sig")

    selected = select_candidates(detail)
    selected = attach_realized_returns(selected, args.price_dir)
    selected = add_learning_date(selected, args.learning_date)
    selected_csv = args.out_dir / "q3_pre_selected_candidates.csv"
    selected.to_csv(selected_csv, index=False, encoding="utf-8-sig")
    operational_selected_csv = args.out_dir / "operational" / "q3_pre_selected_candidates_operational.csv"
    write_operational_csv(selected, operational_selected_csv)

    summary = pd.DataFrame([
        {
            "dataset_id": "q3_pre_analysis_20250630_aligned",
            "learning_date": args.learning_date,
            "training_cutoff_date": args.learning_date,
            "price_axis": "aligned_252_63_trading_days",
            "image_count_total": int(len(library)),
            "selected_count": int(len(selected)),
            "output_dir": str(args.out_dir),
            "source_image_dir": str(args.image_dir),
            "source_library_csv": str(args.library_csv),
        },
        {
            "dataset_id": "q3_pre_analysis_20250630_full",
            "learning_date": args.learning_date,
            "training_cutoff_date": args.learning_date,
            "price_axis": "ytd_plus_calendar_quarter",
            "image_count_total": int(len(library)),
            "selected_count": None,
            "output_dir": "projects/quarterly_ranker/output/q3_pre_analysis_20250630_full",
            "source_image_dir": str(args.image_dir),
            "source_library_csv": str(args.library_csv),
        },
    ])
    summary.to_csv(args.out_dir / "q3_learning_dataset_manifest.csv", index=False, encoding="utf-8-sig")

    print(f"[OUT] base={base_csv}")
    print(f"[OUT] detail={detail_csv}")
    print(f"[OUT] selected={selected_csv}")
    print(f"[OUT] selected_operational={operational_selected_csv}")
    print(f"[INFO] selected={len(selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

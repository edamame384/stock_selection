from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def safe_bool_series(length: int, value: bool) -> pd.Series:
    return pd.Series([value] * length)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert q4_pre selected candidates into condition2-compatible input.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    out = pd.DataFrame()
    out["ticker"] = df["ticker"]
    out["company_name"] = df["company_name"]
    out["market"] = ""
    out["simple_sector"] = df["sector"]
    out["sector"] = df["sector"]
    out["feature_summary"] = ""
    out["segment_summary"] = ""
    out["flags"] = ""
    out["headline_positive_count"] = 0
    out["headline_negative_count"] = 0
    out["headline_score_raw"] = 0.0
    out["shikiho_score_overall"] = 3.0
    out["score_growth"] = pd.to_numeric(df.get("growth_quality_score"), errors="coerce").fillna(0.0) * 5.0
    out["score_profitability"] = pd.to_numeric(df.get("finance_quality_score"), errors="coerce").fillna(0.0) * 5.0
    out["score_safety"] = pd.to_numeric(df.get("finance_quality_score"), errors="coerce").fillna(0.0) * 5.0
    out["score_undervalued"] = pd.to_numeric(df.get("sector_adjusted_per_score"), errors="coerce").fillna(0.0) * 5.0
    out["score_momentum"] = pd.to_numeric(df.get("score"), errors="coerce").fillna(0.0) * 5.0
    out["forecast_per"] = pd.to_numeric(df.get("ocr_per"), errors="coerce")
    out["actual_pbr"] = pd.to_numeric(df.get("ocr_pbr"), errors="coerce")
    out["forecast_yield"] = pd.to_numeric(df.get("ocr_yield"), errors="coerce")
    out["ytd_return_text_pct"] = pd.to_numeric(df.get("annual_return_pct"), errors="coerce")
    out["ma200_gap_pct"] = pd.NA
    out["equity_ratio_pct"] = pd.NA
    out["sales_growth_pct"] = pd.NA
    out["op_growth_pct"] = pd.NA
    out["np_growth_pct"] = pd.NA
    out["eps_growth_pct"] = pd.NA
    out["div_growth_pct"] = pd.NA
    out["annual_return_pct"] = pd.to_numeric(df.get("annual_return_pct"), errors="coerce")
    out["quarter_return_pct"] = pd.to_numeric(df.get("quarter_return_pct"), errors="coerce")
    out["trend_r2"] = pd.to_numeric(df.get("trend_r2"), errors="coerce")
    out["max_drawdown_pct"] = -pd.to_numeric(df.get("max_drawdown_pct"), errors="coerce").abs()
    out["positive_month_ratio_pct"] = pd.to_numeric(df.get("positive_month_ratio_pct"), errors="coerce")
    out["persistence_20d_pct"] = pd.to_numeric(df.get("persistence_20d_pct"), errors="coerce")
    out["year_end_close"] = pd.to_numeric(df.get("quarter_end_close"), errors="coerce")
    out["sector_33"] = df["sector"]
    out["sector_adjusted_per_score"] = pd.to_numeric(df.get("sector_adjusted_per_score"), errors="coerce")
    out["fundamental_score"] = pd.to_numeric(df.get("finance_quality_score"), errors="coerce")
    out["price_score"] = pd.to_numeric(df.get("growth_quality_score"), errors="coerce")
    out["valuation_score"] = pd.to_numeric(df.get("sector_adjusted_per_score"), errors="coerce")
    out["promising_score"] = pd.to_numeric(df.get("score"), errors="coerce")
    out["selected"] = safe_bool_series(len(df), True)
    out["selection_reason"] = df.get("adopt_reason", "").astype(str)

    # Preserve provenance from the image-based 4Q analysis.
    out["source_quarter"] = df.get("quarter", "").astype(str)
    out["learning_date"] = df.get("learning_date", "").astype(str)
    out["training_cutoff_date"] = df.get("training_cutoff_date", "").astype(str)
    out["ocr_style_label"] = df.get("ocr_style_label", "").astype(str)
    out["image_file"] = df.get("image_file", "").astype(str)
    out["realized_return_pct"] = pd.to_numeric(df.get("realized_return_pct"), errors="coerce")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"[OUT] {args.output_csv}")
    print(f"[INFO] rows={len(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from projects.quarterly_ranker.extract_q2_shikiho_features import ocr_image_rapid
from projects.quarterly_ranker.rank_quarterly_promising_stocks import (
    calc_max_drawdown,
    calc_r2_log_trend,
    percentile_score,
)
from projects.quarterly_ranker.extract_q2_shikiho_features import (
    assess_metric_confidence,
    compute_sector_adjusted_per_score,
    extract_compound_percentage,
    extract_metric_from_sources,
)
from projects.quarterly_ranker.operational_csv_utils import write_operational_csv


def normalize_symbol(raw: str) -> str:
    symbol = str(raw).strip().upper()
    if not symbol:
        return symbol
    if symbol.startswith("TYO:"):
        return f"{symbol.split(':', 1)[1]}.T"
    if symbol.endswith(".T"):
        return symbol
    return f"{symbol}.T"


def code_from_symbol(symbol: str) -> str:
    return symbol.replace(".T", "")


def load_name_master(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame(
        {
            "code": df["コード"].astype(str).str.upper(),
            "ticker": [normalize_symbol(v) for v in df["コード"].astype(str)],
            "company_name": df["銘柄名"].astype(str),
            "sector": df["33業種"].astype(str) if "33業種" in df.columns else "",
        }
    )
    return out.drop_duplicates(subset=["code"], keep="first")


def save_metrics_crop(image_path: Path, out_path: Path) -> Path:
    img = Image.open(image_path)
    w, h = img.size
    crop = img.crop((int(w * 0.77), 0, w, int(h * 0.30)))
    crop = ImageOps.grayscale(crop)
    crop = ImageOps.autocontrast(crop)
    crop = crop.resize((crop.width * 6, crop.height * 6))
    crop = crop.convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path)
    return out_path


def resolve_code_from_metrics_text(text: str, valid_codes: set[str]) -> str | None:
    tokens = re.findall(r"[0-9]{3,4}[A-Z]?", str(text).upper())
    for token in reversed(tokens):
        if token in valid_codes:
            return token
    return None


def build_2024_q2_library(image_dir: Path, name_master: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    valid_codes = set(name_master["code"].astype(str))
    rows = []
    for image_path in sorted(image_dir.glob("*.png"), key=lambda p: int(p.stem)):
        crop_path = cache_dir / f"{image_path.stem}_metrics.png"
        if not crop_path.exists():
            save_metrics_crop(image_path, crop_path)
        text = ocr_image_rapid(crop_path)
        code = resolve_code_from_metrics_text(text, valid_codes)
        rows.append(
            {
                "image_file": image_path.name,
                "image_no": int(image_path.stem),
                "ocr_metrics_text": text,
                "code": code,
            }
        )
    lib = pd.DataFrame(rows)
    lib = lib.merge(name_master, how="left", on="code")
    lib["resolved"] = lib["ticker"].notna()
    return lib


def prepare_mapped_image_dir(library: pd.DataFrame, source_dir: Path, mapped_dir: Path) -> Path:
    mapped_dir.mkdir(parents=True, exist_ok=True)
    for old_file in mapped_dir.glob("*.png"):
        old_file.unlink()
    for _, row in library[library["resolved"] == True].iterrows():
        src = source_dir / str(row["image_file"])
        dst = mapped_dir / f"{str(row['code'])}.png"
        if not src.exists():
            continue
        try:
            os.link(src, dst)
        except Exception:
            shutil.copy2(src, dst)
    return mapped_dir


def calc_pre_jul_2024_metrics(symbol: str, price_path: Path) -> dict[str, float] | None:
    df = pd.read_csv(price_path)
    if "Date" not in df.columns or "Close" not in df.columns:
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")

    end = pd.Timestamp("2024-06-30")
    q_start = pd.Timestamp("2024-04-01")
    y_start = pd.Timestamp("2024-01-01")
    hist = df[df.index <= end].copy()
    if hist.empty:
        return None
    qdf = hist[(hist.index >= q_start) & (hist.index <= end)].copy()
    trailing = hist.tail(252).copy()
    year_df = hist[(hist.index >= y_start) & (hist.index <= end)].copy()
    if len(qdf) < 20 or len(trailing) < 120 or len(year_df) < 100:
        return None

    monthly = qdf.resample("ME").agg({"High": "max", "Low": "min"}).dropna()
    if monthly.empty:
        return None

    annual_return = float(year_df["Close"].iloc[-1] / year_df["Close"].iloc[0] - 1.0)
    trailing_return = float(trailing["Close"].iloc[-1] / trailing["Close"].iloc[0] - 1.0)
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
        "quarter_end_close": float(qdf["Close"].iloc[-1]),
        "price_rows": int(len(df)),
    }


def build_pre_jul_2024_base(library: pd.DataFrame, price_dir: Path) -> pd.DataFrame:
    rows = []
    for _, row in library[library["resolved"] == True].iterrows():
        ticker = str(row["ticker"])
        price_path = price_dir / f"{ticker.replace('.', '_')}.csv"
        if not price_path.exists():
            continue
        metrics = calc_pre_jul_2024_metrics(ticker, price_path)
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
    df["quarter"] = "2024Q2_PRE"
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


def enrich_with_numeric_ocr(base_df: pd.DataFrame, library: pd.DataFrame) -> pd.DataFrame:
    metric_rows = []
    library_by_ticker = library[library["resolved"] == True].drop_duplicates(subset=["ticker"], keep="first")
    for _, row in library_by_ticker.iterrows():
        ticker = str(row["ticker"])
        text = str(row["ocr_metrics_text"])
        per, per_text = extract_metric_from_sources([text], "予想PER", 1.0, 120.0)
        pbr, pbr_text = extract_metric_from_sources([text], "実PBR", 0.1, 20.0)
        if pbr is None:
            pbr, pbr_text = extract_metric_from_sources([text], "PBR", 0.1, 20.0)
        roe, roe_text = extract_metric_from_sources([text], "予想ROE", 0.1, 50.0, allow_decimal_repair=True)
        roa, roa_text = extract_metric_from_sources([text], "予想ROA", 0.1, 20.0, allow_decimal_repair=True)
        if roa is None:
            roa, roa_text = extract_metric_from_sources([text], "ROA", 0.1, 20.0, allow_decimal_repair=True)
        yld, yld_text = extract_metric_from_sources([text], "総利回り", 0.1, 20.0)
        if yld is None:
            yld = extract_compound_percentage(text, 0.1, 20.0)
            yld_text = text
        metric_rows.append(
            {
                "ticker": ticker,
                "ocr_per": per,
                "ocr_pbr": pbr,
                "ocr_roe": roe,
                "ocr_roa": roa,
                "ocr_yield": yld,
                "ocr_per_confidence": assess_metric_confidence(per_text, "予想PER", per, 1.0, 120.0),
                "ocr_pbr_confidence": assess_metric_confidence(pbr_text, "PBR", pbr, 0.1, 20.0),
                "ocr_roe_confidence": assess_metric_confidence(roe_text, "ROE", roe, 0.1, 50.0),
                "ocr_roa_confidence": assess_metric_confidence(roa_text, "ROA", roa, 0.1, 20.0),
                "ocr_yield_confidence": assess_metric_confidence(yld_text, "利回り", yld, 0.1, 20.0),
                "ocr_metrics_text": text,
                "image_file": row["image_file"],
            }
        )
    metric_df = pd.DataFrame(metric_rows)
    out = base_df.merge(metric_df, how="left", on="ticker")
    out["sector_adjusted_per_score"] = compute_sector_adjusted_per_score(out)
    out["finance_quality_score"] = (
        0.35 * percentile_score(out["ocr_roe"], ascending=True)
        + 0.20 * percentile_score(out["ocr_roa"], ascending=True)
        + 0.20 * percentile_score(out["sector_adjusted_per_score"], ascending=True)
        + 0.15 * percentile_score(out["ocr_yield"], ascending=True)
        + 0.10 * percentile_score(out["ocr_pbr"], ascending=False)
    )
    out["growth_quality_score"] = (
        0.40 * percentile_score(out["annual_return_pct"], ascending=True)
        + 0.30 * percentile_score(out["quarter_return_pct"], ascending=True)
        + 0.15 * percentile_score(out["trend_r2"], ascending=True)
        + 0.15 * percentile_score(out["persistence_20d_pct"], ascending=True)
    )
    out["overall_score"] = (
        0.35 * out["score"]
        + 0.30 * out["finance_quality_score"]
        + 0.20 * out["growth_quality_score"]
        + 0.15 * percentile_score(out["sector_adjusted_per_score"], ascending=True)
    )
    return out.sort_values(["overall_score", "score", "finance_quality_score"], ascending=[False, False, False]).reset_index(drop=True)


def select_candidates(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty:
        return detail_df
    selected = detail_df.copy()
    selected = selected[selected["sector_adjusted_per_score"].fillna(0) >= 0.45]
    selected = selected[selected["ocr_per"].fillna(999) <= 20]
    selected = selected[selected["persistence_20d_pct"].fillna(0) >= 65]
    selected = selected[selected["positive_month_ratio_pct"].fillna(0) >= 60]
    selected = selected[selected["max_drawdown_pct"].fillna(999) <= 28]
    selected = selected[selected["annual_return_pct"].fillna(-999).between(30, 180)]
    selected = selected[selected["quarter_return_pct"].fillna(-999).between(18, 55)]
    selected = selected[selected["trend_r2"].fillna(0) >= 0.75]
    selected = selected.sort_values(
        ["overall_score", "sector_adjusted_per_score", "ocr_roe", "annual_return_pct"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    selected["selected_rank"] = np.arange(1, len(selected) + 1)
    selected["selected"] = True
    return selected


def attach_realized_returns(selected: pd.DataFrame, price_dir: Path) -> pd.DataFrame:
    if selected.empty:
        return selected
    rows = []
    start = pd.Timestamp("2024-07-01")
    end = pd.Timestamp("2024-09-30")
    for _, row in selected.iterrows():
        ticker = str(row["ticker"])
        price_path = price_dir / f"{ticker.replace('.', '_')}.csv"
        future_ret = np.nan
        if price_path.exists():
            df = pd.read_csv(price_path)
            if "Date" in df.columns and "Close" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.sort_values("Date").set_index("Date")
                window = df[(df.index >= start) & (df.index <= end)]
                if len(window) >= 2:
                    future_ret = float(window["Close"].iloc[-1] / window["Close"].iloc[0] - 1.0) * 100.0
        item = row.to_dict()
        item["future_return_2024Q3_pct"] = future_ret
        rows.append(item)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Select 2024 2Q candidates using only data up to 2024-06-30.")
    parser.add_argument("--image-dir", type=Path, default=Path(r"c:\Users\mitsu\OneDrive\ドキュメント\四季報DB2025\2024-2Q"))
    parser.add_argument("--name-csv", type=Path, default=Path(r"c:\Users\mitsu\Downloads\四季報2026年1集 - 銘柄.csv"))
    parser.add_argument("--price-dir", type=Path, default=Path("data/prices"))
    parser.add_argument("--out-dir", type=Path, default=Path("projects/quarterly_ranker/output/q2_2024_pre_analysis_20240630_full"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    name_master = load_name_master(args.name_csv)
    library = build_2024_q2_library(args.image_dir, name_master, args.out_dir / "library_cache")
    library_csv = args.out_dir / "q2_2024_image_library.csv"
    library.to_csv(library_csv, index=False, encoding="utf-8-sig")

    prepare_mapped_image_dir(library, args.image_dir, args.out_dir / "mapped_images")

    base_df = build_pre_jul_2024_base(library, args.price_dir)
    base_csv = args.out_dir / "q2_2024_pre_base_candidates.csv"
    base_df.to_csv(base_csv, index=False, encoding="utf-8-sig")

    detail_df = enrich_with_numeric_ocr(base_df, library)
    detail_csv = args.out_dir / "q2_2024_pre_shikiho_feature_ranking.csv"
    detail_df.to_csv(detail_csv, index=False, encoding="utf-8-sig")

    selected = select_candidates(detail_df)
    selected = attach_realized_returns(selected, args.price_dir)
    selected["learning_date"] = "2024-06-30"
    selected["training_cutoff_date"] = "2024-06-30"
    selected_csv = args.out_dir / "q2_2024_pre_selected_candidates.csv"
    selected.to_csv(selected_csv, index=False, encoding="utf-8-sig")
    operational_selected_csv = args.out_dir / "operational" / "q2_2024_pre_selected_candidates_operational.csv"
    write_operational_csv(selected, operational_selected_csv)

    manifest = pd.DataFrame(
        [
            {
                "dataset_id": "q2_2024_pre_analysis_20240630_full",
                "learning_date": "2024-06-30",
                "training_cutoff_date": "2024-06-30",
                "image_count_total": int(len(library)),
                "selected_count": int(len(selected)),
                "output_dir": str(args.out_dir),
                "source_image_dir": str(args.image_dir),
            }
        ]
    )
    manifest_csv = args.out_dir / "q2_2024_learning_dataset_manifest.csv"
    manifest.to_csv(manifest_csv, index=False, encoding="utf-8-sig")

    print(f"[OUT] library={library_csv}")
    print(f"[OUT] base={base_csv}")
    print(f"[OUT] detail={detail_csv}")
    print(f"[OUT] selected={selected_csv}")
    print(f"[OUT] selected_operational={operational_selected_csv}")
    print(f"[OUT] manifest={manifest_csv}")
    print(f"[INFO] images={len(library)} selected={len(selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

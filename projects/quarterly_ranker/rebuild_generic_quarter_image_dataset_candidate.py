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
from projects.quarterly_ranker.operational_csv_utils import write_operational_csv
from projects.quarterly_ranker.rank_quarterly_promising_stocks import (
    calc_max_drawdown,
    calc_r2_log_trend,
    percentile_score,
)
from projects.quarterly_ranker.select_2024_q2_pre_jul_candidates import (
    attach_realized_returns as attach_realized_returns_q2,
    enrich_with_numeric_ocr as enrich_with_numeric_ocr_q2,
    load_name_master,
    select_candidates as select_candidates_q2,
)
from projects.quarterly_ranker.select_q3_pre_jul_candidates import (
    attach_realized_returns as attach_realized_returns_q3,
    enrich_with_numeric_ocr as enrich_with_numeric_ocr_q3,
    select_candidates as select_candidates_q3,
)
from projects.quarterly_ranker.select_4q_pre_oct_candidates import (
    attach_realized_returns as attach_realized_returns_q4,
    enrich_with_numeric_ocr as enrich_with_numeric_ocr_q4,
    select_candidates as select_candidates_q4,
)


def normalize_symbol(raw: str) -> str:
    symbol = str(raw).strip().upper()
    if not symbol:
        return symbol
    if symbol.startswith("TYO:"):
        return f"{symbol.split(':', 1)[1]}.T"
    if symbol.endswith(".T"):
        return symbol
    return f"{symbol}.T"


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


def collect_images(image_dir: Path) -> list[Path]:
    return sorted(image_dir.rglob("*.png"), key=lambda p: int(re.sub(r"\D", "", p.stem) or "0"))


def build_generic_library(image_dir: Path, name_master: pd.DataFrame, cache_dir: Path, images: list[Path] | None = None) -> pd.DataFrame:
    valid_codes = set(name_master["code"].astype(str))
    rows = []
    images = images if images is not None else collect_images(image_dir)
    for image_path in images:
        rel = image_path.relative_to(image_dir)
        crop_name = "__".join(rel.parts)
        crop_path = cache_dir / f"{crop_name}_metrics.png"
        if not crop_path.exists():
            save_metrics_crop(image_path, crop_path)
        text = ocr_image_rapid(crop_path)
        code = resolve_code_from_metrics_text(text, valid_codes)
        image_no_match = re.search(r"(\d+)", image_path.stem)
        image_no = int(image_no_match.group(1)) if image_no_match else -1
        rows.append(
            {
                "image_file": rel.as_posix(),
                "image_no": image_no,
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
        src = source_dir / Path(str(row["image_file"]))
        dst = mapped_dir / f"{str(row['code'])}.png"
        if not src.exists():
            continue
        try:
            os.link(src, dst)
        except Exception:
            shutil.copy2(src, dst)
    return mapped_dir


def calc_metrics_generic(symbol: str, price_path: Path, cutoff_date: pd.Timestamp, quarter_start: pd.Timestamp) -> dict[str, float] | None:
    df = pd.read_csv(price_path)
    if "Date" not in df.columns or "Close" not in df.columns:
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")

    hist = df[df.index <= cutoff_date].copy()
    if hist.empty:
        return None
    qdf = hist[(hist.index >= quarter_start) & (hist.index <= cutoff_date)].copy()
    trailing = hist.tail(min(252, len(hist))).copy()
    annual_window_start = cutoff_date - pd.Timedelta(days=365)
    annual_df = hist[(hist.index >= annual_window_start) & (hist.index <= cutoff_date)].copy()
    if len(annual_df) < 100:
        annual_df = trailing.copy()
    if len(qdf) < 20 or len(trailing) < 120 or len(annual_df) < 100:
        return None

    monthly = qdf.resample("ME").agg({"High": "max", "Low": "min"}).dropna()
    if monthly.empty:
        return None

    annual_return = float(annual_df["Close"].iloc[-1] / annual_df["Close"].iloc[0] - 1.0)
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


def build_base_generic(
    library: pd.DataFrame,
    price_dir: Path,
    cutoff_date: pd.Timestamp,
    quarter_start: pd.Timestamp,
    quarter_label: str,
) -> pd.DataFrame:
    output_columns = [
        "quarter", "rank", "ticker", "company_name", "sector", "score",
        "annual_return_pct", "quarter_return_pct", "avg_monthly_high_low_change_pct",
        "max_monthly_high_low_change_pct", "trend_r2", "max_drawdown_pct",
        "positive_month_ratio_pct", "end_to_trailing_high_pct", "persistence_20d_pct",
        "quarter_end_close", "price_rows", "trailing_return_pct",
    ]
    rows = []
    for _, row in library[library["resolved"] == True].iterrows():
        ticker = str(row["ticker"])
        price_path = price_dir / f"{ticker.replace('.', '_')}.csv"
        if not price_path.exists():
            continue
        metrics = calc_metrics_generic(ticker, price_path, cutoff_date, quarter_start)
        if metrics is None:
            continue
        metrics["company_name"] = row["company_name"]
        metrics["sector"] = row["sector"]
        rows.append(metrics)
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=output_columns)
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
    df["quarter"] = quarter_label
    df["rank"] = np.arange(1, len(df) + 1)
    return df[output_columns]


def select_funcs(style: str):
    if style == "q2":
        return enrich_with_numeric_ocr_q2, select_candidates_q2, attach_realized_returns_q2
    if style == "q3":
        return enrich_with_numeric_ocr_q3, select_candidates_q3, attach_realized_returns_q3
    if style == "q4":
        return enrich_with_numeric_ocr_q4, select_candidates_q4, attach_realized_returns_q4
    raise ValueError(f"Unsupported style: {style}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generic image-based quarterly dataset rebuild candidate.")
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--name-csv", type=Path, default=Path(r"c:\Users\mitsu\Downloads\四季報2026年1集 - 銘柄.csv"))
    parser.add_argument("--price-dir", type=Path, default=Path("data/prices"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cutoff-date", type=str, required=True)
    parser.add_argument("--quarter-start", type=str, required=True)
    parser.add_argument("--quarter-label", type=str, required=True)
    parser.add_argument("--selector-style", choices=["q2", "q3", "q4"], required=True)
    parser.add_argument("--realized-start", type=str, required=True)
    parser.add_argument("--realized-end", type=str, required=True)
    parser.add_argument("--library-name", type=str, default="image_library.csv")
    parser.add_argument("--image-start", type=int, default=0)
    parser.add_argument("--image-end", type=int, default=None)
    parser.add_argument("--library-only", action="store_true")
    parser.add_argument("--library-csv", type=Path, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cutoff_date = pd.Timestamp(args.cutoff_date)
    quarter_start = pd.Timestamp(args.quarter_start)
    learning_date = cutoff_date.strftime("%Y-%m-%d")

    name_master = load_name_master(args.name_csv)
    if args.library_csv is not None and args.library_csv.exists():
        library = pd.read_csv(args.library_csv)
        if "resolved" in library.columns:
            library["resolved"] = library["resolved"].astype(bool)
    else:
        images = collect_images(args.image_dir)
        sliced_images = images[args.image_start:args.image_end]
        library = build_generic_library(args.image_dir, name_master, args.out_dir / "library_cache", sliced_images)
    library_csv = args.out_dir / args.library_name
    library.to_csv(library_csv, index=False, encoding="utf-8-sig")
    if args.library_only:
        print(f"[OUT] library={library_csv}")
        print(f"[INFO] images={len(library)} resolved={int(library['resolved'].sum())}")
        return 0

    base_df = build_base_generic(library, args.price_dir, cutoff_date, quarter_start, args.quarter_label)
    base_df["learning_date"] = learning_date
    base_df["training_cutoff_date"] = learning_date
    base_csv = args.out_dir / "pre_base_candidates.csv"
    base_df.to_csv(base_csv, index=False, encoding="utf-8-sig")

    prepare_mapped_image_dir(library, args.image_dir, args.out_dir / "mapped_images")
    enrich_func, select_func, realized_func = select_funcs(args.selector_style)
    detail_df = enrich_func(base_df, library)
    detail_df["learning_date"] = learning_date
    detail_df["training_cutoff_date"] = learning_date
    detail_csv = args.out_dir / "pre_shikiho_feature_ranking.csv"
    detail_df.to_csv(detail_csv, index=False, encoding="utf-8-sig")

    selected = select_func(detail_df)
    realized_start = pd.Timestamp(args.realized_start)
    realized_end = pd.Timestamp(args.realized_end)

    # Reuse existing attachers by monkey-patching their fixed realized window if necessary.
    # Simpler and explicit here: recompute realized return directly.
    rows = []
    for _, row in selected.iterrows():
        item = row.to_dict()
        ticker = str(row["ticker"])
        price_path = args.price_dir / f"{ticker.replace('.', '_')}.csv"
        realized = np.nan
        start_close = np.nan
        end_close = np.nan
        if price_path.exists():
            pdf = pd.read_csv(price_path)
            if "Date" in pdf.columns and "Close" in pdf.columns:
                pdf["Date"] = pd.to_datetime(pdf["Date"])
                pdf = pdf.sort_values("Date").set_index("Date")
                window = pdf[(pdf.index >= realized_start) & (pdf.index <= realized_end)]
                if len(window) >= 2:
                    start_close = float(window["Close"].iloc[0])
                    end_close = float(window["Close"].iloc[-1])
                    realized = (end_close / start_close - 1.0) * 100.0
        item["realized_start_date"] = realized_start.strftime("%Y-%m-%d")
        item["realized_end_date"] = realized_end.strftime("%Y-%m-%d")
        item["realized_start_close"] = start_close
        item["realized_end_close"] = end_close
        item["realized_return_pct"] = realized
        rows.append(item)
    selected = pd.DataFrame(rows)
    selected["learning_date"] = learning_date
    selected["training_cutoff_date"] = learning_date
    selected_csv = args.out_dir / "pre_selected_candidates.csv"
    selected.to_csv(selected_csv, index=False, encoding="utf-8-sig")
    operational_selected_csv = args.out_dir / "operational" / "pre_selected_candidates_operational.csv"
    write_operational_csv(selected, operational_selected_csv)

    manifest = pd.DataFrame(
        [
            {
                "dataset_id": args.out_dir.name,
                "learning_date": learning_date,
                "training_cutoff_date": learning_date,
                "image_count_total": int(len(library)),
                "selected_count": int(len(selected)),
                "output_dir": str(args.out_dir),
                "source_image_dir": str(args.image_dir),
                "source_library_csv": str(library_csv),
                "selector_style": args.selector_style,
                "realized_start": args.realized_start,
                "realized_end": args.realized_end,
            }
        ]
    )
    manifest.to_csv(args.out_dir / "learning_dataset_manifest.csv", index=False, encoding="utf-8-sig")

    print(f"[OUT] library={library_csv}")
    print(f"[OUT] base={base_csv}")
    print(f"[OUT] detail={detail_csv}")
    print(f"[OUT] selected={selected_csv}")
    print(f"[OUT] selected_operational={operational_selected_csv}")
    print(f"[INFO] images={len(library)} resolved={int(library['resolved'].sum())} selected={len(selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

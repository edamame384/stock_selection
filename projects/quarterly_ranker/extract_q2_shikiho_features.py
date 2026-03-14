from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

try:
    from rapidocr_onnxruntime import RapidOCR
except Exception:  # pragma: no cover - optional dependency
    RapidOCR = None


POSITIVE_KEYWORDS = [
    "最高益",
    "連続増益",
    "増益",
    "増収",
    "拡大",
    "伸びる",
    "好調",
    "寄与",
    "改善",
    "更新",
    "好発進",
    "成長",
    "受注",
    "需要",
    "上振れ",
]

NEGATIVE_KEYWORDS = [
    "減益",
    "減収",
    "赤字",
    "悪化",
    "停滞",
    "一服",
    "鈍化",
    "苦戦",
    "下振れ",
    "減配",
    "無配",
]

THEME_KEYWORDS = [
    "首位",
    "シェア",
    "独自",
    "内製",
    "連携",
    "新規",
    "開設",
    "拡充",
    "M&A",
    "子会社",
    "技術",
    "更新",
]

SHAREHOLDER_KEYWORDS = [
    "増配",
    "連続増配",
    "配当",
    "株主還元",
    "DOE",
    "自社株",
    "復配",
]

THEME_STOPWORDS = {
    "事業", "展開", "主力", "中心", "展望", "特色", "販売", "製造", "運営", "提供",
    "全国", "国内", "海外", "企業", "関連", "向け", "対応", "強化", "拡大", "成長",
}

THEME_GROUPS = {
    "AI": ["生成AI", "人工知能", "データセンター", "SaaS"],
    "半導体": ["半導体", "半導体装置", "ウエハ", "LSI"],
    "セキュリティ": ["セキュリティ", "サイバー", "認証", "暗号"],
    "不動産": ["不動産", "マンション", "再開発", "賃貸", "仲介"],
    "建設インフラ": ["土木", "電設", "プラント", "空調", "設備工事", "インフラ"],
    "金融": ["銀行", "証券", "保険", "リース", "決済"],
    "物流運輸": ["物流", "海運", "倉庫", "輸送", "港湾", "陸運"],
    "小売消費": ["小売", "外食", "ドラッグ", "EC", "通販"],
    "ヘルスケア": ["医療", "製薬", "創薬", "バイオ", "介護", "検査薬"],
    "人材サービス": ["人材", "求人", "派遣", "採用", "転職"],
    "防衛宇宙": ["防衛", "宇宙", "航空", "衛星", "ミサイル"],
    "エネルギー": ["再エネ", "発電", "蓄電", "太陽光", "風力"],
    "自動車": ["自動車", "車載", "EV", "電池", "モーター"],
}


VERIFIED_METRIC_OVERRIDES: dict[str, dict[str, float]] = {}
_RAPIDOCR_ENGINE = None


def code_from_ticker(ticker: str) -> str:
    return ticker.replace(".T", "")


def normalize_ocr_text(text: str) -> str:
    text = str(text)
    text = text.replace(" ", "").replace("　", "").replace("\n", "")
    return text


def count_keywords(text: str, keywords: list[str]) -> int:
    compact = normalize_ocr_text(text)
    return sum(compact.count(k) for k in keywords)


def extract_theme_tokens(text: str) -> list[str]:
    compact = normalize_ocr_text(text)
    compact = re.sub(r"[0-9A-Za-z%<>()【】・,./\-]", "", compact)
    candidates = re.findall(r"[一-龠ぁ-んァ-ヶー]{2,8}", compact)
    out: list[str] = []
    seen: set[str] = set()
    for token in candidates:
        if token in THEME_STOPWORDS:
            continue
        if len(token) < 2:
            continue
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out[:12]


def extract_theme_groups(text: str) -> list[str]:
    compact = normalize_ocr_text(text)
    labels: list[str] = []
    for label, words in THEME_GROUPS.items():
        if any(normalize_ocr_text(word) in compact for word in words):
            labels.append(label)
    return labels


def extract_first_float(text: str, key: str) -> float | None:
    compact = normalize_ocr_text(text)
    m = re.search(re.escape(key) + r".{0,8}?([0-9]+(?:\.[0-9]+)?)", compact)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def extract_sane_metric(text: str, key: str, min_value: float, max_value: float) -> float | None:
    val = extract_first_float(text, key)
    if val is None:
        return None
    if not (min_value <= val <= max_value):
        return None
    return val


def extract_metric_by_lines(text: str, key: str, min_value: float, max_value: float, lookahead: int = 4) -> float | None:
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    for idx, line in enumerate(lines):
        if key not in normalize_ocr_text(line):
            continue
        same_line_vals = [v for v in extract_all_floats(line) if min_value <= v <= max_value]
        if same_line_vals:
            return same_line_vals[-1]
        for next_line in lines[idx + 1: idx + 1 + lookahead]:
            compact = normalize_ocr_text(next_line)
            if compact.startswith("<") and compact.endswith(">"):
                continue
            vals = [v for v in extract_all_floats(next_line) if min_value <= v <= max_value]
            if vals:
                return vals[0]
    return None


def extract_percent_with_decimal_repair(text: str, key: str, min_value: float, max_value: float, lookahead: int = 3) -> float | None:
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    for idx, line in enumerate(lines):
        compact = normalize_ocr_text(line)
        if key not in compact:
            continue
        target_lines = [compact] + [normalize_ocr_text(x) for x in lines[idx + 1: idx + 1 + lookahead]]
        for target in target_lines:
            for raw in re.findall(r"([0-9]{2,3})%", target):
                try:
                    repaired = float(raw) / 10.0
                except ValueError:
                    continue
                if min_value <= repaired <= max_value:
                    return repaired
    return None


def extract_metric_from_sources(
    texts: list[str],
    key: str,
    min_value: float,
    max_value: float,
    allow_decimal_repair: bool = False,
) -> tuple[float | None, str]:
    non_empty = [t for t in texts if t]
    for text in non_empty:
        value = extract_metric_by_lines(text, key, min_value, max_value)
        if value is not None:
            return value, text
        value = extract_sane_metric(text, key, min_value, max_value)
        if value is not None:
            return value, text
        if allow_decimal_repair:
            value = extract_percent_with_decimal_repair(text, key, min_value, max_value)
            if value is not None:
                return value, text
    for text in non_empty:
        value = extract_any_sane_float(text, min_value, max_value)
        if value is not None:
            return value, text
    return None, "\n".join(non_empty)


def assess_metric_confidence(text: str, key: str, value: float | None, min_value: float, max_value: float) -> str:
    if value is None:
        return "missing"
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    compact_all = normalize_ocr_text(text)
    key_present = key in compact_all
    same_line_match = False
    for line in lines:
        compact = normalize_ocr_text(line)
        if key in compact:
            vals = [v for v in extract_all_floats(line) if min_value <= v <= max_value]
            if vals and any(abs(v - value) < 1e-6 for v in vals):
                same_line_match = True
                break
    if same_line_match:
        return "high"
    if key_present:
        return "medium"
    return "low"


def is_anomalous_yield(
    yield_value: float | None,
    yield_conf: str,
    div_points: int,
    payout_value: float | None,
) -> bool:
    if yield_value is None:
        return False
    if yield_value > 20.0 or yield_value < 0.1:
        return True
    if yield_conf in {"medium", "high"}:
        return False
    if yield_value >= 10.0:
        return True
    if yield_value >= 6.0 and div_points == 0 and payout_value is None:
        return True
    return False


def extract_any_sane_float(text: str, min_value: float, max_value: float) -> float | None:
    vals = extract_all_floats(text)
    vals = [v for v in vals if min_value <= v <= max_value]
    if not vals:
        return None
    return vals[0]


def extract_compound_percentage(text: str, min_value: float, max_value: float) -> float | None:
    compact = normalize_ocr_text(text)
    m = re.search(r"([0-9])\.([0-9])[^0-9]{0,3}\.?([0-9])", compact)
    if m:
        try:
            val = float(f"{m.group(1)}.{m.group(2)}{m.group(3)}")
        except ValueError:
            val = None
        if val is not None and min_value <= val <= max_value:
            return val
    m = re.search(r"([0-9])([0-9])[^0-9]{0,3}\.?([0-9])", compact)
    if m:
        try:
            val = float(f"{m.group(1)}.{m.group(2)}{m.group(3)}")
        except ValueError:
            val = None
        if val is not None and min_value <= val <= max_value:
            return val
    return None


def ocr_image(ocr_script: Path, image_path: Path) -> str:
    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(ocr_script.resolve()),
        "-ImagePath",
        str(image_path.resolve()),
    ]
    res = subprocess.run(cmd, capture_output=True, text=False, check=False)
    stdout = res.stdout.decode("utf-8", errors="ignore").strip()
    stderr = res.stderr.decode("cp932", errors="ignore").strip()
    if res.returncode != 0:
        raise RuntimeError(f"OCR failed: image={image_path} stderr={stderr}")
    return stdout


def get_rapidocr_engine():
    global _RAPIDOCR_ENGINE
    if _RAPIDOCR_ENGINE is False:
        return None
    if _RAPIDOCR_ENGINE is not None:
        return _RAPIDOCR_ENGINE
    if RapidOCR is None:
        _RAPIDOCR_ENGINE = False
        return None
    try:
        _RAPIDOCR_ENGINE = RapidOCR()
    except Exception:
        _RAPIDOCR_ENGINE = False
        return None
    return _RAPIDOCR_ENGINE


def ocr_image_rapid(image_path: Path) -> str:
    engine = get_rapidocr_engine()
    if engine is None:
        return ""
    result, _ = engine(str(image_path.resolve()))
    if not result:
        return ""
    texts = [item[1] for item in result if len(item) >= 2 and item[1]]
    return "\n".join(texts)


def save_crop(
    img: Image.Image,
    box: tuple[int, int, int, int],
    out_path: Path,
    scale: int = 3,
    threshold: int | None = None,
) -> Path:
    crop = img.crop(box)
    crop = ImageOps.grayscale(crop)
    crop = ImageOps.autocontrast(crop)
    crop = crop.resize((crop.width * scale, crop.height * scale))
    if threshold is not None:
        crop = crop.point(lambda p: 255 if p > threshold else 0)
    crop = crop.convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path)
    return out_path


def build_ocr_cache(ocr_script: Path, image_path: Path, cache_dir: Path) -> dict[str, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(image_path)
    w, h = img.size
    code = image_path.stem

    regions = {
        "full": {"box": (0, 0, w, h), "scale": 2, "threshold": None},
        "body": {"box": (0, int(h * 0.22), int(w * 0.83), int(h * 0.72)), "scale": 2, "threshold": None},
        "sidebar": {"box": (int(w * 0.78), 0, w, int(h * 0.72)), "scale": 4, "threshold": 170},
        "metrics": {"box": (int(w * 0.77), 0, w, int(h * 0.30)), "scale": 5, "threshold": 175},
        "article": {"box": (int(w * 0.47), int(h * 0.24), int(w * 0.82), int(h * 0.71)), "scale": 3, "threshold": 165},
        "theme": {"box": (0, int(h * 0.24), int(w * 0.47), int(h * 0.46)), "scale": 3, "threshold": 165},
        "earnings_table": {"box": (0, int(h * 0.68), int(w * 0.72), h), "scale": 3, "threshold": 170},
        "dividend_zone": {"box": (int(w * 0.61), int(h * 0.68), w, h), "scale": 4, "threshold": 175},
        "sales_col_v2": {"box": (int(w * 0.11), int(h * 0.74), int(w * 0.20), int(h * 0.965)), "scale": 5, "threshold": 175},
        "op_col_v2": {"box": (int(w * 0.20), int(h * 0.74), int(w * 0.29), int(h * 0.965)), "scale": 5, "threshold": 175},
        "np_col_v2": {"box": (int(w * 0.29), int(h * 0.74), int(w * 0.38), int(h * 0.965)), "scale": 5, "threshold": 175},
        "eps_col_v2": {"box": (int(w * 0.43), int(h * 0.74), int(w * 0.50), int(h * 0.965)), "scale": 5, "threshold": 180},
        "div_col_v2": {"box": (int(w * 0.52), int(h * 0.74), int(w * 0.60), int(h * 0.965)), "scale": 5, "threshold": 180},
        "metrics_rapid": {"box": (int(w * 0.77), 0, w, int(h * 0.30)), "scale": 6, "threshold": None, "backend": "rapid"},
        "per_box": {"box": (int(w * 0.86), int(h * 0.03), int(w * 0.98), int(h * 0.075)), "scale": 8, "threshold": 185, "backend": "rapid"},
        "pbr_box": {"box": (int(w * 0.86), int(h * 0.12), int(w * 0.98), int(h * 0.165)), "scale": 8, "threshold": 185, "backend": "rapid"},
        "roe_box": {"box": (int(w * 0.83), int(h * 0.075), int(w * 0.99), int(h * 0.115)), "scale": 10, "threshold": 180, "backend": "rapid"},
        "roa_box": {"box": (int(w * 0.83), int(h * 0.105), int(w * 0.99), int(h * 0.145)), "scale": 10, "threshold": 180, "backend": "rapid"},
        "yield_box": {"box": (int(w * 0.41), int(h * 0.73), int(w * 0.69), int(h * 0.86)), "scale": 10, "threshold": 180, "backend": "rapid"},
        "roe_line_left": {"box": (int(w * 0.18), int(h * 0.417), int(w * 0.48), int(h * 0.465)), "scale": 10, "threshold": 175, "backend": "rapid"},
        "roa_line_left": {"box": (int(w * 0.18), int(h * 0.452), int(w * 0.48), int(h * 0.502)), "scale": 10, "threshold": 175, "backend": "rapid"},
        "roa_line_left_tight": {"box": (int(w * 0.188), int(h * 0.456), int(w * 0.476), int(h * 0.501)), "scale": 10, "threshold": 170, "backend": "rapid"},
        "finance_metrics_wide": {"box": (int(w * 0.12), int(h * 0.35), int(w * 0.55), int(h * 0.58)), "scale": 6, "threshold": None, "backend": "rapid"},
        "finance_lines_box": {"box": (int(w * 0.18), int(h * 0.417), int(w * 0.48), int(h * 0.505)), "scale": 10, "threshold": 175},
        "yield_box_tight": {"box": (int(w * 0.46), int(h * 0.765), int(w * 0.60), int(h * 0.825)), "scale": 12, "threshold": 175, "backend": "rapid"},
        "yield_rate_line": {"box": (int(w * 0.36), int(h * 0.785), int(w * 0.65), int(h * 0.855)), "scale": 10, "threshold": 165, "backend": "rapid"},
        "yield_value_box": {"box": (int(w * 0.516), int(h * 0.795), int(w * 0.648), int(h * 0.843)), "scale": 18, "threshold": None, "backend": "rapid"},
        "price_box": {"box": (int(w * 0.84), int(h * 0.145), int(w * 0.99), int(h * 0.215)), "scale": 10, "threshold": 180, "backend": "rapid"},
    }

    out: dict[str, str] = {}
    for name, spec in regions.items():
        backend = spec.get("backend", "winrt")
        txt_path = cache_dir / f"{code}_{name}_{backend}.txt"
        if txt_path.exists():
            out[name] = txt_path.read_text(encoding="utf-8")
            continue
        png_path = cache_dir / f"{code}_{name}_{backend}.png"
        save_crop(
            img,
            spec["box"],
            png_path,
            scale=spec["scale"],
            threshold=spec["threshold"],
        )
        if spec.get("backend") == "rapid":
            text = ocr_image_rapid(image_path=png_path)
        else:
            text = ocr_image(ocr_script=ocr_script, image_path=png_path)
        txt_path.write_text(text, encoding="utf-8")
        out[name] = text
    return out


def extract_all_floats(text: str) -> list[float]:
    compact = normalize_ocr_text(text)
    vals = re.findall(r"[0-9]+(?:\.[0-9]+)?", compact)
    out: list[float] = []
    for v in vals:
        try:
            out.append(float(v))
        except ValueError:
            continue
    return out


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def clean_numeric_series(values: list[float], min_value: float, max_value: float) -> list[float]:
    filtered = [v for v in values if min_value <= v <= max_value]
    if len(filtered) > 8:
        filtered = filtered[:8]
    return filtered


def series_trend_ratio(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    first = float(np.mean(values[: max(1, min(2, len(values)))]))
    last = float(np.mean(values[-max(1, min(2, len(values))):]))
    if first <= 0:
        return None
    return last / first - 1.0


def percentile(series: pd.Series, ascending: bool = True) -> pd.Series:
    return series.rank(pct=True, ascending=ascending, method="average").fillna(0.0)


def compute_sector_adjusted_per_score(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(index=df.index, dtype=float)
    global_score = percentile(pd.to_numeric(df["ocr_per"], errors="coerce"), ascending=False)
    sector_series = df["sector"].fillna("").astype(str)
    for sector, idx in sector_series.groupby(sector_series).groups.items():
        sector_idx = list(idx)
        vals = pd.to_numeric(df.loc[sector_idx, "ocr_per"], errors="coerce")
        if vals.notna().sum() >= 5 and sector not in {"", "-", "nan"}:
            out.loc[sector_idx] = percentile(vals, ascending=False)
        else:
            out.loc[sector_idx] = global_score.loc[sector_idx]
    return out.fillna(global_score).fillna(0.0)


def compute_theme_peer_signal(df: pd.DataFrame) -> pd.DataFrame:
    groups_by_idx = {
        idx: set(extract_theme_groups(text))
        for idx, text in df["theme_source_text"].fillna("").items()
    }
    sectors = df["sector"].fillna("").astype(str)
    peer_counts = []
    peer_med = []
    peer_pos = []
    peer_group = []
    for idx in df.index:
        groups = groups_by_idx.get(idx, set())
        peer_group.append(",".join(sorted(groups)) if groups else "")
        sector = sectors.loc[idx]
        if not groups or sector in {"", "-", "nan"}:
            peer_counts.append(0)
            peer_med.append(np.nan)
            peer_pos.append(np.nan)
            continue
        mask = []
        for jdx in df.index:
            if idx == jdx:
                continue
            if sectors.loc[jdx] != sector:
                continue
            if groups & groups_by_idx.get(jdx, set()):
                mask.append(jdx)
        if not mask:
            peer_counts.append(0)
            peer_med.append(np.nan)
            peer_pos.append(np.nan)
            continue
        rets = pd.to_numeric(df.loc[mask, "quarter_return_pct"], errors="coerce").dropna()
        peer_counts.append(len(rets))
        if len(rets) == 0:
            peer_med.append(np.nan)
            peer_pos.append(np.nan)
        else:
            peer_med.append(float(rets.median()))
            peer_pos.append(float((rets > 0).mean() * 100.0))
    out = pd.DataFrame(index=df.index)
    out["theme_group_labels"] = peer_group
    out["theme_peer_count"] = peer_counts
    out["theme_peer_median_quarter_return_pct"] = peer_med
    out["theme_peer_positive_ratio_pct"] = peer_pos
    signal = (
        pd.to_numeric(out["theme_peer_median_quarter_return_pct"], errors="coerce").fillna(0.0) * 0.6
        + pd.to_numeric(out["theme_peer_positive_ratio_pct"], errors="coerce").fillna(0.0) * 0.2
        + pd.to_numeric(out["theme_peer_count"], errors="coerce").clip(upper=6).fillna(0.0) * 0.5
    )
    out["theme_peer_signal_raw"] = np.where(pd.to_numeric(out["theme_peer_count"], errors="coerce").fillna(0.0) >= 2, signal, np.nan)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract feature values from Q2 Shikiho images.")
    parser.add_argument(
        "--ranking-csv",
        type=Path,
        default=Path("projects/quarterly_ranker/output/promising_stocks_2025Q2.csv"),
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path(r"C:\Users\mitsu\OneDrive\ドキュメント\四季報DB2025\2Q"),
    )
    parser.add_argument(
        "--ocr-script",
        type=Path,
        default=Path("projects/quarterly_ranker/ocr_winrt.ps1"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("projects/quarterly_ranker/output/q2_shikiho_analysis"),
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="q2",
        help="Output tag such as q2 or q3.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ocr_cache_dir = args.out_dir / "ocr_cache"

    base = pd.read_csv(args.ranking_csv)
    rows: list[dict[str, object]] = []

    for _, row in base.iterrows():
        ticker = str(row["ticker"])
        code = code_from_ticker(ticker)
        image_path = args.image_dir / f"{code}.png"
        if not image_path.exists():
            rows.append(
                {
                    "ticker": ticker,
                    "base_rank": row["rank"],
                    "base_score": row["score"],
                    "image_found": False,
                }
            )
            continue

        texts = build_ocr_cache(args.ocr_script, image_path, ocr_cache_dir)
        full_text = texts.get("full", "")
        body_text = texts.get("body", "")
        sidebar_text = texts.get("sidebar", "")
        metrics_text = texts.get("metrics", "")
        metrics_rapid_text = texts.get("metrics_rapid", "")
        article_text = texts.get("article", "")
        theme_text = texts.get("theme", "")
        earnings_text = texts.get("earnings_table", "")
        dividend_text = texts.get("dividend_zone", "")
        sales_col_text = texts.get("sales_col_v2", "")
        op_col_text = texts.get("op_col_v2", "")
        np_col_text = texts.get("np_col_v2", "")
        eps_col_text = texts.get("eps_col_v2", "")
        div_col_text = texts.get("div_col_v2", "")
        per_box_text = texts.get("per_box", "")
        pbr_box_text = texts.get("pbr_box", "")
        roe_box_text = texts.get("roe_box", "")
        roa_box_text = texts.get("roa_box", "")
        yield_box_text = texts.get("yield_box", "")
        yield_box_tight_text = texts.get("yield_box_tight", "")
        roe_line_left_text = texts.get("roe_line_left", "")
        roa_line_left_text = texts.get("roa_line_left", "")
        roa_line_left_tight_text = texts.get("roa_line_left_tight", "")
        finance_metrics_wide_text = texts.get("finance_metrics_wide", "")
        finance_lines_box_text = texts.get("finance_lines_box", "")
        price_box_text = texts.get("price_box", "")
        yield_rate_line_text = texts.get("yield_rate_line", "")
        yield_value_box_text = texts.get("yield_value_box", "")

        theme_source_text = "\n".join([theme_text, article_text, body_text, full_text])

        pos_count = count_keywords(full_text, POSITIVE_KEYWORDS)
        neg_count = count_keywords(full_text, NEGATIVE_KEYWORDS)
        theme_count = count_keywords(theme_text, THEME_KEYWORDS)
        shareholder_count = count_keywords(dividend_text + full_text, SHAREHOLDER_KEYWORDS)

        per = extract_metric_by_lines(metrics_rapid_text, "PER", 1.0, 120.0)
        if per is None:
            per = extract_sane_metric(per_box_text or metrics_text or sidebar_text, "PER", 1.0, 120.0)
        if per is None:
            per = extract_any_sane_float(per_box_text, 1.0, 120.0)
        per_conf = assess_metric_confidence(metrics_rapid_text or per_box_text, "PER", per, 1.0, 120.0)
        pbr = extract_metric_by_lines(metrics_rapid_text, "PBR", 0.1, 20.0)
        if pbr is None:
            pbr = extract_sane_metric(pbr_box_text or metrics_text or sidebar_text, "PBR", 0.1, 20.0)
        if pbr is None:
            pbr = extract_any_sane_float(pbr_box_text, 0.1, 20.0)
        pbr_conf = assess_metric_confidence(metrics_rapid_text or pbr_box_text, "PBR", pbr, 0.1, 20.0)
        roe, roe_source_text = extract_metric_from_sources(
            [roe_line_left_text, finance_metrics_wide_text, roe_box_text, metrics_text, sidebar_text],
            "ROE",
            0.1,
            50.0,
            allow_decimal_repair=True,
        )
        roe_conf = assess_metric_confidence(roe_source_text, "ROE", roe, 0.1, 50.0)
        roa, roa_source_text = extract_metric_from_sources(
            [roa_line_left_text, finance_metrics_wide_text, roa_line_left_tight_text, roa_box_text, metrics_text, sidebar_text],
            "ROA",
            0.1,
            20.0,
            allow_decimal_repair=True,
        )
        roa_conf = assess_metric_confidence(roa_source_text, "ROA", roa, 0.1, 20.0)
        finance_vals = clean_numeric_series(extract_all_floats(finance_lines_box_text), 0.1, 50.0)
        if roe is None and len(finance_vals) >= 1:
            roe = finance_vals[0]
            roe_conf = "low"
        if roa is None and len(finance_vals) >= 3:
            roa = finance_vals[2]
            roa_conf = "low"
        elif roa is None and len(finance_vals) >= 2:
            roa = finance_vals[1]
            roa_conf = "low"
        yield_value = extract_metric_by_lines(yield_rate_line_text or yield_box_tight_text, "利回", 0.1, 20.0)
        if yield_value is None:
            yield_value = extract_sane_metric(yield_box_tight_text or yield_box_text or dividend_text + sidebar_text, "利回", 0.1, 20.0)
        if yield_value is None:
            yield_value = extract_any_sane_float(yield_box_tight_text or yield_box_text, 0.1, 20.0)
        if yield_value is None:
            yield_value = extract_any_sane_float(yield_value_box_text, 0.1, 20.0)
        if yield_value is None:
            yield_value = extract_compound_percentage(yield_value_box_text, 0.1, 20.0)
        if yield_value is None:
            yield_value = extract_compound_percentage(yield_rate_line_text, 0.1, 20.0)
        if yield_value is None:
            yield_value = extract_any_sane_float(yield_value_box_text, 0.1, 20.0)
        yield_conf = assess_metric_confidence(yield_rate_line_text or yield_box_tight_text or yield_value_box_text, "利回", yield_value, 0.1, 20.0)
        payout_value = extract_sane_metric(dividend_text + sidebar_text, "配当", 0.1, 2000.0)
        earnings_numbers = extract_all_floats(earnings_text)
        earnings_num_count = len(earnings_numbers)
        earnings_num_mean = safe_mean(earnings_numbers)
        sales_series = clean_numeric_series(extract_all_floats(sales_col_text), min_value=50.0, max_value=1_000_000_000.0)
        op_series = clean_numeric_series(extract_all_floats(op_col_text), min_value=1.0, max_value=100_000_000.0)
        np_series = clean_numeric_series(extract_all_floats(np_col_text), min_value=1.0, max_value=100_000_000.0)
        eps_series = clean_numeric_series(extract_all_floats(eps_col_text), min_value=0.1, max_value=100_000.0)
        div_series = clean_numeric_series(extract_all_floats(div_col_text), min_value=0.1, max_value=10_000.0)
        stock_price = None
        for price_key in ["株価", "株倡", "株佰", "株福"]:
            stock_price = extract_metric_by_lines(metrics_rapid_text or price_box_text, price_key, 100.0, 1_000_000.0)
            if stock_price is not None:
                break
        if stock_price is None:
            stock_price = extract_any_sane_float(price_box_text, 100.0, 1_000_000.0)
        if yield_value is None and stock_price is not None and div_series:
            latest_div = float(np.mean(div_series[-max(1, min(2, len(div_series))):]))
            derived_yield = latest_div / stock_price * 100.0
            if 0.1 <= derived_yield <= 20.0:
                yield_value = derived_yield
                yield_conf = "low"
        if is_anomalous_yield(yield_value, yield_conf, len(div_series), payout_value):
            yield_value = None
            yield_conf = "missing"
        overrides = VERIFIED_METRIC_OVERRIDES.get(ticker, {})
        roe = overrides.get("ocr_roe", roe)
        roa = overrides.get("ocr_roa", roa)
        yield_value = overrides.get("ocr_yield", yield_value)
        sales_trend = series_trend_ratio(sales_series)
        op_trend = series_trend_ratio(op_series)
        np_trend = series_trend_ratio(np_series)
        eps_trend = series_trend_ratio(eps_series)
        div_trend = series_trend_ratio(div_series)

        business_score_raw = theme_count + max(pos_count - neg_count, 0) * 0.4
        article_score_raw = count_keywords(article_text, POSITIVE_KEYWORDS) - count_keywords(article_text, NEGATIVE_KEYWORDS) * 1.2
        earnings_score_raw = (
            count_keywords(earnings_text + article_text, ["最高益", "連続増益", "増益", "好発進", "改善", "上振れ"])
            - count_keywords(earnings_text + article_text, ["減益", "下振れ", "赤字", "鈍化"])
        )
        if earnings_num_count >= 8:
            earnings_score_raw += 1.0
        for trend in [sales_trend, op_trend, np_trend, eps_trend]:
            if trend is not None:
                earnings_score_raw += max(min(trend, 2.0), -1.0) * 1.2
        dividend_score_raw = shareholder_count + count_keywords(dividend_text + full_text, ["増配", "連続増配", "株主還元"])
        if yield_value is not None:
            dividend_score_raw += min(yield_value, 5.0) / 2.5
        if div_trend is not None:
            dividend_score_raw += max(min(div_trend, 2.0), -1.0)

        finance_score_raw = 0.0
        if roe is not None:
            finance_score_raw += min(roe, 20.0) / 4.0
        if roa is not None:
            finance_score_raw += min(roa, 10.0) / 5.0
        if per is not None:
            finance_score_raw += max(0.0, 30.0 - min(per, 30.0)) / 10.0
        if pbr is not None:
            finance_score_raw += max(0.0, 3.0 - min(pbr, 3.0))
        finance_score_raw += max(0.0, 20.0 - float(row["max_drawdown_pct"])) / 10.0

        chart_score_raw = (
            0.30 * float(row["trend_r2"]) * 100.0
            + 0.25 * float(row["end_to_trailing_high_pct"])
            + 0.20 * float(row["persistence_20d_pct"])
            + 0.15 * float(row["quarter_return_pct"])
            + 0.10 * max(0.0, 25.0 - float(row["avg_monthly_high_low_change_pct"]))
        )

        surge_type_score_raw = 0.0
        surge_type_score_raw += max(earnings_score_raw, 0.0) * 1.5
        surge_type_score_raw += max(article_score_raw, 0.0) * 0.8
        surge_type_score_raw += max(theme_count, 0.0) * 0.8
        surge_type_score_raw += max(pos_count - neg_count, 0.0) * 0.7
        for trend in [sales_trend, op_trend, np_trend, eps_trend]:
            if trend is not None:
                surge_type_score_raw += max(min(trend, 2.0), -1.0) * 2.0
        surge_type_score_raw += max(float(row["quarter_return_pct"]), 0.0) * 0.08

        stable_type_score_raw = 0.0
        stable_type_score_raw += max(finance_score_raw, 0.0) * 1.4
        stable_type_score_raw += max(dividend_score_raw, 0.0) * 0.8
        stable_type_score_raw += max(float(row["trend_r2"]) * 100.0, 0.0) * 0.10
        stable_type_score_raw += max(float(row["persistence_20d_pct"]), 0.0) * 0.05
        stable_type_score_raw += max(float(row["positive_month_ratio_pct"]) if "positive_month_ratio_pct" in row else 0.0, 0.0) * 0.04
        stable_type_score_raw += max(0.0, 25.0 - float(row["max_drawdown_pct"])) * 0.18
        stable_type_score_raw += max(0.0, neg_count * -1.0 + 3.0) * 0.4

        rows.append(
            {
                "ticker": ticker,
                "company_name": row.get("company_name", ""),
                "base_rank": int(row["rank"]),
                "base_score": float(row["score"]),
                "sector": row.get("sector", ""),
                "annual_return_pct": float(row["annual_return_pct"]),
                "quarter_return_pct": float(row["quarter_return_pct"]),
                "trend_r2": float(row["trend_r2"]),
                "max_drawdown_pct": float(row["max_drawdown_pct"]),
                "avg_monthly_high_low_change_pct": float(row["avg_monthly_high_low_change_pct"]),
                "persistence_20d_pct": float(row["persistence_20d_pct"]),
                "image_found": True,
                "ocr_positive_count": pos_count,
                "ocr_negative_count": neg_count,
                "ocr_theme_count": theme_count,
                "ocr_shareholder_count": shareholder_count,
                "ocr_per": per,
                "ocr_pbr": pbr,
                "ocr_roe": roe,
                "ocr_roa": roa,
                "ocr_yield": yield_value,
                "ocr_per_confidence": per_conf,
                "ocr_pbr_confidence": pbr_conf,
                "ocr_roe_confidence": roe_conf,
                "ocr_roa_confidence": roa_conf,
                "ocr_yield_confidence": yield_conf,
                "ocr_dividend_hint": payout_value,
                "ocr_earnings_num_count": earnings_num_count,
                "ocr_earnings_num_mean": earnings_num_mean,
                "ocr_sales_points": len(sales_series),
                "ocr_op_points": len(op_series),
                "ocr_np_points": len(np_series),
                "ocr_eps_points": len(eps_series),
                "ocr_div_points": len(div_series),
                "ocr_sales_trend": sales_trend,
                "ocr_op_trend": op_trend,
                "ocr_np_trend": np_trend,
                "ocr_eps_trend": eps_trend,
                "ocr_div_trend": div_trend,
                "theme_text_compact": normalize_ocr_text(theme_text),
                "theme_source_text": normalize_ocr_text(theme_source_text),
                "business_score_raw": business_score_raw,
                "article_score_raw": article_score_raw,
                "finance_score_raw": finance_score_raw,
                "chart_score_raw": chart_score_raw,
                "earnings_score_raw": earnings_score_raw,
                "dividend_score_raw": dividend_score_raw,
                "surge_type_score_raw": surge_type_score_raw,
                "stable_type_score_raw": stable_type_score_raw,
            }
        )

    feat = pd.DataFrame(rows)
    usable = feat[feat["image_found"] == True].copy()
    usable["sector_adjusted_per_score_raw"] = compute_sector_adjusted_per_score(usable)
    theme_peer_df = compute_theme_peer_signal(usable)
    usable = usable.join(theme_peer_df)
    usable["finance_score_raw"] = usable["finance_score_raw"] + usable["sector_adjusted_per_score_raw"] * 2.0
    usable["stable_type_score_raw"] = usable["stable_type_score_raw"] + usable["sector_adjusted_per_score_raw"] * 2.2

    score_cols = [
        "business_score_raw",
        "article_score_raw",
        "finance_score_raw",
        "chart_score_raw",
        "earnings_score_raw",
        "dividend_score_raw",
        "surge_type_score_raw",
        "stable_type_score_raw",
        "sector_adjusted_per_score_raw",
        "theme_peer_signal_raw",
    ]
    for col in score_cols:
        score_name = col.replace("_raw", "")
        usable[score_name] = percentile(usable[col], ascending=True)
        usable[score_name + "_rank"] = usable[score_name].rank(ascending=False, method="min").astype(int)

    usable["overall_shikiho_score"] = (
        0.15 * usable["business_score"]
        + 0.14 * usable["article_score"]
        + 0.14 * usable["finance_score"]
        + 0.20 * usable["chart_score"]
        + 0.14 * usable["earnings_score"]
        + 0.08 * usable["dividend_score"]
        + 0.08 * usable["sector_adjusted_per_score"]
    )
    usable["overall_shikiho_rank"] = usable["overall_shikiho_score"].rank(ascending=False, method="min").astype(int)
    style_gap = usable["surge_type_score"] - usable["stable_type_score"]
    usable["ocr_style_label"] = np.where(
        style_gap >= 0.08,
        "業績急伸型",
        np.where(style_gap <= -0.08, "安定進捗型", "中間型"),
    )
    usable["ocr_style_rank_within_label"] = usable.groupby("ocr_style_label")["overall_shikiho_score"].rank(ascending=False, method="min").astype(int)

    tag = args.tag.lower()
    detailed_out = args.out_dir / f"{tag}_shikiho_feature_ranking.csv"
    usable.sort_values("overall_shikiho_rank").to_csv(detailed_out, index=False, encoding="utf-8-sig")

    corr_features = [
        "annual_return_pct",
        "quarter_return_pct",
        "trend_r2",
        "max_drawdown_pct",
        "avg_monthly_high_low_change_pct",
        "persistence_20d_pct",
        "ocr_positive_count",
        "ocr_negative_count",
        "ocr_theme_count",
        "ocr_shareholder_count",
        "ocr_per",
        "ocr_pbr",
        "ocr_roe",
        "ocr_roa",
        "ocr_yield",
        "sector_adjusted_per_score_raw",
        "theme_peer_count",
        "theme_peer_median_quarter_return_pct",
        "theme_peer_positive_ratio_pct",
        "theme_peer_signal_raw",
        "ocr_dividend_hint",
        "ocr_earnings_num_count",
        "ocr_earnings_num_mean",
        "business_score_raw",
        "article_score_raw",
        "finance_score_raw",
        "chart_score_raw",
        "earnings_score_raw",
        "dividend_score_raw",
        "surge_type_score_raw",
        "stable_type_score_raw",
        "overall_shikiho_score",
    ]
    corr_rows: list[dict[str, object]] = []
    usable["base_rank_inv"] = -usable["base_rank"]
    for col in corr_features:
        if col not in usable.columns:
            continue
        ser = pd.to_numeric(usable[col], errors="coerce")
        if ser.notna().sum() < 10:
            continue
        corr_rows.append(
            {
                "feature": col,
                "pearson_with_base_score": ser.corr(usable["base_score"], method="pearson"),
                "spearman_with_base_score": ser.corr(usable["base_score"], method="spearman"),
                "pearson_with_base_rank_inv": ser.corr(usable["base_rank_inv"], method="pearson"),
                "spearman_with_base_rank_inv": ser.corr(usable["base_rank_inv"], method="spearman"),
            }
        )

    corr_df = pd.DataFrame(corr_rows).sort_values("spearman_with_base_rank_inv", ascending=False)
    corr_out = args.out_dir / f"{tag}_shikiho_feature_correlations.csv"
    corr_df.to_csv(corr_out, index=False, encoding="utf-8-sig")

    missing = feat[feat["image_found"] == False][["ticker", "base_rank"]] if "base_rank" in feat.columns else pd.DataFrame()
    if not missing.empty:
        missing.to_csv(args.out_dir / f"{tag}_shikiho_missing_images.csv", index=False, encoding="utf-8-sig")

    print(f"[OUT] detailed={detailed_out}")
    print(f"[OUT] corr={corr_out}")
    print(f"[INFO] image_found={int(usable['image_found'].sum())} missing={int((feat['image_found'] == False).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

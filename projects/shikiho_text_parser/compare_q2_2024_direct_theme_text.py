from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.shikiho_text_parser.compare_phase_adaptive_practical_v3 import load_price_map, normalize_static


DETAIL_CSV = ROOT / "projects" / "quarterly_ranker" / "output" / "q2_2024_pre_analysis_20240630_aligned" / "q2_2024_pre_shikiho_feature_ranking.csv"
IMAGE_LIBRARY_CSV = ROOT / "projects" / "quarterly_ranker" / "output" / "q2_2024_pre_analysis_20240630_aligned" / "q2_2024_image_library.csv"
OCR_CACHE_DIR = ROOT / "projects" / "quarterly_ranker" / "output" / "q2_shikiho_analysis" / "ocr_cache"
OUT_DIR = ROOT / "projects" / "shikiho_text_parser" / "output" / "q2_2024_direct_theme_text"
START = pd.Timestamp("2024-07-01")
END = pd.Timestamp("2024-09-30")


THEME_PATTERNS: dict[str, list[str]] = {
    "semiconductor": [
        r"半\s*導\s*[体休]",
        r"ウ\s*エ\s*ハ",
        r"半\s*導\s*[体休]\s*装\s*置",
        r"電\s*子\s*部\s*品",
    ],
    "defense": [
        r"防\s*衛",
        r"官\s*需",
        r"装\s*備\s*品",
        r"衛\s*星",
    ],
    "ai_datacenter": [
        r"A\s*I",
        r"Ａ\s*Ｉ",
        r"生\s*成\s*A\s*I",
        r"デ\s*ー\s*タ\s*セ\s*ン\s*タ",
        r"ク\s*ラ\s*ウ\s*ド",
        r"D\s*X",
        r"デ\s*ジ\s*タ\s*ル",
    ],
    "power_grid": [
        r"電\s*力",
        r"送\s*配\s*電",
        r"発\s*電",
        r"蓄\s*電",
        r"変\s*電",
        r"再\s*エ\s*ネ",
    ],
    "cyber_security": [
        r"サ\s*イ\s*バ",
        r"セ\s*キュ\s*リ",
        r"認\s*証",
        r"暗\s*号",
    ],
    "ship_heavy": [
        r"造\s*船",
        r"重\s*工",
        r"船\s*舶",
        r"艦",
    ],
    "infra_construction": [
        r"イ\s*ン\s*フ\s*ラ",
        r"再\s*開\s*発",
        r"土\s*木",
        r"建\s*設",
        r"プ\s*ラ\s*ン\s*ト",
        r"橋",
    ],
}


def load_ocr_theme_texts() -> pd.DataFrame:
    image_lib = pd.read_csv(IMAGE_LIBRARY_CSV, usecols=["code", "ticker", "company_name", "sector"])
    image_lib["code"] = image_lib["code"].astype(str).str.strip()
    image_lib["ticker"] = image_lib["ticker"].astype(str).str.strip()
    rows: list[dict[str, str]] = []
    for path in OCR_CACHE_DIR.glob("*_body_winrt.txt"):
        code = path.name.replace("_body_winrt.txt", "").strip()
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        rows.append({"code": code, "ocr_body_text": text})
    ocr_df = pd.DataFrame(rows)
    if ocr_df.empty:
        return pd.DataFrame(columns=["code", "ticker", "company_name", "sector", "ocr_body_text"])
    out = image_lib.merge(ocr_df, on="code", how="inner")
    return out.drop_duplicates(subset=["ticker"], keep="first")


def detect_themes(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    hit: list[str] = []
    for theme, patterns in THEME_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text):
                hit.append(theme)
                break
    return hit


def basket_return(price_map: dict[str, pd.DataFrame], tickers: list[str]) -> dict[str, float]:
    returns: list[float] = []
    rows = []
    for ticker in tickers:
        df = price_map.get(ticker)
        if df is None:
            continue
        window = df[(df.index >= START) & (df.index <= END)]
        if window.empty:
            continue
        start_px = float(window.iloc[0]["Open"] if "Open" in window.columns and not pd.isna(window.iloc[0]["Open"]) else window.iloc[0]["Close"])
        end_px = float(window.iloc[-1]["Close"])
        ret = end_px / start_px - 1.0
        returns.append(ret)
        rows.append({"ticker": ticker, "start_price": start_px, "end_price": end_px, "return_pct": ret * 100.0})
    if not returns:
        return {"count": 0, "avg_return_pct": 0.0, "median_return_pct": 0.0, "win_rate_pct": 0.0}
    s = pd.Series(returns)
    return {
        "count": int(len(returns)),
        "avg_return_pct": float(s.mean() * 100.0),
        "median_return_pct": float(s.median() * 100.0),
        "win_rate_pct": float((s > 0).mean() * 100.0),
        "details": rows,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detail_df = normalize_static(pd.read_csv(DETAIL_CSV)).drop_duplicates(subset=["ticker"], keep="first")
    ocr_df = load_ocr_theme_texts()
    if ocr_df.empty:
        raise SystemExit("No OCR body texts found.")

    merged = detail_df.merge(
        ocr_df[["ticker", "company_name", "sector", "ocr_body_text"]],
        on="ticker",
        how="inner",
        suffixes=("", "_ocr"),
    )
    merged["direct_themes"] = merged["ocr_body_text"].map(detect_themes)
    exploded = merged.explode("direct_themes").dropna(subset=["direct_themes"]).copy()
    exploded.to_csv(OUT_DIR / "direct_theme_matches.csv", index=False, encoding="utf-8-sig")

    price_map = load_price_map(merged["ticker"].astype(str).tolist(), END)

    rows = []
    for theme, g in exploded.groupby("direct_themes"):
        g = g.sort_values(["score", "trend_r2", "annual_return_pct"], ascending=[False, False, False]).drop_duplicates(subset=["ticker"])
        all_stats = basket_return(price_map, g["ticker"].astype(str).tolist())
        top20 = g.head(20)
        top20_stats = basket_return(price_map, top20["ticker"].astype(str).tolist())
        rows.append(
            {
                "theme": theme,
                "source_count": int(len(g)),
                "all_count": int(all_stats["count"]),
                "all_avg_return_pct": float(all_stats["avg_return_pct"]),
                "all_median_return_pct": float(all_stats["median_return_pct"]),
                "all_win_rate_pct": float(all_stats["win_rate_pct"]),
                "top20_count": int(top20_stats["count"]),
                "top20_avg_return_pct": float(top20_stats["avg_return_pct"]),
                "top20_median_return_pct": float(top20_stats["median_return_pct"]),
                "top20_win_rate_pct": float(top20_stats["win_rate_pct"]),
            }
        )
        top20[["ticker", "company_name", "sector", "score", "trend_r2", "annual_return_pct", "quarter_return_pct"]].to_csv(
            OUT_DIR / f"theme_{theme}_top20_constituents.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(top20_stats.get("details", [])).to_csv(
            OUT_DIR / f"theme_{theme}_top20_returns.csv",
            index=False,
            encoding="utf-8-sig",
        )

    summary_df = pd.DataFrame(rows).sort_values("top20_avg_return_pct", ascending=False)
    summary_df.to_csv(OUT_DIR / "direct_theme_summary.csv", index=False, encoding="utf-8-sig")
    payload = {
        "ocr_covered_tickers": int(merged["ticker"].nunique()),
        "theme_counts": summary_df[["theme", "source_count"]].to_dict(orient="records"),
        "top_themes": summary_df.to_dict(orient="records"),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

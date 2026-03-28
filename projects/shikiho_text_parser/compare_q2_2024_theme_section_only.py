from __future__ import annotations

import json
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
OUT_DIR = ROOT / "projects" / "shikiho_text_parser" / "output" / "q2_2024_theme_section_only"
START = pd.Timestamp("2024-07-01")
END = pd.Timestamp("2024-09-30")


THEME_GROUPS: dict[str, list[str]] = {
    "semiconductor": ["半導体", "半導休", "半導", "ウエハ", "半導体装置", "電子部品"],
    "defense_space": ["防衛", "官需", "衛星", "宇宙", "航空", "装備", "艦"],
    "ai_datacenter": ["AI", "ＡＩ", "生成AI", "データセンター", "クラウド", "SaaS", "DX", "デジタル"],
    "cyber_security": ["サイバー", "セキュリティ", "認証", "暗号"],
    "power_energy": ["電力", "送配電", "発電", "再エネ", "太陽光", "風力", "蓄電", "変電"],
    "construction_infra": ["建設", "土木", "空調", "プラント", "設備工事", "再開発", "インフラ", "電設"],
    "real_estate": ["不動産", "マンション", "賃貸", "仲介", "オフィス", "住宅"],
    "logistics_transport": ["物流", "倉庫", "輸送", "海運", "陸運", "港湾", "空運"],
    "finance": ["銀行", "保険", "証券", "リース", "決済", "カード"],
    "retail_consumer": ["小売", "外食", "通販", "EC", "ドラッグ", "スーパー", "百貨店"],
    "healthcare_bio": ["医療", "製薬", "創薬", "バイオ", "介護", "検査薬"],
    "human_services": ["人材", "求人", "派遣", "採用", "転職"],
    "auto_battery": ["自動車", "車載", "EV", "電池", "モーター"],
    "ship_heavy": ["造船", "重工", "船舶"],
}


def compact_text(text: str) -> str:
    return str(text).replace(" ", "").replace("　", "").replace("\n", "")


def load_theme_texts() -> pd.DataFrame:
    image_lib = pd.read_csv(IMAGE_LIBRARY_CSV, usecols=["code", "ticker", "company_name", "sector"])
    image_lib["code"] = image_lib["code"].astype(str).str.strip()
    rows: list[dict[str, str]] = []
    for path in OCR_CACHE_DIR.glob("*_theme_winrt.txt"):
        code = path.name.replace("_theme_winrt.txt", "").strip()
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        rows.append({"code": code, "theme_text_raw": text, "theme_text_compact_only": compact_text(text)})
    theme_df = pd.DataFrame(rows)
    if theme_df.empty:
        return pd.DataFrame(columns=["ticker", "company_name", "sector", "theme_text_raw", "theme_text_compact_only"])
    out = image_lib.merge(theme_df, on="code", how="inner")
    return out.drop_duplicates(subset=["ticker"], keep="first")


def detect_theme_groups(text: str) -> list[str]:
    compact = compact_text(text)
    labels: list[str] = []
    for theme, words in THEME_GROUPS.items():
        if any(compact_text(word) in compact for word in words):
            labels.append(theme)
    return labels


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
    theme_df = load_theme_texts()
    merged = detail_df.merge(
        theme_df[["ticker", "company_name", "sector", "theme_text_raw", "theme_text_compact_only"]],
        on="ticker",
        how="inner",
        suffixes=("", "_theme"),
    )
    merged["section_groups"] = merged["theme_text_raw"].map(detect_theme_groups)
    exploded = merged.explode("section_groups").dropna(subset=["section_groups"]).copy()
    exploded.to_csv(OUT_DIR / "theme_section_matches.csv", index=False, encoding="utf-8-sig")

    price_map = load_price_map(merged["ticker"].astype(str).tolist(), END)

    rows = []
    for theme, g in exploded.groupby("section_groups"):
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
        top20[["ticker", "company_name", "sector", "score", "trend_r2", "annual_return_pct", "quarter_return_pct", "theme_text_raw"]].to_csv(
            OUT_DIR / f"theme_{theme}_constituents.csv",
            index=False,
            encoding="utf-8-sig",
        )

    summary_df = pd.DataFrame(rows).sort_values("top20_avg_return_pct", ascending=False)
    summary_df.to_csv(OUT_DIR / "theme_section_summary.csv", index=False, encoding="utf-8-sig")
    payload = {
        "theme_text_covered_tickers": int(merged["ticker"].nunique()),
        "theme_counts": summary_df[["theme", "source_count"]].to_dict(orient="records"),
        "top_themes": summary_df.to_dict(orient="records"),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

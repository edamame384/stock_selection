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
REGENERATED_CSV = ROOT / "projects" / "shikiho_text_parser" / "output" / "q2_2024_theme_section_regenerated_v2" / "q2_2024_theme_section_regenerated_v2.csv"
OUT_DIR = ROOT / "projects" / "shikiho_text_parser" / "output" / "q2_2024_theme_section_regenerated_compare_v2"
START = pd.Timestamp("2024-07-01")
END = pd.Timestamp("2024-09-30")


THEME_GROUPS: dict[str, list[str]] = {
    "semiconductor": ["半導体", "半導", "ウエハ", "半導体装置", "電子部品", "露光", "検査装置"],
    "defense_space": ["防衛", "官需", "衛星", "宇宙", "航空", "装備", "艦", "レーダ", "ミサイル"],
    "ai_datacenter": ["AI", "ＡＩ", "生成AI", "データセンター", "クラウド", "SaaS", "DX", "GPU"],
    "cyber_security": ["サイバー", "セキュリティ", "認証", "暗号", "SOC", "脆弱性"],
    "power_energy": ["電力", "送配電", "発電", "再エネ", "太陽光", "風力", "蓄電", "変電"],
    "construction_infra": ["建設", "土木", "空調", "プラント", "設備工事", "再開発", "インフラ", "電設"],
    "real_estate": ["不動産", "マンション", "賃貸", "仲介", "オフィス", "住宅"],
    "logistics_transport": ["物流", "倉庫", "輸送", "海運", "陸運", "港湾", "空運"],
    "finance": ["銀行", "保険", "証券", "リース", "決済", "カード", "信託"],
    "retail_consumer": ["小売", "外食", "通販", "EC", "ドラッグ", "スーパー", "百貨店", "量販"],
    "healthcare_bio": ["医療", "製薬", "創薬", "バイオ", "介護", "検査薬"],
    "human_services": ["人材", "求人", "派遣", "採用", "転職"],
    "auto_battery": ["自動車", "車載", "EV", "電池", "モーター", "駆動"],
    "ship_heavy": ["造船", "重工", "船舶", "舶用"],
}

REPLACE_RULES = {
    "半導休": "半導体",
    "半導林": "半導体",
    "半導イ本": "半導体",
    "防衡": "防衛",
    "防街": "防衛",
    "宇亩": "宇宙",
    "衛生": "衛星",
    "データセンタ": "データセンター",
}


def compact_text(text: str) -> str:
    return str(text).replace(" ", "").replace("　", "").replace("\n", "")


def normalize_text(text: str) -> str:
    compact = compact_text(text)
    for src, dst in REPLACE_RULES.items():
        compact = compact.replace(src, dst)
    compact = re.sub(r"[|｜¦]+", "", compact)
    compact = re.sub(r"日本マスタートラスト信託銀行.{0,20}", "", compact)
    compact = re.sub(r"日本カストディ銀行.{0,20}", "", compact)
    compact = re.sub(r"ROE.{0,20}", "", compact)
    compact = re.sub(r"ROA.{0,20}", "", compact)
    return compact


def detect_theme_groups(text: str) -> list[str]:
    normalized = normalize_text(text)
    labels: list[str] = []
    for theme, words in THEME_GROUPS.items():
        if any(normalize_text(word) in normalized for word in words):
            labels.append(theme)
    return labels


def basket_return(price_map: dict[str, pd.DataFrame], tickers: list[str]) -> dict[str, float]:
    returns: list[float] = []
    for ticker in tickers:
        df = price_map.get(ticker)
        if df is None:
            continue
        window = df[(df.index >= START) & (df.index <= END)]
        if window.empty:
            continue
        start_px = float(window.iloc[0]["Open"] if "Open" in window.columns and not pd.isna(window.iloc[0]["Open"]) else window.iloc[0]["Close"])
        end_px = float(window.iloc[-1]["Close"])
        returns.append(end_px / start_px - 1.0)
    if not returns:
        return {"count": 0, "avg_return_pct": 0.0, "median_return_pct": 0.0, "win_rate_pct": 0.0}
    s = pd.Series(returns)
    return {
        "count": int(len(returns)),
        "avg_return_pct": float(s.mean() * 100.0),
        "median_return_pct": float(s.median() * 100.0),
        "win_rate_pct": float((s > 0).mean() * 100.0),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detail_df = normalize_static(pd.read_csv(DETAIL_CSV)).drop_duplicates(subset=["ticker"], keep="first")
    regen_df = pd.read_csv(REGENERATED_CSV).drop_duplicates(subset=["ticker"], keep="first")
    merged = detail_df.merge(
        regen_df[["ticker", "best_variant", "theme_text_regenerated", "theme_text_regenerated_normalized", "theme_hit_count"]],
        on="ticker",
        how="inner",
    )
    merged["section_groups"] = merged["theme_text_regenerated_normalized"].map(detect_theme_groups)
    exploded = merged.explode("section_groups").dropna(subset=["section_groups"]).copy()
    exploded.to_csv(OUT_DIR / "matches.csv", index=False, encoding="utf-8-sig")
    price_map = load_price_map(merged["ticker"].astype(str).tolist(), END)

    rows = []
    for theme, g in exploded.groupby("section_groups"):
        g = g.sort_values(["score", "theme_hit_count", "trend_r2", "annual_return_pct"], ascending=[False, False, False, False]).drop_duplicates(subset=["ticker"])
        stats = basket_return(price_map, g["ticker"].astype(str).tolist())
        rows.append({"theme": theme, "source_count": int(len(g)), **stats})
        g[["ticker", "company_name", "sector", "best_variant", "theme_text_regenerated_normalized"]].to_csv(
            OUT_DIR / f"theme_{theme}_constituents.csv",
            index=False,
            encoding="utf-8-sig",
        )

    summary_df = pd.DataFrame(rows).sort_values("avg_return_pct", ascending=False)
    summary_df.to_csv(OUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    payload = {
        "themes": summary_df.to_dict(orient="records"),
        "covered_tickers": int(merged["ticker"].nunique()),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

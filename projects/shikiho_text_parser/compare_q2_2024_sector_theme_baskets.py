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
OUT_DIR = ROOT / "projects" / "shikiho_text_parser" / "output" / "q2_2024_sector_theme_baskets"
START = pd.Timestamp("2024-07-01")
END = pd.Timestamp("2024-09-30")


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
    df = normalize_static(pd.read_csv(DETAIL_CSV)).drop_duplicates(subset=["ticker"], keep="first")
    price_map = load_price_map(df["ticker"].astype(str).tolist(), END)

    sector_rows = []
    for sector, group in df.groupby("sector"):
        if len(group) < 20:
            continue
        top20 = group.sort_values(["score", "trend_r2", "annual_return_pct"], ascending=[False, False, False]).head(20)
        stats = basket_return(price_map, top20["ticker"].astype(str).tolist())
        sector_rows.append(
            {
                "basket_type": "sector_top20",
                "label": sector,
                "source_count": int(len(group)),
                **{k: v for k, v in stats.items() if k != "details"},
            }
        )
        pd.DataFrame(stats.get("details", [])).to_csv(OUT_DIR / f"sector_{sector}_top20_details.csv", index=False, encoding="utf-8-sig")

    # Theme proxies using company names and sectors
    theme_defs = {
        "power_utilities": df["sector"].eq("電気・ガス業"),
        "shipbuilding": df["company_name"].astype(str).str.contains("造船|重工|IHI|三井E&S|川崎重工", regex=True, na=False),
        "cyber_security": df["company_name"].astype(str).str.contains("サイバー|セキュリティ", regex=True, na=False),
        "construction_infra": df["sector"].eq("建設業"),
        "electronics_semicap_proxy": df["sector"].isin(["電気機器", "機械", "精密機器"]),
        "banks_insurance": df["sector"].isin(["銀行業", "保険業"]),
    }

    theme_rows = []
    for name, mask in theme_defs.items():
        group = df[mask].copy()
        if group.empty:
            continue
        top20 = group.sort_values(["score", "trend_r2", "annual_return_pct"], ascending=[False, False, False]).head(20)
        stats = basket_return(price_map, top20["ticker"].astype(str).tolist())
        theme_rows.append(
            {
                "basket_type": "theme_top20",
                "label": name,
                "source_count": int(len(group)),
                **{k: v for k, v in stats.items() if k != "details"},
            }
        )
        pd.DataFrame(stats.get("details", [])).to_csv(OUT_DIR / f"theme_{name}_top20_details.csv", index=False, encoding="utf-8-sig")

    sector_df = pd.DataFrame(sector_rows).sort_values("avg_return_pct", ascending=False)
    theme_df = pd.DataFrame(theme_rows).sort_values("avg_return_pct", ascending=False)
    sector_df.to_csv(OUT_DIR / "sector_summary.csv", index=False, encoding="utf-8-sig")
    theme_df.to_csv(OUT_DIR / "theme_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "sector_best": sector_df.head(10).to_dict(orient="records"),
        "theme_best": theme_df.to_dict(orient="records"),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

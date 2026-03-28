from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "projects" / "shikiho_text_parser" / "output" / "2025_2q_sector_concentration_candidate"

BASE_CSV = ROOT / "projects" / "quarterly_ranker" / "output" / "2025_2q_pre_analysis_20250331_candidate" / "pre_shikiho_feature_ranking.csv"
PRICE_DIR = ROOT / "data" / "prices"
START_DATE = pd.Timestamp("2025-04-01")
END_DATE = pd.Timestamp("2025-06-30")


def load_realized_return(ticker: str) -> float | None:
    path = PRICE_DIR / f"{ticker.replace('.', '_')}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "Date" not in df.columns or "Close" not in df.columns:
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    mask = (df["Date"] >= START_DATE) & (df["Date"] <= END_DATE)
    q = df.loc[mask, ["Date", "Close"]].copy()
    if q.empty:
        return None
    start = float(q.iloc[0]["Close"])
    end = float(q.iloc[-1]["Close"])
    return (end / start - 1.0) * 100.0


def top_share(values: pd.Series, n: int) -> float | None:
    pos = values.clip(lower=0).sort_values(ascending=False)
    total = float(pos.sum())
    if total <= 0:
        return None
    return float(pos.head(n).sum() / total)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(BASE_CSV)
    df = df.drop_duplicates(subset=["ticker"], keep="first").copy()
    df["realized_return_pct"] = df["ticker"].astype(str).map(load_realized_return)
    df = df.dropna(subset=["realized_return_pct"]).copy()

    sector_rows: list[dict] = []
    for sector, g in df.groupby("sector"):
        realized = g["realized_return_pct"].astype(float)
        row = {
            "sector": sector,
            "count": int(len(g)),
            "mean_realized_return_pct": float(realized.mean()),
            "median_realized_return_pct": float(realized.median()),
            "std_realized_return_pct": float(realized.std(ddof=0)) if len(g) > 1 else 0.0,
            "positive_ratio": float((realized > 0).mean()),
            "top1_positive_share": top_share(realized, 1),
            "top3_positive_share": top_share(realized, 3),
            "avg_score": float(g["score"].astype(float).mean()),
            "avg_trend_r2": float(g["trend_r2"].astype(float).mean()),
            "avg_end_to_high_pct": float(g["end_to_trailing_high_pct"].astype(float).mean()),
        }
        sector_rows.append(row)
    sector_df = pd.DataFrame(sector_rows)

    # Lower concentration is better. Keep sectors with enough breadth and without extreme positive concentration.
    usable = sector_df[sector_df["count"] >= 5].copy()
    if not usable.empty:
        usable["concentration_score"] = (
            usable["top3_positive_share"].fillna(1.0) * 0.5
            + usable["std_realized_return_pct"].rank(pct=True, method="average") * 0.3
            + (1.0 - usable["positive_ratio"]) * 0.2
        )
        usable = usable.sort_values(
            ["concentration_score", "mean_realized_return_pct", "count"],
            ascending=[True, False, False],
        )
    else:
        usable["concentration_score"] = []

    sector_df.sort_values("mean_realized_return_pct", ascending=False).to_csv(
        OUT_DIR / "sector_summary.csv", index=False, encoding="utf-8-sig"
    )
    usable.to_csv(OUT_DIR / "sector_low_concentration_candidates.csv", index=False, encoding="utf-8-sig")

    selected_sectors = usable.head(5)["sector"].tolist() if not usable.empty else []
    filtered = df[df["sector"].isin(selected_sectors)].copy() if selected_sectors else df.iloc[0:0].copy()
    filtered.sort_values(["score", "trend_r2", "annual_return_pct"], ascending=[False, False, False]).to_csv(
        OUT_DIR / "filtered_universe.csv", index=False, encoding="utf-8-sig"
    )

    payload = {
        "period_start": START_DATE.strftime("%Y-%m-%d"),
        "period_end": END_DATE.strftime("%Y-%m-%d"),
        "universe_count": int(len(df)),
        "sector_count": int(sector_df["sector"].nunique()) if not sector_df.empty else 0,
        "selected_sectors": selected_sectors,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

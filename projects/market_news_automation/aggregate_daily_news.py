from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from projects.market_news_automation.paths import DAILY_FEATURES_CSV, OUTPUT_DIR, SNAPSHOT_CSV


def main() -> int:
    if not SNAPSHOT_CSV.exists():
        raise FileNotFoundError(f"snapshot csv not found: {SNAPSHOT_CSV}")
    df = pd.read_csv(SNAPSHOT_CSV)
    if df.empty:
        raise ValueError("snapshot csv is empty")

    df["capture_date"] = pd.to_datetime(df["capture_date"]).dt.normalize()
    for col in ("headline_count", "positive_count", "negative_count", "sentiment_score"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    daily = (
        df.groupby("capture_date")
        .agg(
            snapshot_count=("name", "count"),
            headline_count_sum=("headline_count", "sum"),
            headline_count_mean=("headline_count", "mean"),
            sentiment_mean=("sentiment_score", "mean"),
            sentiment_min=("sentiment_score", "min"),
            sentiment_max=("sentiment_score", "max"),
            positive_count_sum=("positive_count", "sum"),
            negative_count_sum=("negative_count", "sum"),
        )
        .reset_index()
        .rename(columns={"capture_date": "date"})
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_csv(DAILY_FEATURES_CSV, index=False, encoding="utf-8-sig")
    print(json.dumps({"rows": int(len(daily)), "output": str(DAILY_FEATURES_CSV)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

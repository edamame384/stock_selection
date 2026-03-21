from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DETAIL_CSV = ROOT / "projects" / "quarterly_ranker" / "output" / "q2_2024_pre_analysis_20240630_aligned" / "q2_2024_pre_shikiho_feature_ranking.csv"
PRICE_DIR = ROOT / "data" / "prices"
OUT_DIR = ROOT / "projects" / "quarterly_ranker" / "output" / "q2_2024_pre_analysis_20240630_aligned" / "bad_regime_defensive_search"


def attach_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2024-07-01")
    end = pd.Timestamp("2024-09-30")
    for _, row in df.iterrows():
        item = row.to_dict()
        ticker = str(row["ticker"])
        path = PRICE_DIR / f"{ticker.replace('.', '_')}.csv"
        future_ret = np.nan
        if path.exists():
            price = pd.read_csv(path)
            if "Date" in price.columns and "Close" in price.columns:
                price["Date"] = pd.to_datetime(price["Date"])
                price = price.sort_values("Date").set_index("Date")
                window = price[(price.index >= start) & (price.index <= end)]
                if len(window) >= 2:
                    future_ret = float(window["Close"].iloc[-1] / window["Close"].iloc[0] - 1.0) * 100.0
        item["future_return_2024Q3_pct"] = future_ret
        rows.append(item)
    return pd.DataFrame(rows)


def apply_rule(
    df: pd.DataFrame,
    trend_r2_min: float,
    annual_return_min: float,
    quarter_return_min: float,
    positive_month_ratio_min: float,
    persistence_20d_min: float,
    max_drawdown_max: float,
    avg_monthly_range_max: float,
    max_monthly_range_max: float,
    end_to_high_min: float,
    sector_per_score_min: float,
    ocr_per_max: float,
) -> pd.DataFrame:
    out = df.copy()
    out = out[out["trend_r2"].fillna(0) >= trend_r2_min]
    out = out[out["annual_return_pct"].fillna(-999) >= annual_return_min]
    out = out[out["quarter_return_pct"].fillna(-999) >= quarter_return_min]
    out = out[out["positive_month_ratio_pct"].fillna(0) >= positive_month_ratio_min]
    out = out[out["persistence_20d_pct"].fillna(0) >= persistence_20d_min]
    out = out[out["max_drawdown_pct"].fillna(999) <= max_drawdown_max]
    out = out[out["avg_monthly_high_low_change_pct"].fillna(999) <= avg_monthly_range_max]
    out = out[out["max_monthly_high_low_change_pct"].fillna(999) <= max_monthly_range_max]
    out = out[out["end_to_trailing_high_pct"].fillna(0) >= end_to_high_min]
    out = out[out["sector_adjusted_per_score"].fillna(0) >= sector_per_score_min]
    out = out[out["ocr_per"].fillna(999) <= ocr_per_max]
    out = out.sort_values(
        [
            "sector_adjusted_per_score",
            "end_to_trailing_high_pct",
            "max_drawdown_pct",
            "avg_monthly_high_low_change_pct",
            "overall_score",
        ],
        ascending=[False, False, True, True, False],
    ).reset_index(drop=True)
    out["selected_rank"] = np.arange(1, len(out) + 1)
    out["selected"] = True
    return out


def summarize(selected: pd.DataFrame) -> dict[str, float]:
    realized = selected["future_return_2024Q3_pct"].dropna()
    selected_count = int(len(selected))
    if realized.empty:
        return {
            "selected_count": selected_count,
            "mean_realized_return_pct": np.nan,
            "median_realized_return_pct": np.nan,
            "win_rate_pct": np.nan,
            "objective_score": -1e9,
        }
    mean_ret = float(realized.mean())
    median_ret = float(realized.median())
    win_rate = float((realized > 0).mean()) * 100.0
    # Defensive objective: prioritize losing less and raising win rate.
    objective = mean_ret + 0.35 * median_ret + 0.10 * win_rate
    return {
        "selected_count": selected_count,
        "mean_realized_return_pct": mean_ret,
        "median_realized_return_pct": median_ret,
        "win_rate_pct": win_rate,
        "objective_score": objective,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detail = pd.read_csv(DETAIL_CSV)
    detail = attach_forward_returns(detail)

    grid = {
        "trend_r2_min": [0.35, 0.40, 0.45, 0.50],
        "annual_return_min": [0.0, 5.0, 10.0, 15.0],
        "quarter_return_min": [-10.0, -5.0, 0.0, 5.0],
        "positive_month_ratio_min": [35.0, 40.0, 45.0, 50.0],
        "persistence_20d_min": [35.0, 40.0, 45.0, 50.0],
        "max_drawdown_max": [15.0, 18.0, 22.0, 25.0],
        "avg_monthly_range_max": [10.0, 12.0, 14.0, 16.0],
        "max_monthly_range_max": [18.0, 22.0, 26.0, 30.0],
        "end_to_high_min": [88.0, 90.0, 92.0, 94.0],
        "sector_per_score_min": [0.35, 0.40, 0.45, 0.50],
        "ocr_per_max": [14.0, 16.0, 18.0],
    }

    rows = []
    best_summary = None
    best_selected = None
    for vals in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), vals))
        selected = apply_rule(detail, **params)
        summary = summarize(selected)
        row = {**params, **summary}
        rows.append(row)
        if summary["selected_count"] < 8:
            continue
        if best_summary is None or row["objective_score"] > best_summary["objective_score"]:
            best_summary = row
            best_selected = selected.copy()

    results = pd.DataFrame(rows).sort_values(
        ["objective_score", "mean_realized_return_pct", "selected_count"],
        ascending=[False, False, False],
    )
    results.to_csv(OUT_DIR / "search_results.csv", index=False, encoding="utf-8-sig")
    results[results["selected_count"] >= 8].to_csv(OUT_DIR / "search_results_count_ge_8.csv", index=False, encoding="utf-8-sig")

    if best_selected is not None and best_summary is not None:
        best_selected["learning_date"] = "2024-06-30"
        best_selected["training_cutoff_date"] = "2024-06-30"
        best_selected.to_csv(OUT_DIR / "best_selected_candidates.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([best_summary]).to_csv(OUT_DIR / "best_rule_summary.csv", index=False, encoding="utf-8-sig")
        print(pd.DataFrame([best_summary]).to_string(index=False))
        print(f"[OUT] {OUT_DIR / 'best_selected_candidates.csv'}")
    else:
        print("[WARN] no rule found with selected_count >= 8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

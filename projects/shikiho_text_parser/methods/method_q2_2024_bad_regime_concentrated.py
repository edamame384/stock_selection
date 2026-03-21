from __future__ import annotations

METHOD_ID = "q2_2024_bad_regime_concentrated"
METHOD_JA = "悪条件集中防御"

SELECTION_RULE = {
    "trend_r2_min": 0.35,
    "annual_return_min": 0.0,
    "quarter_return_min": -10.0,
    "positive_month_ratio_min": 35.0,
    "persistence_20d_min": 35.0,
    "max_drawdown_max": 18.0,
    "avg_monthly_range_max": 14.0,
    "end_to_trailing_high_min": 88.0,
    "sector_adjusted_per_score_min": 0.35,
    "ocr_per_max": 18.0,
    "sort_col": "sector_adjusted_per_score",
    "sort_ascending": False,
    "top_n": 5,
}

TRADING_RULE = {
    "entry_limit_pct": 1.0,
    "take_profit_pct": 5.0,
    "stop_loss_pct": 4.0,
}


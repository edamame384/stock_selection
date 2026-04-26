from __future__ import annotations

METHOD_ID = "post_crash_broad"
METHOD_JA = "暴落後拡張"

SELECTION_RULE = {
    "trend_r2_min": 0.60,
    "annual_return_min": 25.0,
    "quarter_return_min": 10.0,
    "positive_month_ratio_min": 50.0,
    "persistence_20d_min": 55.0,
    "sector_adjusted_per_score_min": 0.35,
    "ocr_per_max": 20.0,
}

TRADING_RULE = {
    "entry_limit_pct": 1.0,
    "take_profit_pct": 5.0,
    "stop_loss_pct": 4.0,
}

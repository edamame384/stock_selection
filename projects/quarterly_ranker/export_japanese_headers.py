from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError


COLUMN_MAP = {
    "ticker": "ティッカー",
    "company_name": "銘柄名",
    "base_rank": "元ランキング順位",
    "base_score": "元ランキングスコア",
    "annual_return_pct": "年間上昇率",
    "quarter_return_pct": "四半期上昇率",
    "trend_r2": "上昇トレンド滑らかさ",
    "max_drawdown_pct": "最大下落率",
    "avg_monthly_high_low_change_pct": "月間高安騰落率平均",
    "persistence_20d_pct": "20日移動平均維持率",
    "image_found": "画像有無",
    "ocr_positive_count": "OCR好材料語数",
    "ocr_negative_count": "OCR悪材料語数",
    "ocr_theme_count": "OCRテーマ語数",
    "ocr_shareholder_count": "OCR株主還元語数",
    "ocr_per": "OCR予想PER",
    "ocr_pbr": "OCRPBR",
    "ocr_roe": "OCRROE",
    "ocr_roa": "OCRROA",
    "ocr_yield": "OCR総利回り",
    "ocr_per_confidence": "OCR予想PER信頼度",
    "ocr_pbr_confidence": "OCRPBR信頼度",
    "ocr_roe_confidence": "OCRROE信頼度",
    "ocr_roa_confidence": "OCRROA信頼度",
    "ocr_yield_confidence": "OCR総利回り信頼度",
    "ocr_dividend_hint": "OCR配当金ヒント",
    "ocr_earnings_num_count": "OCR業績表数値数",
    "ocr_earnings_num_mean": "OCR業績表数値平均",
    "ocr_sales_points": "売上系列点数",
    "ocr_op_points": "営業利益系列点数",
    "ocr_np_points": "純利益系列点数",
    "ocr_eps_points": "EPS系列点数",
    "ocr_div_points": "配当系列点数",
    "ocr_sales_trend": "売上トレンド",
    "ocr_op_trend": "営業利益トレンド",
    "ocr_np_trend": "純利益トレンド",
    "ocr_eps_trend": "EPSトレンド",
    "ocr_div_trend": "配当トレンド",
    "business_score_raw": "特色事業スコア生値",
    "article_score_raw": "記事スコア生値",
    "finance_score_raw": "財務スコア生値",
    "chart_score_raw": "チャートスコア生値",
    "earnings_score_raw": "業績スコア生値",
    "dividend_score_raw": "配当還元スコア生値",
    "surge_type_score_raw": "業績急伸型スコア生値",
    "stable_type_score_raw": "安定進捗型スコア生値",
    "business_score": "特色事業スコア",
    "business_score_rank": "特色事業順位",
    "article_score": "記事スコア",
    "article_score_rank": "記事順位",
    "finance_score": "財務スコア",
    "finance_score_rank": "財務順位",
    "chart_score": "チャートスコア",
    "chart_score_rank": "チャート順位",
    "earnings_score": "業績スコア",
    "earnings_score_rank": "業績順位",
    "dividend_score": "配当還元スコア",
    "dividend_score_rank": "配当還元順位",
    "surge_type_score": "業績急伸型スコア",
    "surge_type_score_rank": "業績急伸型順位",
    "stable_type_score": "安定進捗型スコア",
    "stable_type_score_rank": "安定進捗型順位",
    "overall_shikiho_score": "四季報総合スコア",
    "overall_shikiho_rank": "四季報総合順位",
    "ocr_style_label": "OCR型分類",
    "ocr_style_rank_within_label": "型内順位",
    "ocr_style_description": "型の説明",
    "adopt_threshold_summary": "採用基準要約",
    "adopt_flag": "採用判定",
    "adopt_reason": "採用判定理由",
    "feature": "特徴量名",
    "pearson_with_base_score": "元スコアとのピアソン相関",
    "spearman_with_base_score": "元スコアとのスピアマン相関",
    "pearson_with_base_rank_inv": "元順位とのピアソン相関",
    "spearman_with_base_rank_inv": "元順位とのスピアマン相関",
    "quarter": "四半期",
    "style": "型分類",
    "count": "件数",
    "score_threshold_top20_min": "上位20下限スコア",
    "score_threshold_top20_median": "上位20中央値スコア",
    "annual_return_pct_min": "上位20年間上昇率下限",
    "annual_return_pct_median": "上位20年間上昇率中央値",
    "quarter_return_pct_min": "上位20四半期上昇率下限",
    "quarter_return_pct_median": "上位20四半期上昇率中央値",
    "trend_r2_min": "上位20トレンド滑らかさ下限",
    "trend_r2_median": "上位20トレンド滑らかさ中央値",
    "max_drawdown_pct_max": "上位20最大下落率上限",
    "max_drawdown_pct_median": "上位20最大下落率中央値",
    "earnings_score_raw_min": "上位20業績スコア生値下限",
    "surge_type_score_raw_min": "上位20業績急伸型生値下限",
    "ocr_positive_count_median": "上位20好材料語数中央値",
    "finance_score_raw_min": "上位20財務スコア生値下限",
    "stable_type_score_raw_min": "上位20安定進捗型生値下限",
    "ocr_roe_median": "上位20ROE中央値",
    "ocr_pbr_max": "上位20PBR上限",
}


TARGETS = [
    Path("projects/quarterly_ranker/output/q2_shikiho_analysis/q2_shikiho_feature_ranking.csv"),
    Path("projects/quarterly_ranker/output/q2_shikiho_analysis/q2_shikiho_feature_ranking_annotated.csv"),
    Path("projects/quarterly_ranker/output/q2_shikiho_analysis/q2_shikiho_feature_correlations.csv"),
    Path("projects/quarterly_ranker/output/q3_shikiho_analysis/q3_shikiho_feature_ranking.csv"),
    Path("projects/quarterly_ranker/output/q3_shikiho_analysis/q3_shikiho_feature_ranking_annotated.csv"),
    Path("projects/quarterly_ranker/output/q3_shikiho_analysis/q3_shikiho_feature_correlations.csv"),
    Path("projects/quarterly_ranker/output/style_summaries/Q2_業績急伸型_top20.csv"),
    Path("projects/quarterly_ranker/output/style_summaries/Q2_安定進捗型_top20.csv"),
    Path("projects/quarterly_ranker/output/style_summaries/Q3_業績急伸型_top20.csv"),
    Path("projects/quarterly_ranker/output/style_summaries/Q3_安定進捗型_top20.csv"),
    Path("projects/quarterly_ranker/output/style_summaries/style_thresholds_q2_q3.csv"),
]


def main() -> int:
    for path in TARGETS:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except EmptyDataError:
            continue
        renamed = df.rename(columns={col: COLUMN_MAP.get(col, col) for col in df.columns})
        out_path = path.with_name(path.stem + "_ja.csv")
        renamed.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[OUT] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

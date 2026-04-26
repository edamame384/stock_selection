from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("projects/quarterly_ranker/output")
THRESHOLDS = ROOT / "style_summaries" / "style_thresholds_q2_q3.csv"


def build_style_description(style: str) -> str:
    if style == "業績急伸型":
        return "業績・EPS・増益ワード・テーマ性を重視する型。短期的な利益成長や業績加速が強い銘柄を優先する。"
    if style == "安定進捗型":
        return "ROE/ROA・PER/PBR・継続上昇・ドローダウンの浅さを重視する型。安定して進捗し続ける銘柄を優先する。"
    return "業績急伸型と安定進捗型の中間。どちらか一方へ強く寄っていない。"


def format_threshold_summary(row: pd.Series) -> str:
    style = row["style"]
    if style == "業績急伸型":
        return (
            f"surge_type_score>={row['score_threshold_top20_min']:.3f}, "
            f"annual_return_pct>={row['annual_return_pct_min']:.1f}, "
            f"quarter_return_pct>={row['quarter_return_pct_min']:.1f}, "
            f"trend_r2>={row['trend_r2_min']:.3f}, "
            f"earnings_score_raw>={row['earnings_score_raw_min']:.1f}"
        )
    return (
        f"stable_type_score>={row['score_threshold_top20_min']:.3f}, "
        f"annual_return_pct>={row['annual_return_pct_min']:.1f}, "
        f"quarter_return_pct>={row['quarter_return_pct_min']:.1f}, "
        f"trend_r2>={row['trend_r2_min']:.3f}, "
        f"finance_score_raw>={row['finance_score_raw_min']:.2f}"
    )


def adopt_flag(record: pd.Series, threshold: pd.Series) -> tuple[str, str]:
    style = record["ocr_style_label"]
    if style == "中間型":
        return "非採用", "中間型のため、型別優位性が不足"

    if style == "業績急伸型":
        checks = [
            ("surge_type_score", record["surge_type_score"] >= threshold["score_threshold_top20_min"]),
            ("annual_return_pct", record["annual_return_pct"] >= threshold["annual_return_pct_min"]),
            ("quarter_return_pct", record["quarter_return_pct"] >= threshold["quarter_return_pct_min"]),
            ("trend_r2", record["trend_r2"] >= threshold["trend_r2_min"]),
            ("earnings_score_raw", record["earnings_score_raw"] >= threshold["earnings_score_raw_min"]),
        ]
    else:
        checks = [
            ("stable_type_score", record["stable_type_score"] >= threshold["score_threshold_top20_min"]),
            ("annual_return_pct", record["annual_return_pct"] >= threshold["annual_return_pct_min"]),
            ("quarter_return_pct", record["quarter_return_pct"] >= threshold["quarter_return_pct_min"]),
            ("trend_r2", record["trend_r2"] >= threshold["trend_r2_min"]),
            ("finance_score_raw", record["finance_score_raw"] >= threshold["finance_score_raw_min"]),
        ]

    passed = [name for name, ok in checks if ok]
    failed = [name for name, ok in checks if not ok]
    if len(passed) >= 4:
        reason = "基準達成: " + ", ".join(passed)
        if failed:
            reason += " | 未達: " + ", ".join(failed)
        return "採用", reason
    reason = "未達: " + ", ".join(failed)
    if passed:
        reason += " | 達成: " + ", ".join(passed)
    return "非採用", reason


def annotate_quarter(quarter: str) -> None:
    thresholds = pd.read_csv(THRESHOLDS)
    detail_path = ROOT / f"{quarter.lower()}_shikiho_analysis" / f"{quarter.lower()}_shikiho_feature_ranking.csv"
    df = pd.read_csv(detail_path)

    threshold_map = {
        row["style"]: row
        for _, row in thresholds[thresholds["quarter"] == quarter].iterrows()
    }

    style_desc = []
    threshold_desc = []
    adopt = []
    adopt_reason = []
    for _, row in df.iterrows():
        style = row["ocr_style_label"]
        style_desc.append(build_style_description(style))
        thr = threshold_map.get(style)
        if thr is None:
            threshold_desc.append("")
            adopt.append("非採用")
            adopt_reason.append("型別基準なし")
            continue
        threshold_desc.append(format_threshold_summary(thr))
        flag, reason = adopt_flag(row, thr)
        adopt.append(flag)
        adopt_reason.append(reason)

    df["ocr_style_description"] = style_desc
    df["adopt_threshold_summary"] = threshold_desc
    df["adopt_flag"] = adopt
    df["adopt_reason"] = adopt_reason

    out_path = ROOT / f"{quarter.lower()}_shikiho_analysis" / f"{quarter.lower()}_shikiho_feature_ranking_annotated.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OUT] {out_path}")


def main() -> int:
    annotate_quarter("Q2")
    annotate_quarter("Q3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

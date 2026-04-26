from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from projects.shikiho_text_parser.parse_shikiho_text import collect_input_files, parse_shikiho_text
from projects.shikiho_text_parser.paths import PRICE_DIR, SECTOR_MASTER_PATH
from projects.quarterly_ranker.operational_csv_utils import write_operational_csv
from projects.quarterly_ranker.rank_quarterly_promising_stocks import calc_max_drawdown, calc_r2_log_trend


POSITIVE_WORDS = [
    "最高益", "連続増益", "増益", "増収", "上振れ", "改善", "好調", "拡大", "増配", "復配", "更新",
]
NEGATIVE_WORDS = [
    "減益", "減収", "赤字", "鈍化", "苦戦", "下振れ", "無配", "停滞", "一服",
]


def load_sector_master(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.rename(columns={"symbol": "ticker", "sector": "sector_33"})[["ticker", "sector_33"]]


def safe_float(text: str) -> float | None:
    if text is None:
        return None
    t = str(text).replace(",", "").replace("円", "").replace("倍", "").replace("%", "").replace("+", "").strip()
    if t in {"", "―", "-"}:
        return None
    try:
        return float(t)
    except ValueError:
        return None


def parse_forecast_per(raw_text: str) -> float | None:
    m = re.search(r"予想PER[\s\S]{0,80}?連27\.[0-9]\s*([0-9.]+)倍", raw_text)
    if not m:
        m = re.search(r"予想PER[\s\S]{0,40}?([0-9.]+)倍", raw_text)
    return safe_float(m.group(1)) if m else None


def parse_forecast_yield(raw_text: str) -> float | None:
    m = re.search(r"予想配当利回り[\s\S]{0,80}?連27\.[0-9]\s*([0-9.]+)%", raw_text)
    if not m:
        m = re.search(r"予想配当利回り[\s\S]{0,40}?([0-9.]+)%", raw_text)
    return safe_float(m.group(1)) if m else None


def extract_earnings_growth(parsed: dict) -> dict[str, float | None]:
    rows = parsed.get("earnings_rows", [])
    actual = None
    forecast = None
    for row in rows:
        period = row.get("period", "")
        if period.startswith("連26.3"):
            actual = row.get("values", [])
        if period.startswith("連27.3予"):
            forecast = row.get("values", [])
    out = {"sales_growth_pct": None, "op_growth_pct": None, "np_growth_pct": None, "eps_growth_pct": None, "div_growth_pct": None}
    if not actual or not forecast or len(actual) < 6 or len(forecast) < 6:
        return out
    names = ["sales_growth_pct", "op_growth_pct", "np_growth_pct", "eps_growth_pct", "div_growth_pct"]
    pairs = [(0, 0), (1, 1), (3, 3), (4, 4), (5, 5)]
    for name, (a_idx, f_idx) in zip(names, pairs):
        a = safe_float(actual[a_idx])
        f = safe_float(re.sub(r"[^0-9.\-]", "", forecast[f_idx]))
        if a is not None and f is not None and a != 0:
            out[name] = (f / a - 1.0) * 100.0
    return out


def compute_price_metrics(ticker: str, price_dir: Path, cutoff_date: pd.Timestamp, quarter_start: pd.Timestamp) -> dict[str, float] | None:
    price_path = price_dir / f"{ticker.replace('.', '_')}.csv"
    if not price_path.exists():
        return None
    df = pd.read_csv(price_path)
    if "Date" not in df.columns or "Close" not in df.columns:
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    hist = df[df.index <= cutoff_date].copy()
    if hist.empty:
        return None
    qdf = hist[(hist.index >= quarter_start) & (hist.index <= cutoff_date)].copy()
    trailing = hist.tail(min(252, len(hist))).copy()
    annual_window_start = cutoff_date - pd.Timedelta(days=365)
    annual_df = hist[(hist.index >= annual_window_start) & (hist.index <= cutoff_date)].copy()
    if len(annual_df) < 100:
        annual_df = trailing.copy()
    if len(qdf) < 20 or len(trailing) < 120 or len(annual_df) < 100:
        return None

    monthly = qdf.resample("ME").agg({"High": "max", "Low": "min"}).dropna()
    if monthly.empty:
        return None

    annual_return = float(annual_df["Close"].iloc[-1] / annual_df["Close"].iloc[0] - 1.0)
    trailing_return = float(trailing["Close"].iloc[-1] / trailing["Close"].iloc[0] - 1.0)
    quarter_return = float(qdf["Close"].iloc[-1] / qdf["Close"].iloc[0] - 1.0)
    avg_monthly_range = float(((monthly["High"] / monthly["Low"]) - 1.0).mean())
    max_monthly_range = float(((monthly["High"] / monthly["Low"]) - 1.0).max())
    trend_r2 = calc_r2_log_trend(trailing["Close"])
    max_drawdown = calc_max_drawdown(trailing["Close"])
    monthly_close = trailing["Close"].resample("ME").last().dropna()
    monthly_ret = monthly_close.pct_change().dropna()
    positive_month_ratio = float((monthly_ret > 0).mean()) if len(monthly_ret) > 0 else np.nan
    ma20 = trailing["Close"].rolling(20).mean()
    ma60 = trailing["Close"].rolling(60).mean()
    last20 = trailing.iloc[-20:].copy()
    ma20_last20 = ma20.reindex(last20.index)
    ma60_last20 = ma60.reindex(last20.index)
    persistence_20d = float(((last20["Close"] > ma20_last20) & (ma20_last20 > ma60_last20)).mean())
    end_to_trailing_high = float(trailing["Close"].iloc[-1] / trailing["High"].max())

    return {
        "annual_return_pct": annual_return * 100.0,
        "trailing_return_pct": trailing_return * 100.0,
        "quarter_return_pct": quarter_return * 100.0,
        "avg_monthly_high_low_change_pct": avg_monthly_range * 100.0,
        "max_monthly_high_low_change_pct": max_monthly_range * 100.0,
        "trend_r2": trend_r2,
        "max_drawdown_pct": max_drawdown * 100.0,
        "positive_month_ratio_pct": positive_month_ratio * 100.0,
        "persistence_20d_pct": persistence_20d * 100.0,
        "end_to_trailing_high_pct": end_to_trailing_high * 100.0,
        "quarter_end_close": float(qdf["Close"].iloc[-1]),
        "price_rows": int(len(df)),
    }


def percentile(series: pd.Series, ascending: bool = True) -> pd.Series:
    return series.rank(pct=True, ascending=ascending, method="average").fillna(0.0)


def build_universe(input_dir: Path, price_dir: Path, sector_master: pd.DataFrame, cutoff_date: pd.Timestamp, quarter_start: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for txt_path in collect_input_files(input_dir):
        raw_text = txt_path.read_text(encoding="utf-8")
        parsed = parse_shikiho_text(raw_text, txt_path.name)
        basic = parsed["basic_info"]
        ticker_code = basic.get("ticker_code", txt_path.stem)
        ticker = f"{ticker_code}.T"
        if not basic.get("company_name"):
            continue
        if len(parsed.get("earnings_rows", [])) < 3:
            continue
        price_metrics = compute_price_metrics(ticker, price_dir, cutoff_date, quarter_start)
        if price_metrics is None:
            continue
        headline_text = " ".join([b["title"] + " " + b["body"] for b in parsed.get("headline_blocks", [])])
        pos_count = sum(headline_text.count(word) for word in POSITIVE_WORDS)
        neg_count = sum(headline_text.count(word) for word in NEGATIVE_WORDS)
        earnings_growth = extract_earnings_growth(parsed)
        indicators = parsed.get("stock_indicators", {})
        financials = parsed.get("financials", {})
        scores = parsed.get("shikiho_scores", {})
        simple_sector = "/".join(basic.get("categories", [])[:2])
        rows.append(
            {
                "ticker": ticker,
                "company_name": basic.get("company_name", ""),
                "market": basic.get("market", ""),
                "simple_sector": simple_sector,
                "sector": simple_sector,
                "feature_summary": parsed.get("feature_summary", ""),
                "segment_summary": parsed.get("segment_summary", ""),
                "flags": " / ".join(basic.get("flags", [])),
                "headline_positive_count": pos_count,
                "headline_negative_count": neg_count,
                "headline_score_raw": pos_count - neg_count,
                "shikiho_score_overall": scores.get("overall"),
                "score_growth": scores.get("成長性"),
                "score_profitability": scores.get("収益性"),
                "score_safety": scores.get("安全性"),
                "score_undervalued": scores.get("割安度"),
                "score_momentum": scores.get("値上がり"),
                "forecast_per": parse_forecast_per(raw_text),
                "actual_pbr": safe_float(indicators.get("actual_pbr", "")),
                "forecast_yield": parse_forecast_yield(raw_text),
                "ytd_return_text_pct": safe_float(indicators.get("ytd_return", "")),
                "ma200_gap_pct": safe_float(indicators.get("ma200_gap", "")),
                "equity_ratio_pct": safe_float(financials.get("自己資本比率", "")),
                **earnings_growth,
                **price_metrics,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.merge(sector_master, how="left", on="ticker")
    out["sector_33"] = out["sector_33"].fillna("-")
    return out


def score_universe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sector_adjusted_per_score"] = 0.0
    global_per = percentile(out["forecast_per"], ascending=False)
    for sector, idx in out["simple_sector"].fillna("").groupby(out["simple_sector"].fillna("")).groups.items():
        vals = pd.to_numeric(out.loc[list(idx), "forecast_per"], errors="coerce")
        if vals.notna().sum() >= 5 and sector not in {"", "/"}:
            out.loc[list(idx), "sector_adjusted_per_score"] = percentile(vals, ascending=False)
        else:
            out.loc[list(idx), "sector_adjusted_per_score"] = global_per.loc[list(idx)]
    out["fundamental_score"] = (
        0.18 * percentile(out["shikiho_score_overall"], ascending=True)
        + 0.12 * percentile(out["score_growth"], ascending=True)
        + 0.10 * percentile(out["score_profitability"], ascending=True)
        + 0.10 * percentile(out["score_safety"], ascending=True)
        + 0.10 * percentile(out["score_undervalued"], ascending=True)
        + 0.10 * percentile(out["headline_score_raw"], ascending=True)
        + 0.15 * percentile(out["op_growth_pct"], ascending=True)
        + 0.15 * percentile(out["eps_growth_pct"], ascending=True)
    )
    out["price_score"] = (
        0.28 * percentile(out["annual_return_pct"], ascending=True)
        + 0.22 * percentile(out["quarter_return_pct"], ascending=True)
        + 0.20 * percentile(out["trend_r2"], ascending=True)
        + 0.12 * percentile(out["persistence_20d_pct"], ascending=True)
        + 0.08 * percentile(out["positive_month_ratio_pct"], ascending=True)
        + 0.10 * percentile(out["max_drawdown_pct"], ascending=False)
    )
    out["valuation_score"] = (
        0.55 * percentile(out["sector_adjusted_per_score"], ascending=True)
        + 0.25 * percentile(out["actual_pbr"], ascending=False)
        + 0.20 * percentile(out["forecast_yield"], ascending=True)
    )
    out["promising_score"] = (
        0.45 * out["price_score"]
        + 0.35 * out["fundamental_score"]
        + 0.20 * out["valuation_score"]
    )
    out["selected"] = (
        (out["promising_score"] >= 0.72)
        & (out["annual_return_pct"].between(20.0, 180.0))
        & (out["quarter_return_pct"].between(10.0, 60.0))
        & (out["trend_r2"] >= 0.60)
        & (out["persistence_20d_pct"] >= 55.0)
        & ((out["headline_score_raw"] >= 0) | (out["shikiho_score_overall"] >= 3))
        & ((out["sector_adjusted_per_score"] >= 0.35) | out["forecast_per"].isna())
    )
    out["selection_reason"] = (
        "score="
        + out["promising_score"].round(3).astype(str)
        + " | annual="
        + out["annual_return_pct"].round(1).astype(str)
        + " | quarter="
        + out["quarter_return_pct"].round(1).astype(str)
        + " | trend_r2="
        + out["trend_r2"].round(3).astype(str)
        + " | per_score="
        + out["sector_adjusted_per_score"].round(3).astype(str)
    )
    return out.sort_values(["selected", "promising_score", "price_score", "fundamental_score"], ascending=[False, False, False, False])


def main() -> int:
    parser = argparse.ArgumentParser(description="Build latest 2026-2Q text-based universe and operational selection.")
    parser.add_argument("--input-dir", type=Path, default=Path(r"C:\Users\mitsu\OneDrive\ドキュメント\四季報DB2025\2026-2Q"))
    parser.add_argument("--price-dir", type=Path, default=PRICE_DIR)
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "projects" / "shikiho_text_parser" / "output" / "2026_2q_selection")
    parser.add_argument("--sector-master", type=Path, default=SECTOR_MASTER_PATH)
    parser.add_argument("--cutoff-date", type=str, default="2026-03-27")
    parser.add_argument("--quarter-start", type=str, default="2025-12-29")
    parser.add_argument("--effective-start", type=str, default="2026-03-30")
    parser.add_argument("--effective-end", type=str, default="2026-06-30")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sector_master = load_sector_master(args.sector_master)
    cutoff_date = pd.Timestamp(args.cutoff_date)
    quarter_start = pd.Timestamp(args.quarter_start)
    universe = build_universe(args.input_dir, args.price_dir, sector_master, cutoff_date, quarter_start)
    scored = score_universe(universe)

    universe_path = args.output_dir / "2026_2q_scored_universe.csv"
    selected_path = args.output_dir / "2026_2q_selected_candidates.csv"
    operational_selected_path = args.output_dir / "operational" / "2026_2q_selected_candidates_operational.csv"
    summary_path = args.output_dir / "2026_2q_selection_summary.json"

    scored.to_csv(universe_path, index=False, encoding="utf-8-sig")
    selected_df = scored[scored["selected"]].copy()
    selected_df.to_csv(selected_path, index=False, encoding="utf-8-sig")
    write_operational_csv(selected_df, operational_selected_path)

    summary = {
        "selection_date": args.cutoff_date,
        "effective_start": args.effective_start,
        "effective_end": args.effective_end,
        "universe_count": int(len(scored)),
        "selected_count": int(scored["selected"].sum()),
        "score_threshold": 0.72,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OUT] universe={universe_path}")
    print(f"[OUT] selected={selected_path}")
    print(f"[OUT] selected_operational={operational_selected_path}")
    print(f"[OUT] summary={summary_path}")
    print(f"[INFO] universe={len(scored)} selected={int(scored['selected'].sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

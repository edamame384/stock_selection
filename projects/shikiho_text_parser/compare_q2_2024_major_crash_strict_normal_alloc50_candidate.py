from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.shikiho_text_parser import backtest_phase_adaptive_practical_v35_alloc50_candidate as alloc50_mod  # noqa: E402
from projects.shikiho_text_parser.backtest_phase_adaptive_practical_v35_alloc50_candidate import TableSpec  # noqa: E402
from projects.shikiho_text_parser.search_phase_method_optimization import PHASE_CSV  # noqa: E402


OUT_DIR = ROOT / "projects" / "shikiho_text_parser" / "output" / "q2_2024_major_crash_strict_normal_alloc50_candidate"


def q2_spec() -> TableSpec:
    return TableSpec(
        "q2_2024",
        ROOT / "projects" / "quarterly_ranker" / "output" / "q2_2024_pre_analysis_20240630_aligned" / "q2_2024_pre_shikiho_feature_ranking.csv",
        ROOT / "projects" / "quarterly_ranker" / "output" / "q2_2024_pre_analysis_20240630_aligned" / "operational" / "q2_2024_pre_selected_candidates_operational.csv",
        "2024-07-01",
        "2024-09-30",
        3_000_000.0,
    )


def load_phase_frame() -> pd.DataFrame:
    df = pd.read_csv(PHASE_CSV, skiprows=[1])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.rename(columns={"phase": "phase_name"})
    for col in ["ret5", "dd20", "vol10"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["Date", "phase_name", "ret5", "dd20", "vol10"]].copy()


def classify_major_crash(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["major_crash"] = (out["phase_name"].astype(str) == "crash") & ((out["ret5"] <= -0.08) | (out["dd20"] <= -0.14))
    return out


def transformed_phase_map(df: pd.DataFrame, normal_lookback_days: int = 10) -> pd.Series:
    x = df.copy().reset_index(drop=True)
    x["phase_name_mod"] = x["phase_name"].astype(str)
    x.loc[x["major_crash"], "phase_name_mod"] = "downtrend"

    for idx in range(len(x)):
        if str(x.loc[idx, "phase_name"]) != "normal":
            continue
        prior_mod = [str(v) for v in x.loc[max(0, idx - normal_lookback_days): idx - 1, "phase_name_mod"].tolist()]
        prior_raw = [str(v) for v in x.loc[max(0, idx - normal_lookback_days): idx - 1, "phase_name"].tolist()]
        if "downtrend" in prior_mod and ("high_vol" in prior_raw or "capitulation_end" in prior_raw):
            x.loc[idx, "phase_name_mod"] = "downtrend"
    return x.set_index("Date")["phase_name_mod"].sort_index()


def prepare_earnings_map(spec: TableSpec) -> dict[str, list[pd.Timestamp]]:
    detail_df = pd.read_csv(spec.detail_csv)
    tickers = set(detail_df["ticker"].astype(str).tolist())
    sources: list[pd.DataFrame] = []
    for path in [
        ROOT / "data" / "earnings_cache" / "irbank_earnings_dates.csv",
        ROOT / "data" / "earnings_cache" / "yf_earnings_dates.csv",
        ROOT / "projects" / "shikiho_text_parser" / "output" / "parsed_summary.csv",
    ]:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "ticker" not in df.columns and "ticker_code" in df.columns:
            df["ticker"] = df["ticker_code"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True) + ".T"
        if "earnings_date" not in df.columns and "disclosure_date" in df.columns:
            df["earnings_date"] = df["disclosure_date"]
        if "ticker" not in df.columns or "earnings_date" not in df.columns:
            continue
        df = df[["ticker", "earnings_date"]].copy()
        df["ticker"] = df["ticker"].astype(str).str.strip()
        df = df[df["ticker"].isin(tickers)]
        sources.append(df)
    merged = pd.concat(sources, ignore_index=True) if sources else pd.DataFrame(columns=["ticker", "earnings_date"])
    if not merged.empty:
        merged["earnings_date"] = pd.to_datetime(merged["earnings_date"], errors="coerce")
        merged = merged.dropna(subset=["earnings_date"]).drop_duplicates(subset=["ticker", "earnings_date"]).sort_values(["ticker", "earnings_date"])
    return alloc50_mod.build_earnings_map(merged)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    spec = q2_spec()
    earnings_map = prepare_earnings_map(spec)
    phase_df = classify_major_crash(load_phase_frame())
    phase_map = transformed_phase_map(phase_df)

    result = alloc50_mod.run_dataset(
        spec,
        phase_map,
        earnings_map,
        earnings_pre_days=1,
        earnings_post_days=5,
        phase_shift_days=1,
        phase_proxy_mode="difficult_v11",
        rebuild_interval_trading_days=10,
        high_vol_take_profit_pct=None,
        weak_uptrend_take_profit_pct=None,
    )

    payload = {
        "variant": "major_crash_strict_normal_alloc50",
        "major_crash_definition": "phase=crash and (ret5<=-0.08 or dd20<=-0.14)",
        "major_crash_rule": "downtrend(no_trade)",
        "normal_after_major_crash_highvol_rule": "downtrend(no_trade)",
        **{k: v for k, v in result.items() if k not in {"trade_log", "equity_curve"}},
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    result["trade_log"].to_csv(OUT_DIR / "trade_log.csv", index=False, encoding="utf-8-sig")
    result["equity_curve"].to_csv(OUT_DIR / "equity_curve.csv", index=False, encoding="utf-8-sig")

    major_days = phase_df[phase_df["major_crash"]][["Date", "phase_name", "ret5", "dd20", "vol10"]].copy()
    major_days["Date"] = major_days["Date"].dt.strftime("%Y-%m-%d")
    major_days.to_csv(OUT_DIR / "major_crash_days.csv", index=False, encoding="utf-8-sig")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

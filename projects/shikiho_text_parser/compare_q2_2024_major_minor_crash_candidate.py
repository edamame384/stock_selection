from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import argparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.shikiho_text_parser.backtest_phase_adaptive_practical_v34_rebuild_interval_candidate import (  # noqa: E402
    TableSpec,
    build_earnings_map,
    run_dataset,
)
from projects.shikiho_text_parser.search_phase_method_optimization import PHASE_CSV  # noqa: E402


OUT_DIR = ROOT / "projects" / "shikiho_text_parser" / "output" / "q2_2024_major_minor_crash_candidate"


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


def classify_crash_scale(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["crash_scale"] = ""
    is_crash = out["phase_name"].astype(str) == "crash"
    is_major = is_crash & ((out["ret5"] <= -0.08) | (out["dd20"] <= -0.14))
    out.loc[is_crash & ~is_major, "crash_scale"] = "minor_crash"
    out.loc[is_major, "crash_scale"] = "major_crash"
    return out


def transformed_phase_map(df: pd.DataFrame, mode: str) -> pd.Series:
    x = df.copy()
    if mode == "baseline_v35":
        pass
    elif mode == "major_notrade_minor_defensive":
        x.loc[x["crash_scale"] == "major_crash", "phase_name"] = "downtrend"
    elif mode == "major_notrade_minor_q3":
        x.loc[x["crash_scale"] == "major_crash", "phase_name"] = "downtrend"
        x.loc[x["crash_scale"] == "minor_crash", "phase_name"] = "capitulation_end"
    else:
        raise ValueError(mode)
    return x.set_index("Date")["phase_name"].sort_index()


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
    return build_earnings_map(merged)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="", choices=["", "baseline_v35", "major_notrade_minor_defensive", "major_notrade_minor_q3"])
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    spec = q2_spec()
    earnings_map = prepare_earnings_map(spec)
    phase_df = classify_crash_scale(load_phase_frame())

    rows: list[dict] = []
    modes = ["baseline_v35", "major_notrade_minor_defensive", "major_notrade_minor_q3"]
    if args.mode:
        modes = [args.mode]
    for mode in modes:
        out_dir = OUT_DIR / mode
        summary_path = out_dir / "summary.json"
        if summary_path.exists():
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            rows.append(payload)
            continue
        phase_map = transformed_phase_map(phase_df, mode)
        result = run_dataset(
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
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "mode": mode,
            "major_crash_rule": "no_trade",
            "minor_crash_rule": "q2_defensive" if mode != "major_notrade_minor_q3" else "q3_post_high_vol",
            "major_crash_definition": "phase=crash and (ret5<=-0.08 or dd20<=-0.14)",
            **{k: v for k, v in result.items() if k not in {"trade_log", "equity_curve"}},
        }
        (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        result["trade_log"].to_csv(out_dir / "trade_log.csv", index=False, encoding="utf-8-sig")
        result["equity_curve"].to_csv(out_dir / "equity_curve.csv", index=False, encoding="utf-8-sig")
        rows.append(payload)

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        summary_df.to_csv(OUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")

    major_minor_days = phase_df.loc[phase_df["crash_scale"] != "", ["Date", "phase_name", "ret5", "dd20", "vol10", "crash_scale"]].copy()
    major_minor_days["Date"] = major_minor_days["Date"].dt.strftime("%Y-%m-%d")
    major_minor_days.to_csv(OUT_DIR / "crash_scale_days.csv", index=False, encoding="utf-8-sig")
    if not summary_df.empty:
        print(summary_df[["mode", "total_return_pct", "num_buys", "max_drawdown_pct"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

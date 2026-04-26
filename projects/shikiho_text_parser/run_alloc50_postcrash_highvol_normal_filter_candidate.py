from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.shikiho_text_parser import backtest_phase_adaptive_practical_v35_alloc50_candidate as mod


def recent_lagged_phases(date: pd.Timestamp, phase_map: pd.Series, phase_shift_days: int, lookback_days: int) -> list[str]:
    hist = mod.lagged_phase_history(date, phase_map, phase_shift_days)
    if hist.empty:
        return []
    return [str(x) for x in hist.tail(lookback_days).tolist()]


def build_earnings_map(spec: mod.TableSpec) -> dict[str, list[pd.Timestamp]]:
    tickers = mod.collect_relevant_tickers(spec)
    frames: list[pd.DataFrame] = []
    local_df = mod.load_local_disclosure_dates(tickers)
    if not local_df.empty:
        frames.append(local_df[["ticker", "earnings_date"]])
    if mod.IRBANK_EARNINGS_CACHE_CSV.exists():
        irbank_df = pd.read_csv(mod.IRBANK_EARNINGS_CACHE_CSV, usecols=["ticker", "earnings_date"])
        irbank_df = irbank_df[irbank_df["ticker"].astype(str).isin(set(tickers))]
        irbank_df["earnings_date"] = pd.to_datetime(irbank_df["earnings_date"], errors="coerce")
        irbank_df = irbank_df.dropna(subset=["earnings_date"])
        if not irbank_df.empty:
            frames.append(irbank_df)
    if mod.EARNINGS_CACHE_CSV.exists():
        yf_df = pd.read_csv(mod.EARNINGS_CACHE_CSV, usecols=["ticker", "earnings_date"])
        yf_df = yf_df[yf_df["ticker"].astype(str).isin(set(tickers))]
        yf_df["earnings_date"] = pd.to_datetime(yf_df["earnings_date"], errors="coerce")
        yf_df = yf_df.dropna(subset=["earnings_date"])
        if not yf_df.empty:
            frames.append(yf_df)
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["ticker", "earnings_date"])
    if not merged.empty:
        merged = merged.drop_duplicates(subset=["ticker", "earnings_date"]).sort_values(["ticker", "earnings_date"])
    return mod.build_earnings_map(merged)


def run_candidate(
    spec: mod.TableSpec,
    phase_map: pd.Series,
    earnings_map: dict[str, list[pd.Timestamp]],
    *,
    lookback_days: int = 10,
) -> dict:
    orig_projected = mod.projected_phase_name

    def custom_projected(date: pd.Timestamp, phase_map_inner: pd.Series, phase_shift_days: int, phase_proxy_mode: str) -> str:
        base = orig_projected(date, phase_map_inner, phase_shift_days, phase_proxy_mode)
        if base != "normal":
            return base
        recent = recent_lagged_phases(date, phase_map_inner, phase_shift_days, lookback_days)
        if "crash" in recent and "high_vol" in recent:
            return "downtrend"
        return base

    mod.projected_phase_name = custom_projected
    try:
        result = mod.run_dataset(
            spec,
            phase_map,
            earnings_map,
            earnings_pre_days=1,
            earnings_post_days=5,
            phase_shift_days=1,
            phase_proxy_mode="difficult_v11",
            rebuild_interval_trading_days=10,
        )
    finally:
        mod.projected_phase_name = orig_projected
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run alloc50 candidate with normal_after_postcrash_highvol treated as no_trade.")
    parser.add_argument("--dataset", required=True, choices=["q2_2024", "q3"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lookback-days", type=int, default=10)
    args = parser.parse_args()

    if args.dataset == "q2_2024":
        spec = mod.TableSpec(
            "q2_2024",
            ROOT / "projects" / "quarterly_ranker" / "output" / "q2_2024_pre_analysis_20240630_aligned" / "q2_2024_pre_shikiho_feature_ranking.csv",
            ROOT / "projects" / "quarterly_ranker" / "output" / "q2_2024_pre_analysis_20240630_aligned" / "operational" / "q2_2024_pre_selected_candidates_operational.csv",
            "2024-07-01",
            "2024-09-30",
            3_000_000.0,
        )
    else:
        spec = mod.TableSpec(
            "q3",
            ROOT / "projects" / "quarterly_ranker" / "output" / "q3_pre_analysis_20250630_aligned" / "q3_pre_shikiho_feature_ranking.csv",
            ROOT / "projects" / "quarterly_ranker" / "output" / "q3_pre_analysis_20250630_aligned" / "threshold_search_post_high_vol" / "operational" / "best_selected_candidates_operational.csv",
            "2025-07-01",
            "2025-09-30",
            3_000_000.0,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    phase_map = mod.load_phase_map(mod.PHASE_CSV)
    earnings_map = build_earnings_map(spec)
    result = run_candidate(spec, phase_map, earnings_map, lookback_days=args.lookback_days)

    payload = {k: v for k, v in result.items() if k not in {"trade_log", "equity_curve"}}
    payload["variant"] = "alloc50_normal_after_postcrash_highvol_no_trade"
    payload["lookback_days"] = args.lookback_days

    (args.output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    result["trade_log"].to_csv(args.output_dir / "trade_log.csv", index=False, encoding="utf-8-sig")
    result["equity_curve"].to_csv(args.output_dir / "equity_curve.csv", index=False, encoding="utf-8-sig")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

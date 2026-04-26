from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.shikiho_text_parser import backtest_phase_adaptive_practical_v35_no_crash_entry_candidate as mod


OUT_DIR = ROOT / "projects" / "shikiho_text_parser" / "output" / "q2_2024_no_crash_improvement_compare"


def load_cached_earnings_only(tickers: list[str]) -> pd.DataFrame:
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
    if not frames:
        return pd.DataFrame(columns=["ticker", "earnings_date"])
    merged = pd.concat(frames, ignore_index=True)
    return merged.drop_duplicates(subset=["ticker", "earnings_date"]).sort_values(["ticker", "earnings_date"])


def recent_lagged_phases(date: pd.Timestamp, phase_map: pd.Series, phase_shift_days: int, lookback_days: int) -> list[str]:
    hist = mod.lagged_phase_history(date, phase_map, phase_shift_days)
    if hist.empty:
        return []
    return [str(x) for x in hist.tail(lookback_days).tolist()]


def run_variant(spec: mod.TableSpec, phase_map: pd.Series, earnings_map: dict[str, list[pd.Timestamp]], name: str) -> dict:
    orig_mapping_fn = mod.practical_v2_mapping
    orig_projected = mod.projected_phase_name

    def custom_mapping() -> dict[str, str]:
        mapping = orig_mapping_fn()
        if name == "cap_end_no_trade":
            mapping["capitulation_end"] = "downtrend"
        elif name == "cap_end_q3":
            mapping["capitulation_end"] = "q3_post_high_vol"
        return mapping

    def custom_projected(date: pd.Timestamp, phase_map_inner: pd.Series, phase_shift_days: int, phase_proxy_mode: str) -> str:
        base = orig_projected(date, phase_map_inner, phase_shift_days, phase_proxy_mode)
        recent = recent_lagged_phases(date, phase_map_inner, phase_shift_days, 5)
        recent10 = recent_lagged_phases(date, phase_map_inner, phase_shift_days, 10)
        post_crash_normal = base == "normal" and any(x in {"crash", "capitulation_end"} for x in recent)
        post_high_vol_normal = base == "normal" and any(x in {"high_vol", "capitulation_end"} for x in recent10)
        if name == "normal_after_crash_no_trade" and post_crash_normal:
            return "downtrend"
        if name == "normal_after_crash_q3" and post_crash_normal:
            return "settling"
        if name == "normal_after_highvol10_no_trade" and post_high_vol_normal:
            return "downtrend"
        if name == "normal_after_highvol10_q3" and post_high_vol_normal:
            return "settling"
        if name == "combo_no_trade":
            if base == "capitulation_end":
                return "downtrend"
            if post_crash_normal:
                return "downtrend"
        if name == "combo_highvol10_no_trade":
            if base == "capitulation_end":
                return "downtrend"
            if post_high_vol_normal:
                return "downtrend"
        if name == "combo_q3":
            if base == "capitulation_end":
                return "settling"
            if post_crash_normal:
                return "settling"
        return base

    mod.practical_v2_mapping = custom_mapping
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
        mod.practical_v2_mapping = orig_mapping_fn
        mod.projected_phase_name = orig_projected
    return result


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    spec = mod.TableSpec(
        "q2_2024",
        ROOT / "projects" / "quarterly_ranker" / "output" / "q2_2024_pre_analysis_20240630_aligned" / "q2_2024_pre_shikiho_feature_ranking.csv",
        ROOT / "projects" / "quarterly_ranker" / "output" / "q2_2024_pre_analysis_20240630_aligned" / "operational" / "q2_2024_pre_selected_candidates_operational.csv",
        "2024-07-01",
        "2024-09-30",
        3_000_000.0,
    )
    phase_map = mod.load_phase_map(mod.PHASE_CSV)
    tickers = mod.collect_relevant_tickers(spec)
    earnings_map = mod.build_earnings_map(load_cached_earnings_only(tickers))

    variants = [
        "baseline_no_crash",
        "cap_end_no_trade",
        "cap_end_q3",
        "normal_after_crash_no_trade",
        "normal_after_crash_q3",
        "normal_after_highvol10_no_trade",
        "normal_after_highvol10_q3",
        "combo_no_trade",
        "combo_highvol10_no_trade",
        "combo_q3",
    ]

    rows = []
    for name in variants:
        if name == "baseline_no_crash":
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
        else:
            result = run_variant(spec, phase_map, earnings_map, name)
        payload = {k: v for k, v in result.items() if k not in {"trade_log", "equity_curve"}}
        payload["variant"] = name
        rows.append(payload)
        out = OUT_DIR / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        result["trade_log"].to_csv(out / "trade_log.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(rows).to_csv(OUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

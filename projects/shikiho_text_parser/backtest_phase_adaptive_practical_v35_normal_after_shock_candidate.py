from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.shikiho_text_parser import backtest_phase_adaptive_practical_v34_rebuild_interval_candidate as mod


def recent_lagged_phases(date: pd.Timestamp, phase_map: pd.Series, phase_shift_days: int, lookback_days: int) -> list[str]:
    hist = mod.lagged_phase_history(date, phase_map, phase_shift_days)
    if hist.empty:
        return []
    return [str(x) for x in hist.tail(lookback_days).tolist()]


def run_dataset_with_normal_after_shock(
    spec: mod.TableSpec,
    phase_map: pd.Series,
    earnings_map: dict[str, list[pd.Timestamp]],
    *,
    lookback_days: int,
    shock_phases: set[str],
    earnings_pre_days: int,
    earnings_post_days: int,
    phase_shift_days: int,
    phase_proxy_mode: str,
    rebuild_interval_trading_days: int,
) -> dict:
    orig_projected = mod.projected_phase_name

    def custom_projected(
        date: pd.Timestamp,
        phase_map_inner: pd.Series,
        phase_shift_days_inner: int,
        phase_proxy_mode_inner: str,
    ) -> str:
        base = orig_projected(date, phase_map_inner, phase_shift_days_inner, phase_proxy_mode_inner)
        recent = recent_lagged_phases(date, phase_map_inner, phase_shift_days_inner, lookback_days)
        if base == "normal" and any(x in shock_phases for x in recent):
            return "downtrend"
        return base

    mod.projected_phase_name = custom_projected
    try:
        result = mod.run_dataset(
            spec,
            phase_map,
            earnings_map,
            earnings_pre_days=earnings_pre_days,
            earnings_post_days=earnings_post_days,
            phase_shift_days=phase_shift_days,
            phase_proxy_mode=phase_proxy_mode,
            rebuild_interval_trading_days=rebuild_interval_trading_days,
        )
    finally:
        mod.projected_phase_name = orig_projected
    return result


def build_earnings_map_for_specs(specs: list[mod.TableSpec]) -> dict[str, list[pd.Timestamp]]:
    all_tickers: set[str] = set()
    for spec in specs:
        all_tickers.update(mod.collect_relevant_tickers(spec))

    frames: list[pd.DataFrame] = []
    local_df = mod.load_local_disclosure_dates(sorted(all_tickers))
    if not local_df.empty:
        frames.append(local_df[["ticker", "earnings_date"]])
    if mod.IRBANK_EARNINGS_CACHE_CSV.exists():
        irbank_df = pd.read_csv(mod.IRBANK_EARNINGS_CACHE_CSV, usecols=["ticker", "earnings_date"])
        irbank_df = irbank_df[irbank_df["ticker"].astype(str).isin(all_tickers)]
        irbank_df["earnings_date"] = pd.to_datetime(irbank_df["earnings_date"], errors="coerce")
        irbank_df = irbank_df.dropna(subset=["earnings_date"])
        if not irbank_df.empty:
            frames.append(irbank_df)
    if mod.EARNINGS_CACHE_CSV.exists():
        yf_df = pd.read_csv(mod.EARNINGS_CACHE_CSV, usecols=["ticker", "earnings_date"])
        yf_df = yf_df[yf_df["ticker"].astype(str).isin(all_tickers)]
        yf_df["earnings_date"] = pd.to_datetime(yf_df["earnings_date"], errors="coerce")
        yf_df = yf_df.dropna(subset=["earnings_date"])
        if not yf_df.empty:
            frames.append(yf_df)

    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["ticker", "earnings_date"])
    if not merged.empty:
        merged = merged.drop_duplicates(subset=["ticker", "earnings_date"]).sort_values(["ticker", "earnings_date"])
    return mod.build_earnings_map(merged)


def save_run(output_dir: Path, name: str, result: dict) -> dict:
    out_dir = output_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in result.items() if k not in {"trade_log", "equity_curve"}}
    payload["variant"] = "normal_after_shock_candidate"
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    result["trade_log"].to_csv(out_dir / "trade_log.csv", index=False, encoding="utf-8-sig")
    result["equity_curve"].to_csv(out_dir / "equity_curve.csv", index=False, encoding="utf-8-sig")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v3.5 candidate with normal_after_shock treated as no_trade.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lookback-days", type=int, default=10)
    parser.add_argument("--earnings-pre-days", type=int, default=1)
    parser.add_argument("--earnings-post-days", type=int, default=5)
    parser.add_argument("--phase-shift-days", type=int, default=1)
    parser.add_argument("--phase-proxy-mode", type=str, default="difficult_v11")
    parser.add_argument("--rebuild-interval-trading-days", type=int, default=10)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--detail-csv", type=Path)
    parser.add_argument("--selected-csv", type=Path)
    parser.add_argument("--start-date", type=str)
    parser.add_argument("--end-date", type=str)
    parser.add_argument("--dataset-name", type=str)
    parser.add_argument("--initial-capital", type=float, default=3_000_000.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    phase_map = mod.load_phase_map(mod.PHASE_CSV)

    if args.batch:
        specs = mod.batch_specs()
    else:
        specs = [
            mod.TableSpec(
                args.dataset_name,
                args.detail_csv,
                args.selected_csv,
                args.start_date,
                args.end_date,
                args.initial_capital,
            )
        ]

    earnings_map = build_earnings_map_for_specs(specs)
    rows: list[dict] = []
    shock_phases = {"crash", "high_vol", "capitulation_end"}

    for spec in specs:
        result = run_dataset_with_normal_after_shock(
            spec,
            phase_map,
            earnings_map,
            lookback_days=args.lookback_days,
            shock_phases=shock_phases,
            earnings_pre_days=args.earnings_pre_days,
            earnings_post_days=args.earnings_post_days,
            phase_shift_days=args.phase_shift_days,
            phase_proxy_mode=args.phase_proxy_mode,
            rebuild_interval_trading_days=args.rebuild_interval_trading_days,
        )
        rows.append(save_run(args.output_dir, spec.name, result))

    pd.DataFrame(rows).to_csv(args.output_dir / "summary_all.csv", index=False, encoding="utf-8-sig")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

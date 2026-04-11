from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_v38_signal
from src.stock_signal import download_index_returns


def refresh_sp500_cache(period: str) -> bool:
    try:
        returns = download_index_returns(period)
    except Exception as exc:
        print(f"[SP500_REFRESH_FAILED] {type(exc).__name__}: {exc}")
        return False
    if returns.empty or "ret_sp500" not in returns.columns:
        print("[SP500_REFRESH_FAILED] ret_sp500 column not available")
        return False

    new_df = returns[["ret_sp500"]].dropna().reset_index()
    new_df = new_df.rename(columns={new_df.columns[0]: "Date"})
    new_df["Date"] = pd.to_datetime(new_df["Date"], errors="coerce")
    new_df = new_df.dropna(subset=["Date"])
    if new_df.empty:
        print("[SP500_REFRESH_FAILED] no usable ret_sp500 rows")
        return False

    path = run_v38_signal.SP500_RETURNS_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old_df = pd.read_csv(path)
        old_date_col = "Date" if "Date" in old_df.columns else old_df.columns[0]
        old_val_col = "ret_sp500" if "ret_sp500" in old_df.columns else old_df.columns[-1]
        old_df = old_df[[old_date_col, old_val_col]].rename(columns={old_date_col: "Date", old_val_col: "ret_sp500"})
        old_df["Date"] = pd.to_datetime(old_df["Date"], errors="coerce")
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined = (
        combined.dropna(subset=["Date", "ret_sp500"])
        .sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
    )
    combined.to_csv(path, index=False, encoding="utf-8-sig")
    latest = combined.iloc[-1]
    print(f"[SP500_REFRESH] rows={len(combined)} latest_date={pd.Timestamp(latest['Date']).date().isoformat()} ret_sp500={float(latest['ret_sp500'])}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate v3.8 rebound daytrade-only signal log.")
    parser.add_argument("--dataset", choices=list(run_v38_signal.DATASET_SPECS.keys()), default=run_v38_signal.DEFAULT_DATASET)
    parser.add_argument("--skip-refresh", action="store_true")
    parser.add_argument("--skip-sp500-refresh", action="store_true")
    parser.add_argument("--sp500-period", default="6mo")
    parser.add_argument("--precheck", action="store_true", help="Emit prior-night provisional candidates without refreshing or requiring S&P500.")
    parser.add_argument("--max-stale-count", type=int, default=25)
    parser.add_argument("--max-selected-stale-count", type=int, default=0)
    parser.add_argument("--max-selected-lag-bdays", type=int, default=0)
    args = parser.parse_args()

    if args.precheck:
        print("[SP500_PRECHECK_SKIPPED] prior-night provisional candidates use only pre-trade local data")
    elif args.skip_sp500_refresh:
        print("[SP500_REFRESH_SKIPPED]")
    else:
        refresh_sp500_cache(args.sp500_period)

    argv = [
        "run_v38_signal.py",
        "--dataset",
        args.dataset,
        "--daytrade-precheck" if args.precheck else "--daytrade-only",
        "--max-stale-count",
        str(args.max_stale_count),
        "--max-selected-stale-count",
        str(args.max_selected_stale_count),
        "--max-selected-lag-bdays",
        str(args.max_selected_lag_bdays),
    ]
    if args.skip_refresh:
        argv.append("--skip-refresh")
    old_argv = sys.argv
    try:
        sys.argv = argv
        return run_v38_signal.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())

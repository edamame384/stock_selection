from __future__ import annotations

import io
import json
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yfinance as yf

from src.stock_signal import _download_yahoo_ticker

NIKKEI_CSV = ROOT / "data" / "external_market" / "nikkei225_daily.csv"
PHASE_CSV = ROOT / "projects" / "shikiho_text_parser" / "output" / "nikkei_market_phase_map" / "nikkei_market_phase_daily_labels.csv"
SSA_OUT_DIR = ROOT / "projects" / "shikiho_text_parser" / "output" / "nikkei_ssa_guard_candidate"
SSA_OUT_DIR.mkdir(parents=True, exist_ok=True)
SSA_DAILY_CSV = SSA_OUT_DIR / "nikkei_ssa_daily.csv"
SSA_SUMMARY_CSV = SSA_OUT_DIR / "ssa_guard_period_summary.csv"
SSA_SUMMARY_JSON = SSA_OUT_DIR / "summary.json"

W = 40
L = 20
R = 1


def download_nikkei() -> pd.DataFrame:
    last_error: Exception | None = None
    try:
        df = _download_yahoo_ticker("^N225", period="max", retries=5)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [str(col[0]) for col in df.columns]
        df.index.name = "Date"
        return df
    except Exception as exc:
        last_error = exc

    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        df = yf.download("^N225", period="max", auto_adjust=False, progress=False, threads=False)
    if not df.empty:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [str(col[0]) for col in df.columns]
        df.index.name = "Date"
        return df

    if NIKKEI_CSV.exists():
        cached = pd.read_csv(NIKKEI_CSV, index_col=0, parse_dates=True)
        if not cached.empty:
            cached.index.name = "Date"
            print(f"[NIKKEI] warning: Yahoo download failed, using cached CSV. reason={last_error}", file=sys.stderr)
            return cached

    raise RuntimeError(f"No Nikkei 225 data downloaded from Yahoo Finance. reason={last_error}")


def rolling_ssa_last(window: np.ndarray, l: int = L, r: int = R) -> float:
    n = len(window)
    k = n - l + 1
    hankel = np.column_stack([window[i : i + k] for i in range(l)])
    u, s, vt = np.linalg.svd(hankel, full_matrices=False)
    recon = np.zeros_like(hankel)
    for i in range(r):
        recon += s[i] * np.outer(u[:, i], vt[i, :])
    last_vals = []
    for row in range(k):
        col = (n - 1) - row
        if 0 <= col < l:
            last_vals.append(recon[row, col])
    return float(np.mean(last_vals)) if last_vals else float("nan")


def zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0).replace(0.0, np.nan)
    return (series - mean) / std


def build_ssa_daily(phase_df: pd.DataFrame) -> pd.DataFrame:
    x = phase_df.copy().sort_values("Date").reset_index(drop=True)
    closes = x["Close"].astype(float).to_numpy()
    ssa = np.full(len(x), np.nan, dtype=float)
    for i in range(W - 1, len(x)):
        ssa[i] = rolling_ssa_last(closes[i - W + 1 : i + 1], l=L, r=R)
    x["ssa_40"] = ssa
    x["price_to_ssa_gap"] = x["Close"] / x["ssa_40"] - 1.0
    x["ssa_slope_3"] = x["ssa_40"] / x["ssa_40"].shift(3) - 1.0
    x["ssa_slope_5"] = x["ssa_40"] / x["ssa_40"].shift(5) - 1.0
    x["ssa_curvature"] = x["ssa_slope_3"] - x["ssa_slope_3"].shift(3)
    x["ssa_gap_z20"] = zscore(x["price_to_ssa_gap"], 20)
    x["ssa_gap_z40"] = zscore(x["price_to_ssa_gap"], 40)
    x["ssa_reclaim"] = x["Close"] > x["ssa_40"]

    streak = 0
    reclaim_streak = []
    for flag in x["ssa_reclaim"].fillna(False):
        streak = streak + 1 if flag else 0
        reclaim_streak.append(streak)
    x["ssa_reclaim_streak"] = reclaim_streak

    recovery_tests = pd.DataFrame(
        {
            "reclaim": x["Close"] > x["ssa_40"],
            "slope3": x["ssa_slope_3"] > 0.0,
            "curvature": x["ssa_curvature"] > 0.0,
            "gap_z20": x["ssa_gap_z20"] > -0.5,
            "streak": x["ssa_reclaim_streak"] >= 2,
        }
    ).fillna(False)
    recovery_strong_tests = pd.DataFrame(
        {
            "reclaim": x["Close"] > x["ssa_40"],
            "slope3": x["ssa_slope_3"] > 0.005,
            "slope5": x["ssa_slope_5"] > 0.0,
            "curvature": x["ssa_curvature"] > 0.0,
            "streak": x["ssa_reclaim_streak"] >= 3,
        }
    ).fillna(False)
    x["ssa_recovery_confirm"] = recovery_tests.sum(axis=1) >= 3
    x["ssa_recovery_strong"] = recovery_strong_tests.sum(axis=1) >= 4
    return x


def refresh_ssa_outputs() -> None:
    if not PHASE_CSV.exists():
        raise RuntimeError(f"Phase CSV not found: {PHASE_CSV}")
    phase_df = pd.read_csv(PHASE_CSV)
    phase_df["Date"] = pd.to_datetime(phase_df["Date"], errors="coerce")
    enriched = build_ssa_daily(phase_df)
    enriched.to_csv(SSA_DAILY_CSV, index=False, encoding="utf-8-sig")

    summary_rows = [
        {
            "latest_date": enriched.iloc[-1]["Date"].strftime("%Y-%m-%d"),
            "latest_close": float(enriched.iloc[-1]["Close"]),
            "ssa_recovery_confirm": bool(enriched.iloc[-1]["ssa_recovery_confirm"]),
            "ssa_recovery_strong": bool(enriched.iloc[-1]["ssa_recovery_strong"]),
        }
    ]
    pd.DataFrame(summary_rows).to_csv(SSA_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    with open(SSA_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump({"rows": summary_rows, "params": {"W": W, "L": L, "R": R}}, f, ensure_ascii=False, indent=2)
    print(f"[SSA] saved {SSA_DAILY_CSV} latest={summary_rows[0]['latest_date']} rows={len(enriched)}")


def main() -> int:
    try:
        NIKKEI_CSV.parent.mkdir(parents=True, exist_ok=True)
        df = download_nikkei()
        df.to_csv(NIKKEI_CSV)
        print(f"[NIKKEI] saved {NIKKEI_CSV} latest={df.index.max().date().isoformat()} rows={len(df)}")

        proc = subprocess.run(
            [sys.executable, "projects/shikiho_text_parser/classify_nikkei_market_phases.py"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        if proc.stdout:
            print(proc.stdout.rstrip())
        if proc.stderr:
            print(proc.stderr.rstrip(), file=sys.stderr)
        if proc.returncode != 0:
            raise RuntimeError("Command failed: classify_nikkei_market_phases.py")

        refresh_ssa_outputs()
        return 0
    except Exception as exc:
        print(f"[NIKKEI_UPDATE_WARNING] {exc}", file=sys.stderr)
        if NIKKEI_CSV.exists() and PHASE_CSV.exists() and SSA_DAILY_CSV.exists():
            print(
                "[NIKKEI_UPDATE_FALLBACK] Using existing Nikkei / phase / SSA files because refresh failed.",
                file=sys.stderr,
            )
            return 0
        raise


if __name__ == "__main__":
    raise SystemExit(main())

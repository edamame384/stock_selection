from __future__ import annotations

import io
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]
NIKKEI_CSV = ROOT / "data" / "external_market" / "nikkei225_daily.csv"


def download_nikkei() -> pd.DataFrame:
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        df = yf.download("^N225", period="max", auto_adjust=False, progress=False, threads=False)
    if df.empty:
        raise RuntimeError("No Nikkei 225 data downloaded from Yahoo Finance.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(col[0]) for col in df.columns]
    df.index.name = "Date"
    return df


def main() -> int:
    NIKKEI_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = download_nikkei()
    df.to_csv(NIKKEI_CSV)
    print(f"[NIKKEI] saved {NIKKEI_CSV} latest={df.index.max().date().isoformat()} rows={len(df)}")

    commands = [
        [sys.executable, "projects/shikiho_text_parser/classify_nikkei_market_phases.py"],
        [sys.executable, "projects/shikiho_text_parser/analyze_nikkei_ssa_guard_candidate.py"],
    ]
    for command in commands:
        proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        if proc.stdout:
            print(proc.stdout.rstrip())
        if proc.stderr:
            print(proc.stderr.rstrip(), file=sys.stderr)
        if proc.returncode != 0:
            raise RuntimeError(f"Command failed: {' '.join(command)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

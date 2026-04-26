from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "projects" / "shikiho_text_parser" / "backtest_phase_adaptive_practical_v31_pre1_post5_except_crash.py"
OUT_DIR = ROOT / "projects" / "shikiho_text_parser" / "output" / "causal_phase_proxy_compare"
MODES = [
    "lagged",
    "crash_to_high_vol",
    "crash_to_high_vol_surge_to_normal",
    "difficult_v1",
    "difficult_v2",
    "difficult_v3",
]


def run_mode(mode: str) -> Path:
    out_dir = OUT_DIR / mode
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--batch",
        "--output-dir",
        str(out_dir),
        "--earnings-pre-days",
        "1",
        "--earnings-post-days",
        "5",
        "--phase-shift-days",
        "1",
        "--phase-proxy-mode",
        mode,
    ]
    subprocess.check_call(cmd)
    return out_dir / "summary_all.csv"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[pd.DataFrame] = []
    for mode in MODES:
        summary_path = run_mode(mode)
        df = pd.read_csv(summary_path)
        df["mode"] = mode
        rows.append(df)

    all_df = pd.concat(rows, ignore_index=True)
    all_df.to_csv(OUT_DIR / "detail.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for mode, g in all_df.groupby("mode", sort=False):
        total_final = float(g["final_capital"].sum())
        total_initial = 3_000_000.0 * len(g)
        summary_rows.append(
            {
                "mode": mode,
                "total_return_pct": (total_final / total_initial - 1.0) * 100.0,
                "total_final_capital": total_final,
                "q2_2024_return_pct": float(g.loc[g["dataset"] == "q2_2024", "total_return_pct"].iloc[0]),
                "q3_return_pct": float(g.loc[g["dataset"] == "q3", "total_return_pct"].iloc[0]),
                "q4_return_pct": float(g.loc[g["dataset"] == "q4", "total_return_pct"].iloc[0]),
                "4q2_return_pct": float(g.loc[g["dataset"] == "4q2", "total_return_pct"].iloc[0]),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("total_return_pct", ascending=False).reset_index(drop=True)
    summary_df.to_csv(OUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

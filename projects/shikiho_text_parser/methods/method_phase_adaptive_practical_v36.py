from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BACKTEST = ROOT / "projects" / "shikiho_text_parser" / "compare_alloc50_post_major_multi_etf_expanded_candidate.py"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run practical phase-adaptive v3.6 (alloc50 + post-major-crash multi-ETF concentrated switch).")
    parser.add_argument("--datasets", nargs="*", default=None)
    args = parser.parse_args()

    cmd = [sys.executable, str(BACKTEST)]
    if args.datasets:
        cmd.extend(["--datasets", *args.datasets])
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT_DIR, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full shikiho_text_parser pipeline.")
    parser.add_argument("--skip-bundle", action="store_true")
    args = parser.parse_args()

    if not args.skip_bundle:
        run([sys.executable, "projects/shikiho_text_parser/prepare_github_bundle.py"])
    run(
        [
            sys.executable,
            "projects/shikiho_text_parser/parse_shikiho_text.py",
            "--input",
            "projects/shikiho_text_parser/data/raw/4Q-2",
            "--output-dir",
            "projects/shikiho_text_parser/output",
        ]
    )
    run([sys.executable, "projects/shikiho_text_parser/select_promising_4q2.py"])
    run(
        [
            sys.executable,
            "projects/shikiho_text_parser/backtest_4q2_signals.py",
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    parser = argparse.ArgumentParser(description='Run named method: etf_dedicated_rebound')
    parser.add_argument('--ticker', type=str, required=True)
    parser.add_argument('--name', type=str, default='ETF')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--start-date', type=str, required=True)
    parser.add_argument('--end-date', type=str, required=True)
    parser.add_argument('--initial-capital', type=float, default=3_000_000.0)
    parser.add_argument('--signal-name', type=str, default='rebound_open')
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(ROOT / 'projects' / 'shikiho_text_parser' / 'backtest_etf_dedicated_signal.py'),
        '--ticker', args.ticker,
        '--name', args.name,
        '--output-dir', str(args.output_dir),
        '--start-date', args.start_date,
        '--end-date', args.end_date,
        '--initial-capital', str(args.initial_capital),
        '--signal-name', args.signal_name,
    ]
    return subprocess.call(cmd)


if __name__ == '__main__':
    raise SystemExit(main())

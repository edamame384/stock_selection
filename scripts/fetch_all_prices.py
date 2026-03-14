from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.stock_signal import download_daily_data, normalize_symbol


def load_symbols(path: Path) -> list[str]:
    df = pd.read_csv(path)
    if "symbol" not in df.columns:
        raise ValueError("source csv must include 'symbol' column")
    symbols = []
    seen: set[str] = set()
    for raw in df["symbol"].dropna().tolist():
        sym = normalize_symbol(str(raw))
        if sym in seen:
            continue
        seen.add(sym)
        symbols.append(sym)
    return symbols


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch price history for all symbols in a csv.")
    parser.add_argument("--source", type=Path, default=Path("data/sector_master_template.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/prices"))
    parser.add_argument("--period", type=str, default="max")
    parser.add_argument("--refresh", action="store_true", help="Refetch even when local csv exists.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for partial runs.")
    parser.add_argument("--status-out", type=Path, default=Path("data/fetch_all_prices_status.csv"))
    args = parser.parse_args()

    symbols = load_symbols(args.source)
    if args.limit > 0:
        symbols = symbols[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    total = len(symbols)

    for idx, sym in enumerate(symbols, start=1):
        out_path = args.out_dir / f"{sym.replace('.', '_')}.csv"
        if out_path.exists() and not args.refresh:
            rows.append({"symbol": sym, "status": "skipped_exists", "rows": None, "path": str(out_path)})
            print(f"[{idx}/{total}] SKIP {sym}")
            continue

        try:
            df = download_daily_data(sym, period=args.period)
            if df.empty:
                raise ValueError("empty dataframe")
            df = df.sort_index()
            df.to_csv(out_path)
            rows.append({"symbol": sym, "status": "ok", "rows": int(len(df)), "path": str(out_path)})
            print(f"[{idx}/{total}] OK {sym} rows={len(df)}")
        except Exception as exc:
            rows.append({"symbol": sym, "status": "error", "rows": None, "path": "", "error": str(exc)})
            print(f"[{idx}/{total}] ERROR {sym}: {exc}")

        if idx % 100 == 0 or idx == total:
            pd.DataFrame(rows).to_csv(args.status_out, index=False)

    pd.DataFrame(rows).to_csv(args.status_out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

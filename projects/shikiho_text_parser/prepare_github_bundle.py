from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from projects.shikiho_text_parser.parse_shikiho_text import collect_input_files
from projects.shikiho_text_parser.paths import PRICE_DIR, PRICE_FULL_DIR, RAW_4Q2_DIR, REFERENCE_DIR, SECTOR_MASTER_PATH


DEFAULT_EXTERNAL_4Q2_DIR = Path(r"C:\Users\mitsu\OneDrive\ドキュメント\四季報DB2025\4Q-2")
DEFAULT_GLOBAL_PRICE_DIR = ROOT_DIR / "data" / "prices"
DEFAULT_GLOBAL_SECTOR_MASTER = ROOT_DIR / "data" / "sector_master_template.csv"


def ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_tree_txt_only(src_dir: Path, dst_dir: Path) -> int:
    ensure_clean_dir(dst_dir)
    count = 0
    for src in collect_input_files(src_dir):
        shutil.copy2(src, dst_dir / src.name)
        count += 1
    return count


def infer_tickers(raw_dir: Path) -> list[str]:
    tickers = []
    for txt in collect_input_files(raw_dir):
        code = txt.stem.strip()
        if code:
            tickers.append(f"{code}.T")
    return sorted(set(tickers))


def copy_price_subset(tickers: list[str], src_dir: Path, dst_dir: Path) -> tuple[int, int]:
    ensure_clean_dir(dst_dir)
    copied = 0
    missing = 0
    for ticker in tickers:
        src = src_dir / f"{ticker.replace('.', '_')}.csv"
        if src.exists():
            shutil.copy2(src, dst_dir / src.name)
            copied += 1
        else:
            missing += 1
    return copied, missing


def copy_price_full(src_dir: Path, dst_dir: Path) -> int:
    ensure_clean_dir(dst_dir)
    count = 0
    for src in sorted(src_dir.glob("*.csv")):
        shutil.copy2(src, dst_dir / src.name)
        count += 1
    return count


def write_inventory(manifest: dict, out_dir: Path) -> None:
    lines = [
        "# Data Inventory",
        "",
        "## Included",
        f"- raw 4Q-2 text: {manifest['raw_4q2_files']} files",
        f"- subset price CSV for 4Q-2 universe: {manifest['price_files']} files",
        f"- full price CSV archive: {manifest['full_price_files']} files",
        f"- sector master rows: {manifest['sector_master_rows']}",
        "",
        "## Not Included as actual datasets",
        "- night futures / overnight futures CSV: not found in local workspace",
        "- Dow/S&P500/Nikkei futures time series CSV: not found in local workspace",
        "",
        "## Reference only",
        "- futures-related run logs if present",
        "",
        "## Reproducibility note",
        "- this project can reproduce the current 4Q-2 text parsing, selection, and condition2 backtest without downloading new data",
        "- futures-based strategies cannot be reproduced from this bundle because the underlying futures dataset is not present locally",
    ]
    (out_dir / "data_inventory.md").write_text("\n".join(lines), encoding="utf-8")

    pd.DataFrame(
        [
            {"group": "included", "name": "raw_4q2_text", "path": str(RAW_4Q2_DIR), "count": manifest["raw_4q2_files"], "note": "4Q-2 text inputs"},
            {"group": "included", "name": "prices_4q2", "path": str(PRICE_DIR), "count": manifest["price_files"], "note": "subset price data for 4Q-2 universe"},
            {"group": "included", "name": "prices_full", "path": str(PRICE_FULL_DIR), "count": manifest["full_price_files"], "note": "full local price archive"},
            {"group": "included", "name": "sector_master", "path": str(SECTOR_MASTER_PATH), "count": manifest["sector_master_rows"], "note": "formal 33-sector master"},
            {"group": "missing", "name": "night_futures_data", "path": "", "count": 0, "note": "actual futures dataset not found; only logs exist"},
            {"group": "missing", "name": "global_index_futures_data", "path": "", "count": 0, "note": "DJI/GSPC/NKD futures csv not found"},
        ]
    ).to_csv(out_dir / "data_inventory.csv", index=False, encoding="utf-8-sig")


def copy_sector_master(src: Path, dst: Path) -> int:
    ensure_clean_dir(dst.parent)
    shutil.copy2(src, dst)
    df = pd.read_csv(dst)
    return int(len(df))


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a shareable GitHub bundle for shikiho_text_parser.")
    parser.add_argument("--external-4q2-dir", type=Path, default=DEFAULT_EXTERNAL_4Q2_DIR)
    parser.add_argument("--global-price-dir", type=Path, default=DEFAULT_GLOBAL_PRICE_DIR)
    parser.add_argument("--global-sector-master", type=Path, default=DEFAULT_GLOBAL_SECTOR_MASTER)
    args = parser.parse_args()

    raw_count = copy_tree_txt_only(args.external_4q2_dir, RAW_4Q2_DIR)
    tickers = infer_tickers(RAW_4Q2_DIR)
    price_count, price_missing = copy_price_subset(tickers, args.global_price_dir, PRICE_DIR)
    full_price_count = copy_price_full(args.global_price_dir, PRICE_FULL_DIR)
    sector_rows = copy_sector_master(args.global_sector_master, SECTOR_MASTER_PATH)

    manifest = {
        "raw_4q2_dir": str(RAW_4Q2_DIR),
        "raw_4q2_files": raw_count,
        "price_dir": str(PRICE_DIR),
        "price_files": price_count,
        "price_missing_files": price_missing,
        "price_full_dir": str(PRICE_FULL_DIR),
        "full_price_files": full_price_count,
        "sector_master": str(SECTOR_MASTER_PATH),
        "sector_master_rows": sector_rows,
    }
    ensure_clean_dir(REFERENCE_DIR)
    (REFERENCE_DIR / "bundle_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    futures_logs = [str(p) for p in sorted((ROOT_DIR / "data").glob("*futures*.log"))]
    (REFERENCE_DIR / "futures_log_manifest.json").write_text(
        json.dumps({"futures_log_files": futures_logs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_inventory(manifest, REFERENCE_DIR)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

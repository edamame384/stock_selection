from __future__ import annotations

import argparse
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
REBUILD_SCRIPT = ROOT_DIR / "projects" / "quarterly_ranker" / "rebuild_generic_quarter_image_dataset_candidate.py"


def collect_images(image_dir: Path) -> list[Path]:
    return sorted(image_dir.rglob("*.png"), key=lambda p: int("".join(ch for ch in p.stem if ch.isdigit()) or "0"))


def chunk_ranges(total: int, chunk_size: int) -> list[tuple[int, int]]:
    return [(start, min(start + chunk_size, total)) for start in range(0, total, chunk_size)]


def run_chunk(args: argparse.Namespace, start: int, end: int, chunk_csv: Path) -> tuple[int, int, int]:
    cmd = [
        sys.executable,
        str(REBUILD_SCRIPT),
        "--image-dir",
        str(args.image_dir),
        "--name-csv",
        str(args.name_csv),
        "--price-dir",
        str(args.price_dir),
        "--out-dir",
        str(args.out_dir),
        "--cutoff-date",
        args.cutoff_date,
        "--quarter-start",
        args.quarter_start,
        "--quarter-label",
        args.quarter_label,
        "--selector-style",
        args.selector_style,
        "--realized-start",
        args.realized_start,
        "--realized-end",
        args.realized_end,
        "--library-name",
        str(chunk_csv.relative_to(args.out_dir)).replace("\\", "/"),
        "--image-start",
        str(start),
        "--image-end",
        str(end),
        "--library-only",
    ]
    result = subprocess.run(cmd, cwd=ROOT_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"chunk {start}:{end} failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return start, end, result.returncode


def merge_libraries(chunk_csvs: list[Path], merged_csv: Path) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in chunk_csvs if path.exists()]
    if not frames:
        raise RuntimeError("No chunk csvs to merge.")
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["image_file"], keep="first").sort_values("image_file").reset_index(drop=True)
    if "resolved" in merged.columns:
        merged["resolved"] = merged["resolved"].astype(bool)
    merged_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(merged_csv, index=False, encoding="utf-8-sig")
    return merged


def run_finalize(args: argparse.Namespace, merged_csv: Path) -> None:
    cmd = [
        sys.executable,
        str(REBUILD_SCRIPT),
        "--image-dir",
        str(args.image_dir),
        "--name-csv",
        str(args.name_csv),
        "--price-dir",
        str(args.price_dir),
        "--out-dir",
        str(args.out_dir),
        "--cutoff-date",
        args.cutoff_date,
        "--quarter-start",
        args.quarter_start,
        "--quarter-label",
        args.quarter_label,
        "--selector-style",
        args.selector_style,
        "--realized-start",
        args.realized_start,
        "--realized-end",
        args.realized_end,
        "--library-csv",
        str(merged_csv),
    ]
    result = subprocess.run(cmd, cwd=ROOT_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"finalize failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Chunked runner for generic quarter image rebuild candidate.")
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--name-csv", type=Path, default=Path(r"c:\Users\mitsu\Downloads\四季報2026年1集 - 銘柄.csv"))
    parser.add_argument("--price-dir", type=Path, default=Path("data/prices"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cutoff-date", type=str, required=True)
    parser.add_argument("--quarter-start", type=str, required=True)
    parser.add_argument("--quarter-label", type=str, required=True)
    parser.add_argument("--selector-style", choices=["q2", "q3", "q4"], required=True)
    parser.add_argument("--realized-start", type=str, required=True)
    parser.add_argument("--realized-end", type=str, required=True)
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = args.out_dir / "library_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(args.image_dir)
    ranges = chunk_ranges(len(images), args.chunk_size)
    chunk_csvs = [chunks_dir / f"chunk_{start:05d}_{end:05d}.csv" for start, end in ranges]

    pending: list[tuple[int, int, Path]] = []
    for (start, end), chunk_csv in zip(ranges, chunk_csvs):
        if chunk_csv.exists():
            continue
        pending.append((start, end, chunk_csv))

    if pending:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(run_chunk, args, start, end, chunk_csv): (start, end, chunk_csv)
                for start, end, chunk_csv in pending
            }
            for future in as_completed(futures):
                start, end, chunk_csv = futures[future]
                future.result()
                print(f"[DONE] chunk {start}:{end} -> {chunk_csv}")

    merged_csv = args.out_dir / "image_library.csv"
    merged = merge_libraries(chunk_csvs, merged_csv)
    print(f"[INFO] merged images={len(merged)} resolved={int(merged['resolved'].sum())}")
    run_finalize(args, merged_csv)
    print(f"[DONE] finalized {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

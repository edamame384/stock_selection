from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from projects.quarterly_ranker.operational_csv_utils import write_operational_csv
from projects.quarterly_ranker.select_4q_pre_oct_candidates import (
    attach_realized_returns,
    build_4q_library,
    build_pre_oct_base,
    enrich_with_numeric_ocr,
    load_name_master,
    prepare_mapped_image_dir,
    select_candidates,
)


def add_learning_date(df: pd.DataFrame, learning_date: str) -> pd.DataFrame:
    out = df.copy()
    out["learning_date"] = learning_date
    out["training_cutoff_date"] = learning_date
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Incrementally rebuild full 4Q pre-Oct selection using an existing library as a base."
    )
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--name-csv", type=Path, required=True)
    parser.add_argument("--price-dir", type=Path, default=Path("data/prices"))
    parser.add_argument("--existing-library-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--learning-date", default="2025-09-30")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    seed_library = pd.read_csv(args.existing_library_csv)
    seed_library["image_no"] = seed_library["image_no"].astype(int)
    max_existing_image_no = int(seed_library["image_no"].max())
    library_csv = args.out_dir / "q4_image_library.csv"
    if library_csv.exists():
        existing_library = pd.read_csv(library_csv)
        existing_library["image_no"] = existing_library["image_no"].astype(int)
    else:
        existing_library = seed_library.copy()

    name_master = load_name_master(args.name_csv)

    processed_image_nos = set(existing_library["image_no"].astype(int))
    pending_images = []
    for image_path in sorted(args.image_dir.glob("*.png"), key=lambda p: int(p.stem)):
        image_no = int(image_path.stem)
        if image_no <= max_existing_image_no:
            continue
        if image_no in processed_image_nos:
            continue
        pending_images.append(image_path)

    batch_images = pending_images[: args.batch_size]

    incremental_dir = args.out_dir / "incremental_image_dir"
    incremental_dir.mkdir(parents=True, exist_ok=True)
    for old_file in incremental_dir.glob("*.png"):
        old_file.unlink()

    for image_path in batch_images:
        dst = incremental_dir / image_path.name
        try:
            dst.hardlink_to(image_path)
        except Exception:
            dst.write_bytes(image_path.read_bytes())

    if batch_images:
        new_library = build_4q_library(incremental_dir, name_master, args.out_dir / "library_cache")
        new_library["image_no"] = new_library["image_no"].astype(int)
        full_library = (
            pd.concat([existing_library, new_library], ignore_index=True)
            .sort_values("image_no")
            .drop_duplicates(subset=["image_no"], keep="last")
            .reset_index(drop=True)
        )
    else:
        full_library = existing_library.copy()

    full_library.to_csv(library_csv, index=False, encoding="utf-8-sig")

    remaining_after_batch = max(0, len(pending_images) - len(batch_images))
    if remaining_after_batch > 0:
        print(
            f"[INFO] library_progress existing={len(seed_library)} added_total={len(full_library) - len(seed_library)} "
            f"processed_this_run={len(batch_images)} remaining={remaining_after_batch}"
        )
        print(f"[OUT] library={library_csv}")
        return 0

    base_df = build_pre_oct_base(full_library, args.price_dir)
    base_df = add_learning_date(base_df, args.learning_date)
    base_csv = args.out_dir / "q4_pre_base_candidates.csv"
    base_df.to_csv(base_csv, index=False, encoding="utf-8-sig")

    prepare_mapped_image_dir(full_library, args.image_dir, args.out_dir / "mapped_images")

    detail = enrich_with_numeric_ocr(base_df, full_library)
    detail = add_learning_date(detail, args.learning_date)
    detail_csv = args.out_dir / "q4_pre_shikiho_feature_ranking.csv"
    detail.to_csv(detail_csv, index=False, encoding="utf-8-sig")

    selected = select_candidates(detail)
    selected = attach_realized_returns(selected, args.price_dir)
    selected = add_learning_date(selected, args.learning_date)
    selected_csv = args.out_dir / "q4_pre_selected_candidates.csv"
    selected.to_csv(selected_csv, index=False, encoding="utf-8-sig")
    operational_selected_csv = args.out_dir / "operational" / "q4_pre_selected_candidates_operational.csv"
    write_operational_csv(selected, operational_selected_csv)

    summary = pd.DataFrame(
        [
            {
                "dataset_id": "q4_pre_analysis_20250930_full",
                "learning_date": args.learning_date,
                "training_cutoff_date": args.learning_date,
                "image_count_total": int(len(full_library)),
                "image_count_existing": int(len(seed_library)),
                "image_count_new": int(len(full_library) - len(seed_library)),
                "image_no_existing_max": max_existing_image_no,
                "selected_count": int(len(selected)),
                "output_dir": str(args.out_dir),
                "source_image_dir": str(args.image_dir),
            },
            {
                "dataset_id": "4q2_selection_20251231",
                "learning_date": "2025-12-31",
                "training_cutoff_date": "2025-12-31",
                "image_count_total": None,
                "image_count_existing": None,
                "image_count_new": None,
                "image_no_existing_max": None,
                "selected_count": None,
                "output_dir": "projects/shikiho_text_parser/output/4q2_selection",
                "source_image_dir": "C:/Users/mitsu/OneDrive/ドキュメント/四季報DB2025/4Q-2",
            },
        ]
    )
    summary_csv = args.out_dir / "q4_learning_dataset_manifest.csv"
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print(
        f"[INFO] existing_images={len(seed_library)} "
        f"new_images={len(full_library) - len(seed_library)} full_images={len(full_library)}"
    )
    print(f"[OUT] library={library_csv}")
    print(f"[OUT] base={base_csv}")
    print(f"[OUT] detail={detail_csv}")
    print(f"[OUT] selected={selected_csv}")
    print(f"[OUT] selected_operational={operational_selected_csv}")
    print(f"[OUT] manifest={summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

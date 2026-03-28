from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[2]
IMAGE_LIBRARY_CSV = ROOT / "projects" / "quarterly_ranker" / "output" / "q2_2024_pre_analysis_20240630_aligned" / "q2_2024_image_library.csv"
IMAGE_DIR = ROOT / "projects" / "quarterly_ranker" / "output" / "q2_2024_pre_analysis_20240630_aligned" / "mapped_images"
OCR_SCRIPT = ROOT / "projects" / "quarterly_ranker" / "ocr_winrt.ps1"
OUT_DIR = ROOT / "projects" / "shikiho_text_parser" / "output" / "q2_2024_theme_section_regenerated"


def ocr_image(image_path: Path) -> str:
    cmd = [
        "powershell",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(OCR_SCRIPT.resolve()),
        "-ImagePath",
        str(image_path.resolve()),
    ]
    res = subprocess.run(cmd, capture_output=True, text=False, check=False)
    stdout = res.stdout.decode("utf-8", errors="ignore").strip()
    if res.returncode != 0:
        stderr = res.stderr.decode("cp932", errors="ignore").strip()
        raise RuntimeError(f"OCR failed: image={image_path} stderr={stderr}")
    return stdout


def save_crop(img: Image.Image, box: tuple[int, int, int, int], out_path: Path, scale: int, threshold: int | None) -> Path:
    crop = img.crop(box)
    crop = ImageOps.grayscale(crop)
    crop = ImageOps.autocontrast(crop)
    crop = crop.resize((crop.width * scale, crop.height * scale))
    if threshold is not None:
        crop = crop.point(lambda p: 255 if p > threshold else 0)
    crop = crop.convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path)
    return out_path


def compact_text(text: str) -> str:
    return str(text).replace(" ", "").replace("　", "").replace("\n", "")


def build_variants(img: Image.Image) -> dict[str, tuple[tuple[int, int, int, int], int, int | None]]:
    w, h = img.size
    return {
        "theme_tight": ((0, int(h * 0.23), int(w * 0.47), int(h * 0.43)), 4, 165),
        "theme_wide": ((0, int(h * 0.21), int(w * 0.55), int(h * 0.48)), 4, 160),
        "theme_bodyleft": ((0, int(h * 0.20), int(w * 0.60), int(h * 0.56)), 3, None),
    }


def choose_best_text(variant_texts: dict[str, str]) -> tuple[str, str]:
    scored: list[tuple[int, str, str]] = []
    for name, text in variant_texts.items():
        compact = compact_text(text)
        score = len(compact)
        score += sum(compact.count(k) * 5 for k in ["半導", "防衛", "宇宙", "電力", "物流", "倉庫", "建設", "不動産", "医療", "人材", "AI"])
        scored.append((score, name, text))
    scored.sort(reverse=True)
    _, best_name, best_text = scored[0]
    return best_name, best_text


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate q2_2024 theme/feature-section OCR with separate pipeline.")
    parser.add_argument("--limit", type=int, default=0, help="Optional number of images to process.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    crop_dir = OUT_DIR / "crops"
    detail_df = pd.read_csv(IMAGE_LIBRARY_CSV, usecols=["code", "ticker", "company_name", "sector"]).drop_duplicates(subset=["ticker"])
    if args.limit and args.limit > 0:
        detail_df = detail_df.head(args.limit)

    out_csv = OUT_DIR / "q2_2024_theme_section_regenerated.csv"
    if out_csv.exists():
        existing = pd.read_csv(out_csv)
        done = set(existing["ticker"].astype(str))
        rows: list[dict[str, str]] = existing.to_dict(orient="records")
    else:
        done = set()
        rows = []
    for _, row in detail_df.iterrows():
        code = str(row["code"]).strip()
        ticker = str(row["ticker"])
        if ticker in done:
            continue
        image_path = IMAGE_DIR / f"{code}.png"
        if not image_path.exists():
            continue
        img = Image.open(image_path)
        variants = build_variants(img)
        variant_texts: dict[str, str] = {}
        for name, (box, scale, threshold) in variants.items():
            crop_path = crop_dir / f"{code}_{name}.png"
            txt_path = crop_dir / f"{code}_{name}.txt"
            if txt_path.exists():
                text = txt_path.read_text(encoding="utf-8", errors="ignore")
            else:
                save_crop(img, box, crop_path, scale=scale, threshold=threshold)
                text = ocr_image(crop_path)
                txt_path.write_text(text, encoding="utf-8")
            variant_texts[name] = text
        best_variant, best_text = choose_best_text(variant_texts)
        rows.append(
            {
                "code": code,
                "ticker": ticker,
                "company_name": str(row["company_name"]),
                "sector": str(row["sector"]),
                "best_variant": best_variant,
                "theme_text_regenerated": best_text,
                "theme_text_regenerated_compact": compact_text(best_text),
            }
        )
        pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

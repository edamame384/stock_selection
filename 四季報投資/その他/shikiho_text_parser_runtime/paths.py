from __future__ import annotations

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RAW_4Q2_DIR = DATA_DIR / "raw" / "4Q-2"
PRICE_DIR = DATA_DIR / "prices_4q2"
PRICE_FULL_DIR = DATA_DIR / "prices_full"
REFERENCE_DIR = DATA_DIR / "reference"
SECTOR_MASTER_PATH = REFERENCE_DIR / "sector_master_template.csv"
OUTPUT_DIR = PROJECT_DIR / "output"

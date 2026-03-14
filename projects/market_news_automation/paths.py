from __future__ import annotations

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = PROJECT_DIR / "config"
OUTPUT_DIR = PROJECT_DIR / "output"
QUERIES_PATH = CONFIG_DIR / "queries.json"
SNAPSHOT_JSONL = OUTPUT_DIR / "news_snapshots.jsonl"
SNAPSHOT_CSV = OUTPUT_DIR / "news_snapshots.csv"
DAILY_FEATURES_CSV = OUTPUT_DIR / "daily_news_features.csv"

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from projects.market_news_automation.paths import (
    OUTPUT_DIR,
    QUERIES_PATH,
    SNAPSHOT_CSV,
    SNAPSHOT_JSONL,
)
from projects.shikiho_text_parser.fetch_market_news_signal import (
    fetch_feed_titles,
    google_news_rss_url,
    score_titles,
)


def load_queries() -> list[dict]:
    payload = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    return list(payload.get("queries", []))


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(ZoneInfo("Asia/Tokyo"))
    rows: list[dict] = []
    for item in load_queries():
        name = str(item.get("name", "")).strip()
        query = str(item.get("query", "")).strip()
        if not name or not query:
            continue
        url = google_news_rss_url(query)
        titles = fetch_feed_titles(url, limit=30)
        scores = score_titles(titles)
        rows.append(
            {
                "captured_at_jst": now.strftime("%Y-%m-%d %H:%M:%S"),
                "capture_date": now.strftime("%Y-%m-%d"),
                "capture_hour": now.strftime("%H"),
                "name": name,
                "query": query,
                "headline_count": int(scores["headline_count"]),
                "positive_count": int(scores["positive_count"]),
                "negative_count": int(scores["negative_count"]),
                "sentiment_score": float(scores["sentiment_score"]),
                "titles_json": json.dumps(titles, ensure_ascii=False),
            }
        )

    with SNAPSHOT_JSONL.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_header = not SNAPSHOT_CSV.exists()
    with SNAPSHOT_CSV.open("a", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    summary = {
        "captured_at_jst": now.strftime("%Y-%m-%d %H:%M:%S"),
        "feeds": len(rows),
        "output_jsonl": str(SNAPSHOT_JSONL),
        "output_csv": str(SNAPSHOT_CSV),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

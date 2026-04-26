from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from datetime import timedelta
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from projects.shikiho_text_parser.fetch_market_news_signal import NEGATIVE_WORDS, POSITIVE_WORDS


DEFAULT_QUERIES = {
    "jp_market": '"Nikkei 225" OR "Nikkei" OR "日経平均" OR "Japanese stocks" OR "日本株"',
    "us_market": '"Dow Jones" OR "ダウ平均" OR NASDAQ OR "Nasdaq" OR "S&P 500" OR "S&P500"',
    "volatility": 'VIX OR "CBOE Volatility Index" OR "恐怖指数"',
    "fx": '"USDJPY" OR "USD/JPY" OR "ドル円" OR "yen"',
}


def _gdelt_doc_url(query: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp, max_records: int = 50) -> str:
    params = {
        "query": query,
        "mode": "ArtList",
        "maxrecords": str(max_records),
        "format": "json",
        "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end_dt.strftime("%Y%m%d%H%M%S"),
    }
    return "https://api.gdeltproject.org/api/v2/doc/doc?" + urllib.parse.urlencode(params)


def _fetch_gdelt_articles(query: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp, max_records: int = 50) -> list[dict]:
    url = _gdelt_doc_url(query, start_dt, end_dt, max_records=max_records)
    req = urllib.request.Request(url, headers={"User-Agent": "stock-ml-news/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return list(data.get("articles", []) or [])


def _score_texts(texts: list[str]) -> dict[str, float]:
    joined = "\n".join(texts)
    pos = sum(joined.count(w) for w in POSITIVE_WORDS)
    neg = sum(joined.count(w) for w in NEGATIVE_WORDS)
    total = max(len(texts), 1)
    return {
        "headline_count": float(len(texts)),
        "positive_count": float(pos),
        "negative_count": float(neg),
        "sentiment_score": float((pos - neg) / total),
    }


def build_historical_news_signal(start_date: str, end_date: str, max_records: int = 50) -> pd.DataFrame:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    rows: list[dict] = []
    current = start
    while current <= end:
        next_day = current + timedelta(days=1)
        row = {"date": current.date().isoformat()}
        agg_texts: list[str] = []
        for name, query in DEFAULT_QUERIES.items():
            try:
                articles = _fetch_gdelt_articles(query, current, next_day, max_records=max_records)
            except Exception:
                articles = []
            texts = []
            for article in articles:
                title = str(article.get("title", "") or "").strip()
                seendate = str(article.get("seendate", "") or "").strip()
                if title:
                    texts.append(title)
                elif seendate:
                    texts.append(seendate)
            scores = _score_texts(texts)
            row[f"{name}_headline_count"] = scores["headline_count"]
            row[f"{name}_sentiment_score"] = scores["sentiment_score"]
            agg_texts.extend(texts)
        agg_scores = _score_texts(agg_texts)
        row["aggregate_headline_count"] = agg_scores["headline_count"]
        row["aggregate_sentiment_score"] = agg_scores["sentiment_score"]
        rows.append(row)
        current = next_day
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch historical market-news sentiment by day using GDELT.")
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, required=True)
    parser.add_argument("--max-records", type=int, default=50)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("projects/shikiho_text_parser/output/historical_market_news_signal.csv"),
    )
    args = parser.parse_args()

    df = build_historical_news_signal(
        start_date=args.start_date,
        end_date=args.end_date,
        max_records=args.max_records,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(df.head().to_string(index=False))
    print(f"[OUT] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

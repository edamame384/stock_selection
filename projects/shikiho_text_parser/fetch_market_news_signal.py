from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


POSITIVE_WORDS = [
    "上昇",
    "反発",
    "続伸",
    "最高値",
    "改善",
    "緩和",
    "増益",
    "買い",
    "堅調",
    "強い",
]

NEGATIVE_WORDS = [
    "下落",
    "反落",
    "続落",
    "急落",
    "悪化",
    "警戒",
    "減益",
    "売り",
    "軟調",
    "弱い",
]

DEFAULT_FEEDS = {
    "jp_market": "https://news.google.com/rss/search?q=%E6%97%A5%E7%B5%8C%E5%B9%B3%E5%9D%87+OR+%E6%97%A5%E6%9C%AC%E6%A0%AA&hl=ja&gl=JP&ceid=JP:ja",
    "us_market": "https://news.google.com/rss/search?q=%E3%83%80%E3%82%A6%E5%B9%B3%E5%9D%87+OR+NASDAQ+OR+S%26P500&hl=ja&gl=JP&ceid=JP:ja",
    "volatility": "https://news.google.com/rss/search?q=VIX+OR+%E6%81%90%E6%80%96%E6%8C%87%E6%95%B0&hl=ja&gl=JP&ceid=JP:ja",
    "fx": "https://news.google.com/rss/search?q=%E3%83%89%E3%83%AB%E5%86%86+OR+USDJPY&hl=ja&gl=JP&ceid=JP:ja",
}


def google_news_rss_url(query: str, hl: str = "ja", gl: str = "JP", ceid: str = "JP:ja") -> str:
    encoded = urllib.parse.quote(query, safe="")
    return f"https://news.google.com/rss/search?q={encoded}&hl={hl}&gl={gl}&ceid={urllib.parse.quote(ceid, safe='')}"


def fetch_feed_titles(url: str, limit: int = 30) -> list[str]:
    req = urllib.request.Request(url, headers={"User-Agent": "stock-ml-news/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        raw = resp.read()
    root = ET.fromstring(raw)
    titles: list[str] = []
    for item in root.findall(".//item"):
        title = item.findtext("title") or ""
        title = title.strip()
        if not title:
            continue
        titles.append(title)
        if len(titles) >= limit:
            break
    return titles


def score_titles(titles: list[str]) -> dict[str, float]:
    joined = "\n".join(titles)
    pos = sum(joined.count(w) for w in POSITIVE_WORDS)
    neg = sum(joined.count(w) for w in NEGATIVE_WORDS)
    total = max(len(titles), 1)
    sentiment = (pos - neg) / total
    return {
        "headline_count": float(len(titles)),
        "positive_count": float(pos),
        "negative_count": float(neg),
        "sentiment_score": float(sentiment),
    }


def build_market_news_signal(limit_per_feed: int = 30) -> dict:
    feeds_out: dict[str, dict] = {}
    aggregate_titles: list[str] = []
    for name, url in DEFAULT_FEEDS.items():
        titles = fetch_feed_titles(url, limit=limit_per_feed)
        feed_score = score_titles(titles)
        feeds_out[name] = {
            "url": url,
            "titles": titles,
            **feed_score,
        }
        aggregate_titles.extend(titles)

    aggregate = score_titles(aggregate_titles)
    aggregate["as_of_jst"] = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d %H:%M:%S")
    return {
        "aggregate": aggregate,
        "feeds": feeds_out,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch live market-news headlines and build a simple sentiment signal.")
    parser.add_argument("--output", type=Path, default=Path("projects/shikiho_text_parser/output/live_market_news_signal.json"))
    parser.add_argument("--limit-per-feed", type=int, default=30)
    parser.add_argument("--query", type=str, default="", help="If set, fetch this Google News query as a single custom feed.")
    args = parser.parse_args()

    if args.query.strip():
        url = google_news_rss_url(args.query.strip())
        titles = fetch_feed_titles(url, limit=args.limit_per_feed)
        scores = score_titles(titles)
        result = {
            "aggregate": {
                **scores,
                "as_of_jst": datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d %H:%M:%S"),
                "query": args.query.strip(),
            },
            "feeds": {
                "custom_query": {
                    "url": url,
                    "titles": titles,
                    **scores,
                }
            },
        }
    else:
        result = build_market_news_signal(limit_per_feed=args.limit_per_feed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from projects.market_news_automation.paths import OUTPUT_DIR, QUERIES_PATH
from projects.shikiho_text_parser.fetch_market_news_signal import (
    POSITIVE_WORDS,
    NEGATIVE_WORDS,
    google_news_rss_url,
)


CANDIDATE_DIR = OUTPUT_DIR / "title_vs_body_candidate"
SUMMARY_CSV = CANDIDATE_DIR / "summary.csv"
ARTICLES_CSV = CANDIDATE_DIR / "articles.csv"
SUMMARY_JSON = CANDIDATE_DIR / "summary.json"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36 stock-ml-news-compare/1.0"
)


@dataclass
class FeedItem:
    query_name: str
    query: str
    title: str
    rss_link: str
    article_url: str
    pub_date: str


def load_queries() -> list[dict]:
    payload = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    return list(payload.get("queries", []))


def fetch_url_text(url: str, timeout: int = 20) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        raw = resp.read()
    return raw.decode(charset, errors="replace")


def fetch_rss_items(query_name: str, query: str, limit: int) -> list[FeedItem]:
    url = google_news_rss_url(query)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as resp:
        raw = resp.read()
    root = ET.fromstring(raw)
    out: list[FeedItem] = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        rss_link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        if not title or not rss_link:
            continue
        out.append(
            FeedItem(
                query_name=query_name,
                query=query,
                title=title,
                rss_link=rss_link,
                article_url=unwrap_google_news_link(rss_link),
                pub_date=pub_date,
            )
        )
        if len(out) >= limit:
            break
    return out


def unwrap_google_news_link(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    qs = urllib.parse.parse_qs(parsed.query)
    for key in ("url", "u"):
        value = qs.get(key)
        if value and value[0]:
            return value[0]
    return url


def score_text(text: str) -> tuple[int, int, float]:
    pos = sum(text.count(word) for word in POSITIVE_WORDS)
    neg = sum(text.count(word) for word in NEGATIVE_WORDS)
    score = float(pos - neg)
    return pos, neg, score


def normalize_space(text: str) -> str:
    return " ".join(text.replace("\u3000", " ").split())


def extract_article_body(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "footer", "header", "nav", "aside"]):
        tag.decompose()

    candidates: list[str] = []
    selectors = [
        "article",
        "main",
        "[role='main']",
        ".article",
        ".article-body",
        ".articleBody",
        ".post-content",
        ".entry-content",
        ".content",
    ]
    for selector in selectors:
        for node in soup.select(selector):
            text = normalize_space(node.get_text(" ", strip=True))
            if len(text) >= 200:
                candidates.append(text)
    if not candidates:
        paragraphs = [
            normalize_space(p.get_text(" ", strip=True))
            for p in soup.find_all(["p", "div"])
        ]
        paragraphs = [p for p in paragraphs if len(p) >= 40]
        if paragraphs:
            candidates.append(" ".join(paragraphs[:30]))
    if not candidates:
        return ""
    candidates.sort(key=len, reverse=True)
    return candidates[0][:12000]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare title-only sentiment vs article-body sentiment for Google News RSS queries.")
    parser.add_argument("--limit-per-query", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=20)
    args = parser.parse_args()

    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d %H:%M:%S")

    summary_rows: list[dict] = []
    article_rows: list[dict] = []

    for query_item in load_queries():
        query_name = str(query_item.get("name", "")).strip()
        query = str(query_item.get("query", "")).strip()
        if not query_name or not query:
            continue
        feed_items = fetch_rss_items(query_name, query, limit=args.limit_per_query)
        title_joined = "\n".join(item.title for item in feed_items)
        title_pos, title_neg, title_score = score_text(title_joined)

        body_texts: list[str] = []
        fetched_count = 0
        for item in feed_items:
            body_text = ""
            fetch_error = ""
            if item.article_url:
                try:
                    html = fetch_url_text(item.article_url, timeout=args.timeout)
                    body_text = extract_article_body(html)
                    if body_text:
                        fetched_count += 1
                        body_texts.append(body_text)
                except Exception as exc:  # pragma: no cover - network variability
                    fetch_error = f"{type(exc).__name__}: {exc}"
            body_pos, body_neg, body_score = score_text(body_text)
            article_rows.append(
                {
                    "captured_at_jst": now,
                    "query_name": query_name,
                    "query": query,
                    "title": item.title,
                    "rss_link": item.rss_link,
                    "article_url": item.article_url,
                    "pub_date": item.pub_date,
                    "title_positive_count": sum(item.title.count(word) for word in POSITIVE_WORDS),
                    "title_negative_count": sum(item.title.count(word) for word in NEGATIVE_WORDS),
                    "title_sentiment_score": float(sum(item.title.count(word) for word in POSITIVE_WORDS) - sum(item.title.count(word) for word in NEGATIVE_WORDS)),
                    "body_char_count": len(body_text),
                    "body_positive_count": body_pos,
                    "body_negative_count": body_neg,
                    "body_sentiment_score": body_score,
                    "fetch_error": fetch_error,
                    "body_preview": body_text[:500],
                }
            )

        body_joined = "\n".join(body_texts)
        body_pos, body_neg, body_score = score_text(body_joined)
        summary_rows.append(
            {
                "captured_at_jst": now,
                "query_name": query_name,
                "query": query,
                "rss_item_count": len(feed_items),
                "article_body_fetched_count": fetched_count,
                "title_positive_count": title_pos,
                "title_negative_count": title_neg,
                "title_sentiment_score": title_score,
                "body_positive_count": body_pos,
                "body_negative_count": body_neg,
                "body_sentiment_score": body_score,
                "score_delta_body_minus_title": body_score - title_score,
            }
        )

    write_csv(SUMMARY_CSV, summary_rows)
    write_csv(ARTICLES_CSV, article_rows)
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "captured_at_jst": now,
                "summary_csv": str(SUMMARY_CSV),
                "articles_csv": str(ARTICLES_CSV),
                "query_count": len(summary_rows),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"summary_csv": str(SUMMARY_CSV), "articles_csv": str(ARTICLES_CSV)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

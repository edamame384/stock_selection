"""
DB上の株価データを最新（昨日）まで更新するスクリプト。
stock_pricesテーブルの最新日の翌日から昨日までを yfinance でフェッチして INSERT する。
"""
from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from pathlib import Path

# SSL fix for corporate proxy
os.environ.setdefault("CURL_CA_BUNDLE",
    str(Path(__file__).resolve().parents[1] / "win_certs.pem"))

import psycopg2
import psycopg2.extras
import pandas as pd
import yfinance as yf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DB_DSN = "postgresql://postgres:ogm384@localhost:5432/stock_selection"

YESTERDAY = date.today() - timedelta(days=1)


def get_db_max_date(conn) -> date:
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(date)::date FROM stock_prices")
        row = cur.fetchone()
        return row[0]


def get_symbols(conn) -> list[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT symbol FROM stocks ORDER BY symbol")
        return [r[0] for r in cur.fetchall()]


def fetch_prices(symbol: str, start: date, end: date) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(), auto_adjust=False)
    if df.empty:
        return df
    df.index = pd.to_datetime(df.index).date
    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Adj Close": "adj_close", "Volume": "volume",
    })
    df["symbol"] = symbol
    df.index.name = "date"
    df = df.reset_index()
    cols = ["symbol", "date", "adj_close", "close", "high", "low", "open", "volume"]
    available = [c for c in cols if c in df.columns]
    return df[available]


def upsert_prices(conn, rows: list[tuple]) -> int:
    if not rows:
        return 0
    sql = """
        INSERT INTO stock_prices (symbol, date, adj_close, close, high, low, open, volume)
        VALUES %s
        ON CONFLICT (symbol, date) DO NOTHING
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, rows, page_size=1000)
    conn.commit()
    return len(rows)


def main():
    conn = psycopg2.connect(DB_DSN)
    max_date = get_db_max_date(conn)
    start = max_date + timedelta(days=1)
    end = YESTERDAY

    if start > end:
        print(f"すでに最新です（DB最終日: {max_date}、昨日: {end}）")
        conn.close()
        return

    print(f"取得期間: {start} ～ {end}")
    symbols = get_symbols(conn)
    print(f"対象銘柄数: {len(symbols)}")

    total_inserted = 0
    for i, sym in enumerate(symbols, 1):
        try:
            df = fetch_prices(sym, start, end)
            if df.empty:
                print(f"[{i}/{len(symbols)}] {sym}: データなし")
                continue
            rows = [
                (r.symbol, r.date,
                 getattr(r, "adj_close", None), getattr(r, "close", None),
                 getattr(r, "high", None), getattr(r, "low", None),
                 getattr(r, "open", None), getattr(r, "volume", None))
                for r in df.itertuples(index=False)
            ]
            n = upsert_prices(conn, rows)
            total_inserted += n
            print(f"[{i}/{len(symbols)}] {sym}: {n}件追加")
        except Exception as e:
            print(f"[{i}/{len(symbols)}] {sym}: ERROR {e}")

    print(f"\n完了: 合計 {total_inserted} 件追加")
    conn.close()


if __name__ == "__main__":
    main()

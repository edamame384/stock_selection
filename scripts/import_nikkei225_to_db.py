"""
日経平均株価（^N225）を 2008-01-01 から今日まで取得して
market_indices テーブルに保存するスクリプト。
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

# SSL fix for corporate proxy
os.environ.setdefault("CURL_CA_BUNDLE",
    str(Path(__file__).resolve().parents[1] / "win_certs.pem"))

import psycopg2
import psycopg2.extras
import pandas as pd
import yfinance as yf

DB_DSN = "postgresql://postgres:ogm384@localhost:5432/stock_selection"

START_DATE = date(2008, 1, 1)
END_DATE   = date.today()


def ensure_table(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS market_indices (
                symbol  VARCHAR(20)  NOT NULL,
                date    DATE         NOT NULL,
                open    DOUBLE PRECISION,
                high    DOUBLE PRECISION,
                low     DOUBLE PRECISION,
                close   DOUBLE PRECISION,
                adj_close DOUBLE PRECISION,
                volume  BIGINT,
                PRIMARY KEY (symbol, date)
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_market_indices_date ON market_indices (date)")
    conn.commit()
    print("market_indices テーブル準備完了")


def fetch_nikkei(start: date, end: date) -> pd.DataFrame:
    ticker = yf.Ticker("^N225")
    df = ticker.history(
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=False,
    )
    if df.empty:
        raise ValueError("^N225 データが取得できませんでした")
    df.index = pd.to_datetime(df.index).date
    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Adj Close": "adj_close", "Volume": "volume",
    })
    df["symbol"] = "^N225"
    df.index.name = "date"
    df = df.reset_index()
    return df[["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"]]


def upsert(conn, df: pd.DataFrame) -> int:
    rows = [
        (r.symbol, r.date, r.open, r.high, r.low, r.close,
         getattr(r, "adj_close", None), getattr(r, "volume", None))
        for r in df.itertuples(index=False)
    ]
    sql = """
        INSERT INTO market_indices (symbol, date, open, high, low, close, adj_close, volume)
        VALUES %s
        ON CONFLICT (symbol, date) DO UPDATE SET
            open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low,
            close=EXCLUDED.close, adj_close=EXCLUDED.adj_close, volume=EXCLUDED.volume
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, rows, page_size=500)
    conn.commit()
    return len(rows)


def main():
    conn = psycopg2.connect(DB_DSN)
    ensure_table(conn)

    print(f"^N225 取得: {START_DATE} ～ {END_DATE}")
    df = fetch_nikkei(START_DATE, END_DATE)
    print(f"取得件数: {len(df)} 行")

    n = upsert(conn, df)
    print(f"DB保存完了: {n} 件")

    # 確認
    with conn.cursor() as cur:
        cur.execute("SELECT MIN(date), MAX(date), COUNT(*) FROM market_indices WHERE symbol='^N225'")
        row = cur.fetchone()
        print(f"DB確認 ^N225: {row[0]} ～ {row[1]}、計 {row[2]} 件")

    conn.close()


if __name__ == "__main__":
    main()

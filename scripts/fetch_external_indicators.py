"""
Fetch and cache external indicators (VIX, DJI, USDJPY, S&P500) from yfinance.
SSL issue worked around with curl_cffi session (verify=False, impersonate=chrome).
Cached to data/external_market/.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import pandas as pd
import yfinance as yf
from curl_cffi import requests as curl_req

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "external_market"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = {
    "vix":     "^VIX",
    "dji":     "^DJI",
    "usdjpy":  "JPY=X",
    "sp500":   "^GSPC",
}


def _new_session():
    return curl_req.Session(verify=False, impersonate="chrome")


def fetch_one(name: str, ticker: str, force: bool = False) -> pd.DataFrame:
    cache_path = CACHE_DIR / f"{name}_daily.csv"
    if cache_path.exists() and not force:
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        df.index.name = "date"
        return df
    session = _new_session()
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        raw = yf.download(ticker, period="max", interval="1d",
                          auto_adjust=False, progress=False, threads=False,
                          session=session)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    if raw.empty:
        raise RuntimeError(f"empty download for {ticker}")
    raw.index.name = "date"
    raw.to_csv(cache_path)
    return raw


def fetch_all(force: bool = False) -> dict[str, pd.DataFrame]:
    out = {}
    for name, ticker in TICKERS.items():
        try:
            df = fetch_one(name, ticker, force=force)
            print(f"[{name}] {ticker}  rows={len(df)}  "
                  f"first={df.index.min().date()}  last={df.index.max().date()}")
            out[name] = df
        except Exception as e:
            print(f"[{name}] {ticker} FAILED: {type(e).__name__}: {e}")
    return out


if __name__ == "__main__":
    force = "--force" in sys.argv
    fetch_all(force=force)

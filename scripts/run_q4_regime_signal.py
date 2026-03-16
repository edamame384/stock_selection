from __future__ import annotations

import argparse
import io
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.shikiho_text_parser.backtest_4q2_signals import calc_signal, load_selected
from projects.shikiho_text_parser.backtest_q4_regime_switch import nikkei_regime_custom


DEFAULT_SELECTED = ROOT / "projects" / "quarterly_ranker" / "output" / "q4_pre_analysis_20250930_full" / "q4_pre_selected_candidates_condition2_input.csv"
DEFAULT_PRICE_DIR = ROOT / "data" / "prices"
EXTERNAL_MARKET_DIR = ROOT / "data" / "external_market"


def next_business_day(ts: pd.Timestamp) -> pd.Timestamp:
    d = ts + pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d += pd.Timedelta(days=1)
    return d.normalize()


def load_or_update_nikkei_close() -> pd.Series:
    EXTERNAL_MARKET_DIR.mkdir(parents=True, exist_ok=True)
    path = EXTERNAL_MARKET_DIR / "nikkei225_daily.csv"
    if not path.exists():
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            df = yf.download("^N225", period="5y", auto_adjust=False, progress=False, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.reset_index()
        df.to_csv(path, index=False, encoding="utf-8-sig")
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
    return df.set_index("Date")["Close"].astype(float)


def fetch_symbol_name(symbol: str) -> str:
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            info = yf.Ticker(symbol).get_info()
        if isinstance(info, dict):
            name = info.get("shortName") or info.get("longName") or info.get("displayName")
            if name:
                return str(name).strip()
    except Exception:
        pass
    return symbol.removesuffix(".T")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate regime-switch signal log for Q4 notifications.")
    parser.add_argument("--selected-csv", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--price-dir", type=Path, default=DEFAULT_PRICE_DIR)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--up-ret20-threshold", type=float, default=0.0)
    parser.add_argument("--sideways-abs-ret20-threshold", type=float, default=0.07)
    args = parser.parse_args()

    selected = load_selected(args.selected_csv).drop_duplicates(subset=["ticker"], keep="first").copy()
    static_lookup = selected.set_index("ticker").to_dict("index")

    price_map: dict[str, pd.DataFrame] = {}
    latest_dates: list[pd.Timestamp] = []
    for ticker in selected["ticker"]:
        path = args.price_dir / f"{ticker.replace('.', '_')}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").set_index("Date")
        if df.empty:
            continue
        price_map[ticker] = df
        latest_dates.append(df.index.max())
    if not latest_dates:
        raise SystemExit("No price data found for selected universe.")

    signal_date = max(latest_dates)
    trade_date = next_business_day(signal_date)
    n225_close = load_or_update_nikkei_close()
    regime = nikkei_regime_custom(
        n225_close,
        signal_date,
        up_ret20_threshold=args.up_ret20_threshold,
        sideways_abs_ret20_threshold=args.sideways_abs_ret20_threshold,
    )
    if regime == "up":
        method_name = "condition2"
        thresholds = {"promising": 0.66, "trend": 0.50, "sector": 0.55}
    elif regime == "sideways":
        method_name = "breakout_1.5"
        thresholds = {"promising": 0.72, "trend": 0.60, "sector": 0.55}
    else:
        method_name = "no_trade"
        thresholds = None

    print(
        f"[META] regime={regime} method={method_name} "
        f"signal_date={signal_date.date().isoformat()} trade_date={trade_date.date().isoformat()}"
    )
    if thresholds is None:
        return 0

    candidates = []
    for ticker, df in price_map.items():
        if signal_date not in df.index:
            continue
        sig, metrics = calc_signal(
            df,
            signal_date,
            static_lookup[ticker],
            pd.DataFrame(),
            thresholds["promising"],
            thresholds["trend"],
            thresholds["sector"],
            -1.0,
            0.0,
        )
        if not sig:
            continue
        close_price = float(df.loc[signal_date, "Close"])
        entry_price = close_price * 1.015
        tp_price = entry_price * 1.08
        sl_price = entry_price * 0.95
        signal_score_pct = float(metrics.get("signal_score", 0.0)) * 100.0
        sector = str(static_lookup[ticker].get("simple_sector", static_lookup[ticker].get("sector", "UNKNOWN")))
        company = fetch_symbol_name(ticker)
        candidates.append(
            {
                "ticker": ticker,
                "company": company,
                "sector": sector,
                "signal_score_pct": signal_score_pct,
                "close_price": close_price,
                "entry_price": entry_price,
                "tp_price": tp_price,
                "sl_price": sl_price,
                "method": method_name,
            }
        )

    candidates.sort(key=lambda x: x["signal_score_pct"], reverse=True)
    for row in candidates[: args.top_n]:
        print(
            f"[BUY] {row['ticker']} | method={row['method']} close={row['close_price']:.2f} "
            f"tp_prob={row['signal_score_pct']:.2f}% lmt=1.5% lmt_price={row['entry_price']:.2f} "
            f"tp=8.0% tp_price={row['tp_price']:.2f} sl=5.0% sl_price={row['sl_price']:.2f}"
        )
        print(
            f"[PICK][regime_switch] {row['ticker']} tp_prob={row['signal_score_pct']:.2f}% "
            f"sector={row['sector']} method={row['method']} company={row['company']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

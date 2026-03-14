from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from time import sleep
from urllib import error as urlerror
from urllib import request as urlrequest
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

SECTOR_INDEX_TICKER_MAP = {
    "Basic Materials": "XLB",
    "Communication Services": "XLC",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Real Estate": "XLRE",
    "Technology": "XLK",
    "Utilities": "XLU",
}

SECTOR_ONEHOT_COLS = [
    "sec_basic_materials",
    "sec_communication_services",
    "sec_consumer_cyclical",
    "sec_consumer_defensive",
    "sec_energy",
    "sec_financial_services",
    "sec_healthcare",
    "sec_industrials",
    "sec_real_estate",
    "sec_technology",
    "sec_utilities",
    "sec_unknown",
]

FEATURE_COLS = [
    "ret_1",
    "ret_5",
    "ma_gap",
    "vol_20",
    "volume_ratio_20",
    "volume_z_20",
    "value_traded_log",
    "ret_dji",
    "corr_n225_20",
    "corr_dji_20",
    "corr_sp500_20",
    "fut_nk_night_ret",
    "ret_dow_fut",
    "corr_dow_fut_20",
    "sector_ret_1",
    "sector_corr_20",
    *SECTOR_ONEHOT_COLS,
]


@dataclass
class SignalResult:
    symbol: str
    last_date: pd.Timestamp
    last_close: float
    buy: bool
    tp_probability: float
    stop_loss_ratio: float
    limit_entry_ratio: float
    trade_date: date
    sector: str = "UNKNOWN"


@dataclass
class StopLossTuningResult:
    stop_loss_ratio: float
    win_rate: float
    avg_return: float
    trades: int


@dataclass
class LimitTuningResult:
    limit_entry_ratio: float
    win_rate: float
    avg_return: float
    fills: int
    candidates: int


@dataclass
class TradeCase:
    signal_close: float
    next_day_low: float
    future_closes: list[float]
    tp_probability: float
    signal_date: pd.Timestamp


@dataclass
class StrategyTuningResult:
    threshold: float
    stop_loss_ratio: float
    limit_entry_ratio: float
    win_rate: float
    avg_return_fill: float
    avg_return_signal: float
    fills: int
    candidates: int
    fill_rate: float
    max_drawdown: float
    avg_monthly_fills: float


DEFAULT_DISCORD_WEBHOOK_URL = (
    "https://discord.com/api/webhooks/1479858388090355762/"
    "WfKk_sdIufDciR-g-LksZCjlzdSrLHZQ__558rfwGrv-wH9E9nQzdeiSRE9gmfHbrjN8"
)


def fetch_sector(symbol: str) -> str:
    ticker = normalize_symbol(symbol)
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            info = yf.Ticker(ticker).get_info()
        if not isinstance(info, dict):
            return "UNKNOWN"
        sector = info.get("sectorDisp") or info.get("sector")
        if not sector:
            return "UNKNOWN"
        return str(sector)
    except Exception:
        return "UNKNOWN"


def fetch_symbol_sectors(symbols: list[str]) -> dict[str, str]:
    sectors: dict[str, str] = {}
    for s in symbols:
        n = normalize_symbol(s)
        sectors[n] = fetch_sector(n)
    return sectors


def sector_to_onehot_col(sector: str) -> str:
    mapping = {
        "Basic Materials": "sec_basic_materials",
        "Communication Services": "sec_communication_services",
        "Consumer Cyclical": "sec_consumer_cyclical",
        "Consumer Defensive": "sec_consumer_defensive",
        "Energy": "sec_energy",
        "Financial Services": "sec_financial_services",
        "Healthcare": "sec_healthcare",
        "Industrials": "sec_industrials",
        "Real Estate": "sec_real_estate",
        "Technology": "sec_technology",
        "Utilities": "sec_utilities",
    }
    return mapping.get(sector, "sec_unknown")


def normalize_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if symbol.startswith("TYO:"):
        symbol = symbol.split(":", 1)[1]
    if symbol.endswith(".T"):
        return symbol
    return f"{symbol}.T"


def base_symbol_code(symbol: str) -> str:
    code = symbol.strip().upper()
    if code.startswith("TYO:"):
        code = code.split(":", 1)[1]
    if code.endswith(".T"):
        code = code[:-2]
    return code


def next_business_day_jp(d: date) -> date:
    nxt = d + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt


def _shift_to_next_jp_business_day(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    shifted = series.copy()
    shifted.index = pd.to_datetime([next_business_day_jp(pd.Timestamp(d).date()) for d in shifted.index])
    shifted = shifted[~shifted.index.duplicated(keep="last")]
    return shifted.sort_index()


def send_discord_webhook(webhook_url: str, message: str) -> None:
    # Discord content hard limit is 2000 chars.
    safe_message = message if len(message) <= 1900 else message[:1900] + "\n...(truncated)"
    payload = {"content": safe_message}
    req = urlrequest.Request(
        webhook_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "stock-ml-bot/1.0",
        },
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=15) as resp:
            if resp.status not in (200, 204):
                raise ValueError(f"discord webhook status={resp.status}")
    except urlerror.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise ValueError(f"discord webhook failed: status={exc.code} body={body}") from exc
    except urlerror.URLError as exc:
        raise ValueError(f"discord webhook failed: {exc}") from exc


def _download_from_yahoo(symbol: str, period: str = "2y", retries: int = 3) -> pd.DataFrame:
    ticker = normalize_symbol(symbol)
    return _download_yahoo_ticker(ticker=ticker, period=period, retries=retries)


def _download_yahoo_ticker(ticker: str, period: str, retries: int = 3) -> pd.DataFrame:
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            # yfinance can emit retry/failure text directly to stdout/stderr.
            # We silence that noise because we already handle retries/fallback ourselves.
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                df = yf.download(
                    ticker,
                    period=period,
                    interval="1d",
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
            if isinstance(df.columns, pd.MultiIndex):
                if ticker in df.columns.get_level_values(-1):
                    df = df.xs(ticker, axis=1, level=-1, drop_level=True)
                else:
                    df.columns = df.columns.get_level_values(0)
            if not df.empty:
                return df
            last_error = ValueError("empty dataframe from Yahoo")
        except Exception as exc:
            last_error = exc

        if attempt < retries:
            sleep(1.0)

    if last_error is None:
        raise ValueError(f"No price data from Yahoo: {ticker}")
    raise ValueError(f"Yahoo fetch failed for {ticker}: {last_error}")


def _download_from_stooq(symbol: str) -> pd.DataFrame:
    code = base_symbol_code(symbol)
    url = f"https://stooq.com/q/d/l/?s={code}.jp&i=d"
    df = pd.read_csv(url)
    if df.empty or "Close" not in df.columns:
        raise ValueError(f"No price data from stooq: {symbol}")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    return df


def download_daily_data(symbol: str, period: str = "max") -> pd.DataFrame:
    try:
        return _download_from_yahoo(symbol=symbol, period=period)
    except Exception:
        df = _download_from_stooq(symbol=symbol)
        if df.empty:
            raise ValueError(f"No price data: {symbol}")
        return df


def download_index_returns(period: str) -> pd.DataFrame:
    index_tickers = {
        "n225": "^N225",
        "dji": "^DJI",
        "sp500": "^GSPC",
    }
    close_map: dict[str, pd.Series] = {}
    for key, ticker in index_tickers.items():
        try:
            idx_df = _download_yahoo_ticker(ticker=ticker, period=period, retries=3)
            close_map[key] = idx_df["Close"].astype(float)
        except Exception:
            continue

    if not close_map:
        return pd.DataFrame(columns=["ret_n225", "ret_dji", "ret_sp500"])

    closes = pd.concat(close_map, axis=1).sort_index()
    returns = closes.pct_change()
    for key in ("dji", "sp500"):
        if key in returns.columns:
            returns[key] = _shift_to_next_jp_business_day(returns[key].dropna())
    rename_map = {
        "n225": "ret_n225",
        "dji": "ret_dji",
        "sp500": "ret_sp500",
    }
    returns = returns.rename(columns=rename_map)
    return returns


def download_nikkei_futures_night_feature() -> pd.DataFrame:
    # Intraday history on Yahoo is limited, so this feature typically covers recent ~730 days.
    ticker = "NKD=F"
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            fut = yf.download(
                ticker,
                period="730d",
                interval="60m",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        if isinstance(fut.columns, pd.MultiIndex):
            if ticker in fut.columns.get_level_values(-1):
                fut = fut.xs(ticker, axis=1, level=-1, drop_level=True)
            else:
                fut.columns = fut.columns.get_level_values(0)
    except Exception:
        return pd.DataFrame(columns=["fut_nk_night_ret"])

    if fut.empty or "Close" not in fut.columns:
        return pd.DataFrame(columns=["fut_nk_night_ret"])

    idx = fut.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert("Asia/Tokyo")
    fut = fut.copy()
    fut.index = idx
    fut["date_jst"] = fut.index.date
    fut["hour_jst"] = fut.index.hour

    day_close = fut[fut["hour_jst"] == 15].groupby("date_jst")["Close"].last()
    night_23 = fut[fut["hour_jst"] == 23].groupby("date_jst")["Close"].last()

    common = day_close.index.intersection(night_23.index)
    if len(common) == 0:
        return pd.DataFrame(columns=["fut_nk_night_ret"])

    feature = pd.DataFrame(index=pd.to_datetime(common))
    feature["fut_nk_night_ret"] = night_23.loc[common] / day_close.loc[common] - 1.0
    feature["fut_nk_night_ret"] = _shift_to_next_jp_business_day(feature["fut_nk_night_ret"])
    feature.index.name = "Date"
    return feature.sort_index()


def download_dow_futures_feature(period: str) -> pd.DataFrame:
    ticker = "YM=F"
    try:
        df = _download_yahoo_ticker(ticker=ticker, period=period, retries=3)
    except Exception:
        return pd.DataFrame(columns=["ret_dow_fut"])
    if df.empty or "Close" not in df.columns:
        return pd.DataFrame(columns=["ret_dow_fut"])
    ret = pd.to_numeric(df["Close"], errors="coerce").pct_change().dropna()
    ret = _shift_to_next_jp_business_day(ret)
    out = pd.DataFrame(index=ret.index)
    out["ret_dow_fut"] = ret
    out.index.name = "Date"
    return out.sort_index()


def download_additional_macro_features(period: str) -> pd.DataFrame:
    macro_tickers = {
        "ret_vix": "^VIX",
        "ret_ixic": "^IXIC",
        "ret_usdjpy": "JPY=X",
    }
    series_map: dict[str, pd.Series] = {}
    for col, ticker in macro_tickers.items():
        try:
            df = _download_yahoo_ticker(ticker=ticker, period=period, retries=3)
        except Exception:
            continue
        if df.empty or "Close" not in df.columns:
            continue
        ret = pd.to_numeric(df["Close"], errors="coerce").pct_change().dropna()
        ret = _shift_to_next_jp_business_day(ret)
        series_map[col] = ret.rename(col)
    if not series_map:
        return pd.DataFrame(columns=list(macro_tickers.keys()))
    out = pd.concat(series_map.values(), axis=1).sort_index()
    out.index.name = "Date"
    return out


def download_sector_index_returns(period: str, sectors: set[str]) -> dict[str, pd.Series]:
    ret_map: dict[str, pd.Series] = {}
    for sector in sectors:
        if sector == "UNKNOWN":
            continue
        ticker = SECTOR_INDEX_TICKER_MAP.get(sector)
        if not ticker:
            continue
        try:
            df = _download_yahoo_ticker(ticker=ticker, period=period, retries=3)
            if df.empty or "Close" not in df.columns:
                continue
            ret_map[sector] = df["Close"].pct_change().rename("sector_ret_1")
        except Exception:
            continue
    return ret_map


def build_features(
    df: pd.DataFrame,
    index_returns: pd.DataFrame,
    futures_feature: pd.DataFrame,
    dow_futures_feature: pd.DataFrame,
    sector: str,
    sector_ret_map: dict[str, pd.Series],
    horizon_days: int,
    take_profit: float,
) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    feat["close"] = df["Close"]
    feat["ret_1"] = df["Close"].pct_change(1)
    feat["ret_5"] = df["Close"].pct_change(5)
    feat["ma_5"] = df["Close"].rolling(5).mean()
    feat["ma_20"] = df["Close"].rolling(20).mean()
    feat["ma_gap"] = feat["ma_5"] / feat["ma_20"] - 1.0
    feat["vol_20"] = feat["ret_1"].rolling(20).std()
    if "Volume" in df.columns:
        vol = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0)
        vol_ma20 = vol.rolling(20).mean()
        vol_std20 = vol.rolling(20).std()
        feat["volume_ratio_20"] = (vol / vol_ma20).replace([np.inf, -np.inf], 0.0)
        feat["volume_z_20"] = ((vol - vol_ma20) / vol_std20).replace([np.inf, -np.inf], 0.0)
        value_traded = pd.to_numeric(df["Close"], errors="coerce").fillna(0.0) * vol
        feat["value_traded_log"] = np.log(value_traded + 1.0)
    else:
        feat["volume_ratio_20"] = 0.0
        feat["volume_z_20"] = 0.0
        feat["value_traded_log"] = 0.0
    feat = feat.join(index_returns, how="left")
    feat = feat.join(futures_feature, how="left")
    feat = feat.join(dow_futures_feature, how="left")
    sector_ret = sector_ret_map.get(sector)
    if sector_ret is not None:
        feat = feat.join(sector_ret, how="left")
    else:
        feat["sector_ret_1"] = 0.0

    if "ret_n225" in feat.columns:
        feat["corr_n225_20"] = feat["ret_1"].rolling(20).corr(feat["ret_n225"])
    else:
        feat["corr_n225_20"] = 0.0
    if "ret_dji" in feat.columns:
        feat["corr_dji_20"] = feat["ret_1"].rolling(20).corr(feat["ret_dji"])
    else:
        feat["corr_dji_20"] = 0.0
    if "ret_sp500" in feat.columns:
        feat["corr_sp500_20"] = feat["ret_1"].rolling(20).corr(feat["ret_sp500"])
    else:
        feat["corr_sp500_20"] = 0.0
    feat["corr_n225_20"] = feat["corr_n225_20"].fillna(0.0)
    feat["corr_dji_20"] = feat["corr_dji_20"].fillna(0.0)
    feat["corr_sp500_20"] = feat["corr_sp500_20"].fillna(0.0)
    if "ret_dow_fut" in feat.columns:
        feat["corr_dow_fut_20"] = feat["ret_1"].rolling(20).corr(feat["ret_dow_fut"])
    else:
        feat["corr_dow_fut_20"] = 0.0
    feat["corr_dow_fut_20"] = feat["corr_dow_fut_20"].fillna(0.0)
    feat["volume_ratio_20"] = feat["volume_ratio_20"].fillna(0.0)
    feat["volume_z_20"] = feat["volume_z_20"].fillna(0.0)
    feat["value_traded_log"] = feat["value_traded_log"].fillna(0.0)
    if "ret_dji" not in feat.columns:
        feat["ret_dji"] = 0.0
    feat["ret_dji"] = feat["ret_dji"].fillna(0.0)
    if "fut_nk_night_ret" not in feat.columns:
        feat["fut_nk_night_ret"] = 0.0
    feat["fut_nk_night_ret"] = feat["fut_nk_night_ret"].fillna(0.0)
    if "ret_dow_fut" not in feat.columns:
        feat["ret_dow_fut"] = 0.0
    feat["ret_dow_fut"] = feat["ret_dow_fut"].fillna(0.0)
    if "sector_ret_1" not in feat.columns:
        feat["sector_ret_1"] = 0.0
    feat["sector_ret_1"] = feat["sector_ret_1"].fillna(0.0)
    feat["sector_corr_20"] = feat["ret_1"].rolling(20).corr(feat["sector_ret_1"]).fillna(0.0)
    for col in SECTOR_ONEHOT_COLS:
        feat[col] = 0.0
    feat[sector_to_onehot_col(sector)] = 1.0

    future_prices = pd.concat(
        [df["Close"].shift(-k) for k in range(1, horizon_days + 1)],
        axis=1,
    )
    future_max = future_prices.max(axis=1)
    future_obs = future_prices.notna().sum(axis=1)
    full_window = future_obs == horizon_days

    feat["target"] = pd.NA
    feat.loc[full_window, "target"] = (
        future_max.loc[full_window] >= df["Close"].loc[full_window] * (1.0 + take_profit)
    ).astype(int)
    return feat


def train_global_model(train_df: pd.DataFrame, min_rows: int = 3000) -> RandomForestClassifier:
    train = train_df.dropna(subset=FEATURE_COLS + ["target"])
    if len(train) < min_rows:
        raise ValueError(f"Not enough history rows ({len(train)} < {min_rows})")
    x_train = train[FEATURE_COLS]
    y_train = train["target"].astype(int)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=4,
        min_samples_leaf=5,
        random_state=42,
    )
    model.fit(x_train, y_train)
    return model


def _build_trade_cases(
    raw_df: pd.DataFrame,
    feat: pd.DataFrame,
    model: RandomForestClassifier,
    horizon_days: int,
) -> list[TradeCase]:
    eval_rows = feat.dropna(subset=FEATURE_COLS + ["target"]).copy()
    if eval_rows.empty:
        return []
    eval_rows["tp_probability"] = model.predict_proba(eval_rows[FEATURE_COLS])[:, 1]
    entry_rows = eval_rows

    closes = raw_df["Close"]
    lows = raw_df["Low"] if "Low" in raw_df.columns else raw_df["Close"]
    trades: list[TradeCase] = []
    for dt in entry_rows.index:
        i = closes.index.get_loc(dt)
        close_window = closes.iloc[i + 1 : i + horizon_days + 1]
        if len(close_window) < horizon_days:
            continue
        if i + 1 >= len(lows):
            continue
        trades.append(
            TradeCase(
                signal_close=float(closes.iloc[i]),
                next_day_low=float(lows.iloc[i + 1]),
                future_closes=[float(v) for v in close_window.values],
                tp_probability=float(entry_rows.loc[dt, "tp_probability"]),
                signal_date=pd.Timestamp(dt),
            )
        )
    return trades


def _evaluate_stop_loss(
    trades: list[TradeCase],
    stop_loss_ratio: float,
    take_profit: float,
    horizon_days: int,
) -> tuple[float, float]:
    if not trades:
        return 0.0, 0.0

    wins = 0
    total_return = 0.0
    for trade in trades:
        entry = trade.signal_close
        path = trade.future_closes
        tp_price = entry * (1.0 + take_profit)
        sl_price = entry * (1.0 - stop_loss_ratio)
        exit_price = path[horizon_days - 1]

        for px in path:
            if px >= tp_price:
                exit_price = tp_price
                break
            if px <= sl_price:
                exit_price = sl_price
                break

        ret = exit_price / entry - 1.0
        total_return += ret
        if ret > 0:
            wins += 1

    win_rate = wins / len(trades)
    avg_return = total_return / len(trades)
    return win_rate, avg_return


def _evaluate_limit_entry(
    trades: list[TradeCase],
    limit_entry_ratio: float,
    stop_loss_ratio: float,
    take_profit: float,
    horizon_days: int,
) -> tuple[float, float, int]:
    if not trades:
        return 0.0, 0.0, 0

    wins = 0
    total_return = 0.0
    fills = 0
    for trade in trades:
        entry = trade.signal_close * (1.0 - limit_entry_ratio)
        if trade.next_day_low > entry:
            continue
        fills += 1
        path = trade.future_closes
        tp_price = entry * (1.0 + take_profit)
        sl_price = entry * (1.0 - stop_loss_ratio)
        exit_price = path[horizon_days - 1]

        for px in path:
            if px >= tp_price:
                exit_price = tp_price
                break
            if px <= sl_price:
                exit_price = sl_price
                break

        ret = exit_price / entry - 1.0
        total_return += ret
        if ret > 0:
            wins += 1

    if fills == 0:
        return 0.0, 0.0, 0
    win_rate = wins / fills
    avg_return = total_return / fills
    return win_rate, avg_return, fills


def _evaluate_strategy(
    trades: list[TradeCase],
    threshold: float,
    limit_entry_ratio: float,
    stop_loss_ratio: float,
    take_profit: float,
    horizon_days: int,
) -> tuple[float, float, float, int, int]:
    filtered = [t for t in trades if t.tp_probability >= threshold]
    candidates = len(filtered)
    if candidates == 0:
        return 0.0, 0.0, 0.0, 0, 0

    win_rate, avg_return_fill, fills = _evaluate_limit_entry(
        trades=filtered,
        limit_entry_ratio=limit_entry_ratio,
        stop_loss_ratio=stop_loss_ratio,
        take_profit=take_profit,
        horizon_days=horizon_days,
    )
    if candidates == 0:
        avg_return_signal = 0.0
    else:
        # Expected return per signal when unfilled orders are treated as 0 return.
        avg_return_signal = avg_return_fill * (fills / candidates) if fills > 0 else 0.0
    return win_rate, avg_return_fill, avg_return_signal, fills, candidates


def _evaluate_strategy_with_risk(
    trades: list[TradeCase],
    threshold: float,
    limit_entry_ratio: float,
    stop_loss_ratio: float,
    take_profit: float,
    horizon_days: int,
) -> tuple[float, float, float, int, int, float, float]:
    filtered = [t for t in trades if t.tp_probability >= threshold]
    candidates = len(filtered)
    if candidates == 0:
        return 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0

    filled_returns: list[tuple[pd.Timestamp, float]] = []
    wins = 0
    total_return = 0.0
    for trade in filtered:
        entry = trade.signal_close * (1.0 - limit_entry_ratio)
        if trade.next_day_low > entry:
            continue
        path = trade.future_closes
        tp_price = entry * (1.0 + take_profit)
        sl_price = entry * (1.0 - stop_loss_ratio)
        exit_price = path[horizon_days - 1]
        for px in path:
            if px >= tp_price:
                exit_price = tp_price
                break
            if px <= sl_price:
                exit_price = sl_price
                break
        ret = exit_price / entry - 1.0
        filled_returns.append((trade.signal_date, ret))
        total_return += ret
        if ret > 0:
            wins += 1

    fills = len(filled_returns)
    fill_rate = fills / candidates if candidates > 0 else 0.0
    if fills == 0:
        return 0.0, 0.0, 0.0, 0, candidates, fill_rate, 0.0

    win_rate = wins / fills
    avg_return_fill = total_return / fills
    avg_return_signal = avg_return_fill * fill_rate

    filled_returns.sort(key=lambda x: x[0])
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for _, r in filled_returns:
        equity *= 1.0 + r
        if equity > peak:
            peak = equity
        drawdown = (peak - equity) / peak if peak > 0 else 0.0
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    return (
        win_rate,
        avg_return_fill,
        avg_return_signal,
        fills,
        candidates,
        fill_rate,
        max_drawdown,
    )


def _fill_stats_for_threshold_limit(
    trades: list[TradeCase],
    threshold: float,
    limit_entry_ratio: float,
) -> tuple[int, int, float]:
    filtered = [t for t in trades if t.tp_probability >= threshold]
    candidates = len(filtered)
    if candidates == 0:
        return 0, 0, 0.0

    fill_dates: list[pd.Timestamp] = []
    for t in filtered:
        entry = t.signal_close * (1.0 - limit_entry_ratio)
        if t.next_day_low <= entry:
            fill_dates.append(t.signal_date)

    fills = len(fill_dates)
    if fills == 0:
        return candidates, fills, 0.0

    dates = pd.to_datetime(pd.Series(fill_dates)).dt.to_period("M")
    fills_per_month = dates.value_counts()
    avg_monthly_fills = float(fills_per_month.mean()) if len(fills_per_month) > 0 else 0.0
    return candidates, fills, avg_monthly_fills


def optimize_stop_loss_ratio(
    all_symbol_data: dict[str, pd.DataFrame],
    feature_map: dict[str, pd.DataFrame],
    model: RandomForestClassifier,
    threshold: float,
    horizon_days: int,
    take_profit: float,
    min_sl: float = 0.01,
    max_sl: float = 0.15,
    step: float = 0.005,
) -> StopLossTuningResult:
    trades: list[TradeCase] = []
    for symbol, raw_df in all_symbol_data.items():
        feat = feature_map.get(symbol)
        if feat is None:
            continue
        trades.extend(
            _build_trade_cases(
                raw_df=raw_df,
                feat=feat,
                model=model,
                horizon_days=horizon_days,
            )
        )

    if not trades:
        return StopLossTuningResult(stop_loss_ratio=0.05, win_rate=0.0, avg_return=0.0, trades=0)

    best: StopLossTuningResult | None = None
    sl = min_sl
    while sl <= max_sl + 1e-9:
        win_rate, avg_return = _evaluate_stop_loss(
            trades=trades,
            stop_loss_ratio=sl,
            take_profit=take_profit,
            horizon_days=horizon_days,
        )
        candidate = StopLossTuningResult(
            stop_loss_ratio=sl,
            win_rate=win_rate,
            avg_return=avg_return,
            trades=len(trades),
        )
        if best is None or candidate.win_rate > best.win_rate or (
            candidate.win_rate == best.win_rate and candidate.avg_return > best.avg_return
        ):
            best = candidate
        sl += step

    if best is None:
        return StopLossTuningResult(stop_loss_ratio=0.05, win_rate=0.0, avg_return=0.0, trades=0)
    return best


def optimize_limit_entry_ratio(
    all_symbol_data: dict[str, pd.DataFrame],
    feature_map: dict[str, pd.DataFrame],
    model: RandomForestClassifier,
    threshold: float,
    horizon_days: int,
    take_profit: float,
    stop_loss_ratio: float,
    min_limit: float = 0.0,
    max_limit: float = 0.05,
    step: float = 0.0025,
    min_fill_rate: float = 0.15,
) -> LimitTuningResult:
    trades: list[TradeCase] = []
    for symbol, raw_df in all_symbol_data.items():
        feat = feature_map.get(symbol)
        if feat is None:
            continue
        trades.extend(
            _build_trade_cases(
                raw_df=raw_df,
                feat=feat,
                model=model,
                horizon_days=horizon_days,
            )
        )

    if not trades:
        return LimitTuningResult(
            limit_entry_ratio=0.0,
            win_rate=0.0,
            avg_return=0.0,
            fills=0,
            candidates=0,
        )

    best: LimitTuningResult | None = None
    limit_ratio = min_limit
    min_fills = max(1, int(len(trades) * min_fill_rate))

    while limit_ratio <= max_limit + 1e-9:
        win_rate, avg_return, fills = _evaluate_limit_entry(
            trades=trades,
            limit_entry_ratio=limit_ratio,
            stop_loss_ratio=stop_loss_ratio,
            take_profit=take_profit,
            horizon_days=horizon_days,
        )
        if fills >= min_fills:
            candidate = LimitTuningResult(
                limit_entry_ratio=limit_ratio,
                win_rate=win_rate,
                avg_return=avg_return,
                fills=fills,
                candidates=len(trades),
            )
            if best is None or candidate.win_rate > best.win_rate or (
                candidate.win_rate == best.win_rate and candidate.avg_return > best.avg_return
            ):
                best = candidate
        limit_ratio += step

    if best is None:
        win_rate, avg_return, fills = _evaluate_limit_entry(
            trades=trades,
            limit_entry_ratio=0.0,
            stop_loss_ratio=stop_loss_ratio,
            take_profit=take_profit,
            horizon_days=horizon_days,
        )
        return LimitTuningResult(
            limit_entry_ratio=0.0,
            win_rate=win_rate,
            avg_return=avg_return,
            fills=fills,
            candidates=len(trades),
        )
    return best


def optimize_strategy_for_return(
    all_symbol_data: dict[str, pd.DataFrame],
    feature_map: dict[str, pd.DataFrame],
    model: RandomForestClassifier,
    horizon_days: int,
    take_profit: float,
    threshold_values: list[float] | None = None,
    stop_loss_values: list[float] | None = None,
    limit_values: list[float] | None = None,
    min_fill_rate: float = 0.10,
    min_monthly_fills: float = 0.0,
    max_drawdown_limit: float = 1.0,
) -> StrategyTuningResult:
    if threshold_values is None:
        threshold_values = [round(x, 2) for x in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]]
    if stop_loss_values is None:
        stop_loss_values = [round(0.03 + i * 0.01, 2) for i in range(18)]  # 3%..20%
    if limit_values is None:
        # Negative limit means placing a buy limit above previous close.
        # Search range: -3.0% .. +5.0% with 0.5% steps.
        limit_values = [round(-0.03 + i * 0.005, 4) for i in range(17)]

    all_trades: list[TradeCase] = []
    for symbol, raw_df in all_symbol_data.items():
        feat = feature_map.get(symbol)
        if feat is None:
            continue
        all_trades.extend(
            _build_trade_cases(
                raw_df=raw_df,
                feat=feat,
                model=model,
                horizon_days=horizon_days,
            )
        )

    if not all_trades:
        return StrategyTuningResult(
            threshold=0.60,
            stop_loss_ratio=0.10,
            limit_entry_ratio=0.0,
            win_rate=0.0,
            avg_return_fill=0.0,
            avg_return_signal=0.0,
            fills=0,
            candidates=0,
            fill_rate=0.0,
            max_drawdown=0.0,
            avg_monthly_fills=0.0,
        )

    best: StrategyTuningResult | None = None
    for threshold in threshold_values:
        candidates = sum(1 for t in all_trades if t.tp_probability >= threshold)
        if candidates == 0:
            continue
        for sl in stop_loss_values:
            for limit_ratio in limit_values:
                cand_count, fills_for_constraint, avg_monthly_fills = _fill_stats_for_threshold_limit(
                    trades=all_trades,
                    threshold=threshold,
                    limit_entry_ratio=limit_ratio,
                )
                if cand_count == 0:
                    continue
                req_fills = max(1, int(cand_count * min_fill_rate))
                if fills_for_constraint < req_fills:
                    continue
                if avg_monthly_fills < min_monthly_fills:
                    continue

                (
                    win_rate,
                    avg_fill,
                    avg_signal,
                    fills,
                    _,
                    fill_rate,
                    max_drawdown,
                ) = _evaluate_strategy_with_risk(
                    trades=all_trades,
                    threshold=threshold,
                    limit_entry_ratio=limit_ratio,
                    stop_loss_ratio=sl,
                    take_profit=take_profit,
                    horizon_days=horizon_days,
                )
                if max_drawdown > max_drawdown_limit:
                    continue
                cand = StrategyTuningResult(
                    threshold=threshold,
                    stop_loss_ratio=sl,
                    limit_entry_ratio=limit_ratio,
                    win_rate=win_rate,
                    avg_return_fill=avg_fill,
                    avg_return_signal=avg_signal,
                    fills=fills,
                    candidates=candidates,
                    fill_rate=fill_rate,
                    max_drawdown=max_drawdown,
                    avg_monthly_fills=avg_monthly_fills,
                )
                if best is None or (
                    cand.avg_return_signal > best.avg_return_signal
                    or (
                        cand.avg_return_signal == best.avg_return_signal
                        and cand.max_drawdown < best.max_drawdown
                    )
                    or (
                        cand.avg_return_signal == best.avg_return_signal
                        and cand.max_drawdown == best.max_drawdown
                        and cand.fill_rate > best.fill_rate
                    )
                    or (
                        cand.avg_return_signal == best.avg_return_signal
                        and cand.max_drawdown == best.max_drawdown
                        and cand.fill_rate == best.fill_rate
                        and cand.win_rate > best.win_rate
                    )
                ):
                    best = cand

    if best is None:
        win_rate, avg_fill, avg_signal, fills, candidates = _evaluate_strategy(
            trades=all_trades,
            threshold=0.60,
            limit_entry_ratio=0.0,
            stop_loss_ratio=0.10,
            take_profit=take_profit,
            horizon_days=horizon_days,
        )
        return StrategyTuningResult(
            threshold=0.60,
            stop_loss_ratio=0.10,
            limit_entry_ratio=0.0,
            win_rate=win_rate,
            avg_return_fill=avg_fill,
            avg_return_signal=avg_signal,
            fills=fills,
            candidates=candidates,
            fill_rate=(fills / candidates) if candidates > 0 else 0.0,
            max_drawdown=0.0,
            avg_monthly_fills=0.0,
        )
    return best


def select_diversified_buys(
    buy_candidates: list[SignalResult],
    all_symbol_data: dict[str, pd.DataFrame],
    max_corr: float,
    corr_lookback_days: int,
    max_positions: int,
    max_per_sector: int,
) -> list[SignalResult]:
    if not buy_candidates:
        return []

    close_map: dict[str, pd.Series] = {}
    for c in buy_candidates:
        raw_df = all_symbol_data.get(c.symbol)
        if raw_df is None:
            raw_df = all_symbol_data.get(c.symbol.removesuffix(".T"))
        if raw_df is None:
            continue
        close_map[c.symbol] = raw_df["Close"].pct_change().dropna().tail(corr_lookback_days)

    corr: pd.DataFrame | None = None
    if close_map:
        returns_df = pd.concat(close_map, axis=1).dropna()
        if not returns_df.empty:
            corr = returns_df.corr()

    ordered = sorted(buy_candidates, key=lambda x: x.tp_probability, reverse=True)
    selected: list[SignalResult] = []
    sector_count: dict[str, int] = {}

    for cand in ordered:
        if max_positions > 0 and len(selected) >= max_positions:
            break

        if max_per_sector > 0:
            if sector_count.get(cand.sector, 0) >= max_per_sector:
                continue

        corr_ok = True
        if corr is not None and cand.symbol in corr.index:
            for picked in selected:
                if picked.symbol in corr.columns:
                    value = corr.loc[cand.symbol, picked.symbol]
                    if pd.notna(value) and abs(float(value)) > max_corr:
                        corr_ok = False
                        break
        if not corr_ok:
            continue

        selected.append(cand)
        sector_count[cand.sector] = sector_count.get(cand.sector, 0) + 1

    return selected


def decide_buy(tp_probability: float, threshold: float) -> bool:
    return tp_probability >= threshold


def analyze_symbol(
    symbol: str,
    raw_df: pd.DataFrame,
    feat: pd.DataFrame,
    model: RandomForestClassifier,
    threshold: float,
    stop_loss_ratio: float,
    limit_entry_ratio: float,
) -> SignalResult:
    latest_feature = feat.dropna(subset=FEATURE_COLS).iloc[-1]
    x_last = latest_feature[FEATURE_COLS].to_frame().T
    tp_probability = float(model.predict_proba(x_last)[0][1])
    buy = decide_buy(tp_probability, threshold)

    last_date = pd.Timestamp(raw_df.index[-1])
    last_close = float(raw_df["Close"].iloc[-1])
    trade_date = next_business_day_jp(last_date.date())

    return SignalResult(
        symbol=normalize_symbol(symbol),
        last_date=last_date,
        last_close=last_close,
        buy=buy,
        tp_probability=tp_probability,
        stop_loss_ratio=stop_loss_ratio,
        limit_entry_ratio=limit_entry_ratio,
        trade_date=trade_date,
    )


def save_price_data(symbol: str, raw_df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized = normalize_symbol(symbol)
    out_path = output_dir / f"{normalized.replace('.', '_')}.csv"
    raw_df.to_csv(out_path, index_label="Date")
    return out_path


def notify(result: SignalResult) -> None:
    limit_price = result.last_close * (1.0 - result.limit_entry_ratio)
    if result.buy:
        sl_price = result.last_close * (1.0 - result.stop_loss_ratio)
        print(
            f"[BUY] {result.symbol} | signal_date={result.last_date.date()} trade_date={result.trade_date} "
            f"close={result.last_close:.2f} tp_prob={result.tp_probability:.2%} "
            f"lmt={result.limit_entry_ratio:.2%} lmt_price={limit_price:.2f} "
            f"sl={result.stop_loss_ratio:.2%} sl_price={sl_price:.2f}"
        )
    else:
        print(
            f"[HOLD] {result.symbol} | signal_date={result.last_date.date()} trade_date={result.trade_date} "
            f"close={result.last_close:.2f} tp_prob={result.tp_probability:.2%} "
            f"lmt={result.limit_entry_ratio:.2%} lmt_price={limit_price:.2f} "
            f"sl={result.stop_loss_ratio:.2%}"
        )


def read_watchlist(path: Path) -> list[str]:
    df = pd.read_csv(path)
    if "symbol" not in df.columns:
        raise ValueError(f"'symbol' column is required in {path}")
    return [str(v) for v in df["symbol"].dropna().tolist()]


def read_watchlist_groups(path: Path) -> dict[str, list[str]]:
    df = pd.read_csv(path)
    if "symbol" not in df.columns:
        raise ValueError(f"'symbol' column is required in {path}")

    if "group" not in df.columns:
        symbols = [str(v) for v in df["symbol"].dropna().tolist()]
        return {"default": symbols}

    out: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        symbol = str(row["symbol"]).strip()
        if not symbol:
            continue
        group = str(row["group"]).strip() if pd.notna(row["group"]) else "default"
        if not group:
            group = "default"
        out.setdefault(group, []).append(symbol)
    return out


def load_sector_master(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    if not path.exists():
        raise ValueError(f"sector master file not found: {path}")
    df = pd.read_csv(path)
    required = {"symbol", "sector"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"sector master must include columns: {required}")
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        sym_raw = str(row["symbol"]).strip()
        sec = str(row["sector"]).strip()
        if not sym_raw or not sec:
            continue
        out[normalize_symbol(sym_raw)] = sec
    return out


def load_symbol_universe(watchlist_path: Path, universe_path: Path | None) -> list[str]:
    if universe_path is not None and universe_path.exists():
        df = pd.read_csv(universe_path)
        if "symbol" not in df.columns:
            raise ValueError(f"'symbol' column is required in {universe_path}")
        return [normalize_symbol(str(v)) for v in df["symbol"].dropna().tolist()]

    grouped = read_watchlist_groups(watchlist_path)
    symbols: list[str] = []
    for vals in grouped.values():
        symbols.extend([normalize_symbol(s) for s in vals])
    # preserve order while removing duplicates
    return list(dict.fromkeys(symbols))


def detect_bullish_sectors(
    period: str,
    short_window: int = 5,
    long_window: int = 20,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for sector, ticker in SECTOR_INDEX_TICKER_MAP.items():
        try:
            df = _download_yahoo_ticker(ticker=ticker, period=period, retries=3)
            if df.empty or "Close" not in df.columns:
                continue
            close = pd.to_numeric(df["Close"], errors="coerce").dropna()
            if len(close) < max(short_window, long_window) + 1:
                continue
            ma_s = close.rolling(short_window).mean().iloc[-1]
            ma_l = close.rolling(long_window).mean().iloc[-1]
            c = close.iloc[-1]
            ret_s = close.pct_change(short_window).iloc[-1]
            # Bullish sector signal: short trend and level are both positive.
            is_bull = bool(c > ma_l and ma_s > ma_l and ret_s > 0.0)
            if not is_bull:
                continue
            score = float((c / ma_l - 1.0) + ret_s)
            scores[sector] = score
        except Exception:
            continue
    return scores


def build_local_sector_return_map(
    symbol_data: dict[str, pd.DataFrame],
    symbol_sectors: dict[str, str],
    min_symbols_per_sector: int = 3,
) -> dict[str, pd.Series]:
    frames: list[pd.Series] = []
    for sym, df in symbol_data.items():
        if df is None or df.empty or "Close" not in df.columns:
            continue
        sector = symbol_sectors.get(sym, "UNKNOWN")
        if sector == "UNKNOWN":
            continue
        close = pd.to_numeric(df["Close"], errors="coerce")
        ret = close.pct_change().rename(sym)
        frames.append(ret)
    if not frames:
        return {}

    ret_df = pd.concat(frames, axis=1)
    sector_map: dict[str, list[str]] = {}
    for s in ret_df.columns:
        sec = symbol_sectors.get(s, "UNKNOWN")
        if sec == "UNKNOWN":
            continue
        sector_map.setdefault(sec, []).append(s)

    out: dict[str, pd.Series] = {}
    for sec, cols in sector_map.items():
        valid_cols = [c for c in cols if c in ret_df.columns]
        if len(valid_cols) < min_symbols_per_sector:
            continue
        out[sec] = ret_df[valid_cols].mean(axis=1).rename("sector_ret_1")
    return out


def detect_bullish_sectors_from_local_returns(
    sector_ret_map: dict[str, pd.Series],
    short_window: int = 5,
    long_window: int = 20,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for sector, ret_ser in sector_ret_map.items():
        ser = pd.to_numeric(ret_ser, errors="coerce").dropna()
        if len(ser) < max(short_window, long_window) + 5:
            continue
        idx = (1.0 + ser.fillna(0.0)).cumprod()
        ma_s = idx.rolling(short_window).mean().iloc[-1]
        ma_l = idx.rolling(long_window).mean().iloc[-1]
        c = idx.iloc[-1]
        ret_s = idx.pct_change(short_window).iloc[-1]
        is_bull = bool(c > ma_l and ma_s > ma_l and ret_s > 0.0)
        if not is_bull:
            continue
        scores[sector] = float((c / ma_l - 1.0) + ret_s)
    return scores


def rank_symbols_for_sector_expansion(
    symbols: list[str],
    data_map: dict[str, pd.DataFrame],
) -> list[str]:
    scored: list[tuple[str, float]] = []
    for s in symbols:
        df = data_map.get(s)
        if df is None or df.empty or "Close" not in df.columns:
            continue
        c = pd.to_numeric(df["Close"], errors="coerce").dropna()
        if len(c) < 21:
            continue
        ret20 = float(c.pct_change(20).iloc[-1])
        ret5 = float(c.pct_change(5).iloc[-1])
        if np.isnan(ret20) or np.isnan(ret5):
            continue
        scored.append((s, ret20 + ret5))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scored]


def run(
    watchlist_path: Path,
    threshold: float,
    period: str,
    output_dir: Path,
    horizon_days: int,
    take_profit: float,
    max_corr: float,
    corr_lookback_days: int,
    max_positions: int,
    max_per_sector: int,
    include_sector: bool,
    expected_jst_hour: int,
    discord_webhook_url: str,
    min_fill_rate: float,
    min_monthly_fills: float,
    max_drawdown_limit: float,
    min_buy_signals: int,
    fallback_min_threshold: float,
    sector_extra_enabled: bool,
    sector_extra_max_symbols: int,
    sector_extra_universe_path: Path | None,
    sector_signal_short_window: int,
    sector_signal_long_window: int,
    sector_signal_source: str,
    sector_ret_source: str,
    sector_master_path: Path | None,
    sector_extra_min_prob: float,
    sector_extra_min_value: float,
    sector_extra_top_percent: float,
) -> int:
    cache_dir = Path("data") / "yf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_dir_abs = str(cache_dir.resolve())
    yf.cache.set_cache_location(cache_dir_abs)
    yf.set_tz_cache_location(cache_dir_abs)

    grouped_symbols = read_watchlist_groups(watchlist_path)
    if not grouped_symbols:
        raise ValueError(f"Watchlist is empty: {watchlist_path}")

    now_jst = datetime.now(ZoneInfo("Asia/Tokyo"))
    print(f"[TIME] jst_now={now_jst.strftime('%Y-%m-%d %H:%M:%S')} expected_hour={expected_jst_hour}")
    if now_jst.hour != expected_jst_hour:
        print("[WARN] Running outside the expected JST hour for night-session feature capture.")

    index_returns = download_index_returns(period=period)
    futures_feature = download_nikkei_futures_night_feature()
    dow_futures_feature = download_dow_futures_feature(period=period)
    symbol_universe = load_symbol_universe(watchlist_path=watchlist_path, universe_path=sector_extra_universe_path)
    downloaded_data_cache: dict[str, pd.DataFrame] = {}
    sector_cache: dict[str, str] = load_sector_master(sector_master_path)
    if sector_cache:
        print(f"[SECTOR_MASTER] loaded={len(sector_cache)} from={sector_master_path}")
    local_sector_ret_map_global: dict[str, pd.Series] = {}

    if sector_signal_source == "local" or sector_ret_source == "local":
        for sym in symbol_universe:
            if sym in downloaded_data_cache:
                continue
            try:
                raw_df = download_daily_data(sym, period=period)
                downloaded_data_cache[sym] = raw_df
            except Exception:
                continue
        unresolved_global = [s for s in downloaded_data_cache.keys() if s not in sector_cache]
        if unresolved_global:
            sector_cache.update(fetch_symbol_sectors(unresolved_global))
        local_sector_ret_map_global = build_local_sector_return_map(
            symbol_data=downloaded_data_cache,
            symbol_sectors=sector_cache,
            min_symbols_per_sector=3,
        )

    if sector_signal_source == "local":
        sector_bull_scores = detect_bullish_sectors_from_local_returns(
            sector_ret_map=local_sector_ret_map_global,
            short_window=sector_signal_short_window,
            long_window=sector_signal_long_window,
        )
    else:
        sector_bull_scores = detect_bullish_sectors(
            period=period,
            short_window=sector_signal_short_window,
            long_window=sector_signal_long_window,
        )
    bullish_sectors = set(sector_bull_scores.keys())
    print(
        f"[SECTOR_SIGNAL] bullish={len(bullish_sectors)} "
        f"extra_enabled={sector_extra_enabled} extra_max={sector_extra_max_symbols} "
        f"signal_source={sector_signal_source} ret_source={sector_ret_source}"
    )
    print(
        f"[FUT] rows_nk={len(futures_feature)} ticker_nk=NKD=F feature_nk=fut_nk_night_ret "
        f"rows_dow={len(dow_futures_feature)} ticker_dow=YM=F feature_dow=ret_dow_fut"
    )
    discord_lines = [f"Stock ML Signal ({now_jst.strftime('%Y-%m-%d %H:%M JST')})"]

    for group_name, symbols in grouped_symbols.items():
        print(f"[GROUP] start={group_name} symbols={len(symbols)}")
        base_symbols_norm = [normalize_symbol(s) for s in symbols]
        symbol_data_norm: dict[str, pd.DataFrame] = {}
        for normalized in base_symbols_norm:
            if normalized in downloaded_data_cache:
                symbol_data_norm[normalized] = downloaded_data_cache[normalized]
                continue
            try:
                raw_df = download_daily_data(normalized, period=period)
                saved_path = save_price_data(symbol=normalized, raw_df=raw_df, output_dir=output_dir)
                downloaded_data_cache[normalized] = raw_df
                symbol_data_norm[normalized] = raw_df
                print(f"[DATA][{group_name}] {normalized} -> {saved_path}")
            except Exception as exc:
                print(f"[ERROR][{group_name}] {normalized}: {exc}")

        if not symbol_data_norm:
            print(f"[GROUP] skip={group_name} reason=no_data")
            discord_lines.append(f"{group_name}: no available price data")
            continue

        # Optional dynamic expansion:
        # add up to N symbols from bullish sectors; if bullish signal disappears,
        # symbols are naturally excluded in subsequent runs.
        extra_symbols: list[str] = []
        if sector_extra_enabled and sector_extra_max_symbols > 0 and bullish_sectors:
            pool = [s for s in symbol_universe if s not in base_symbols_norm]
            unresolved_pool = [s for s in pool if s not in sector_cache]
            if unresolved_pool:
                sector_cache.update(fetch_symbol_sectors(unresolved_pool))
            pool_bull = [s for s in pool if sector_cache.get(s, "UNKNOWN") in bullish_sectors]

            for s in pool_bull:
                if s in downloaded_data_cache:
                    symbol_data_norm[s] = downloaded_data_cache[s]
                    continue
                try:
                    raw_df = download_daily_data(s, period=period)
                    saved_path = save_price_data(symbol=s, raw_df=raw_df, output_dir=output_dir)
                    downloaded_data_cache[s] = raw_df
                    symbol_data_norm[s] = raw_df
                    print(f"[DATA][{group_name}][EXTRA] {s} -> {saved_path}")
                except Exception:
                    continue

            ranked_pool = rank_symbols_for_sector_expansion(
                symbols=[s for s in pool_bull if s in symbol_data_norm],
                data_map=symbol_data_norm,
            )
            extra_symbols = ranked_pool[:sector_extra_max_symbols]
            print(
                f"[SECTOR_EXTRA][{group_name}] pool={len(pool_bull)} "
                f"selected={len(extra_symbols)} bullish_sectors={len(bullish_sectors)}"
            )

        active_symbols = list(dict.fromkeys(base_symbols_norm + extra_symbols))

        unresolved = [s for s in symbol_data_norm.keys() if s not in sector_cache]
        if unresolved:
            sector_cache.update(fetch_symbol_sectors(unresolved))
        symbol_sectors = {s: sector_cache.get(s, "UNKNOWN") for s in symbol_data_norm.keys()}
        unique_sectors = set(symbol_sectors.values())
        if sector_ret_source == "local":
            # Recompute local sector map with latest downloaded set to reflect all fetched symbols.
            local_sector_ret_map_global = build_local_sector_return_map(
                symbol_data=downloaded_data_cache,
                symbol_sectors=sector_cache,
                min_symbols_per_sector=3,
            )
            sector_ret_map = {sec: local_sector_ret_map_global.get(sec) for sec in unique_sectors if sec in local_sector_ret_map_global}
        else:
            sector_ret_map = download_sector_index_returns(period=period, sectors=unique_sectors)
        print(
            f"[SECTOR][{group_name}] symbols={len(symbol_sectors)} unique={len(unique_sectors)} "
            f"index_loaded={len(sector_ret_map)}"
        )

        feature_map: dict[str, pd.DataFrame] = {}
        training_frames: list[pd.DataFrame] = []
        for symbol_norm, raw_df in symbol_data_norm.items():
            sector = symbol_sectors.get(symbol_norm, "UNKNOWN")
            feat = build_features(
                df=raw_df,
                index_returns=index_returns,
                futures_feature=futures_feature,
                dow_futures_feature=dow_futures_feature,
                sector=sector,
                sector_ret_map=sector_ret_map,
                horizon_days=horizon_days,
                take_profit=take_profit,
            )
            feature_map[symbol_norm] = feat
            labeled = feat.dropna(subset=FEATURE_COLS + ["target"]).copy()
            if not labeled.empty:
                labeled["symbol"] = symbol_norm
                training_frames.append(labeled)

        if not training_frames:
            print(f"[GROUP] skip={group_name} reason=no_training_rows")
            discord_lines.append(f"{group_name}: no training rows")
            continue

        train_df = pd.concat(training_frames, axis=0)
        model = train_global_model(train_df)
        print(
            f"[ML][{group_name}] trained_rows={len(train_df)} symbols={train_df['symbol'].nunique()} "
            f"features={len(FEATURE_COLS)}"
        )

        strategy = optimize_strategy_for_return(
            all_symbol_data=symbol_data_norm,
            feature_map=feature_map,
            model=model,
            horizon_days=horizon_days,
            take_profit=take_profit,
            threshold_values=[x for x in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75] if x >= threshold],
            min_fill_rate=min_fill_rate,
            min_monthly_fills=min_monthly_fills,
            max_drawdown_limit=max_drawdown_limit,
        )
        print(
            f"[OPTRET][{group_name}] threshold={strategy.threshold:.2f} "
            f"stop_loss={strategy.stop_loss_ratio:.2%} limit={strategy.limit_entry_ratio:.2%} "
            f"win_rate={strategy.win_rate:.2%} avg_fill={strategy.avg_return_fill:.2%} "
            f"avg_signal={strategy.avg_return_signal:.2%} fills={strategy.fills}/{strategy.candidates} "
            f"fill_rate={strategy.fill_rate:.2%} max_dd={strategy.max_drawdown:.2%} "
            f"constraints(fill_rate>={min_fill_rate:.2%}, monthly_fills>={min_monthly_fills:.2f}, "
            f"max_dd<={max_drawdown_limit:.2%})"
        )

        all_results: list[SignalResult] = []
        for symbol in active_symbols:
            try:
                normalized = normalize_symbol(symbol)
                if normalized not in symbol_data_norm:
                    continue
                raw_df = symbol_data_norm[normalized]
                feat = feature_map[normalized]
                result = analyze_symbol(
                    symbol=symbol,
                    raw_df=raw_df,
                    feat=feat,
                    model=model,
                    threshold=strategy.threshold,
                    stop_loss_ratio=strategy.stop_loss_ratio,
                    limit_entry_ratio=strategy.limit_entry_ratio,
                )
                if include_sector:
                    result.sector = symbol_sectors.get(normalized, "UNKNOWN")
                all_results.append(result)
            except Exception as exc:
                print(f"[ERROR][{group_name}] {symbol}: {exc}")

        effective_threshold = strategy.threshold
        if min_buy_signals > 0:
            current_buys = sum(1 for r in all_results if r.tp_probability >= effective_threshold)
            if current_buys < min_buy_signals:
                t = effective_threshold
                while t > fallback_min_threshold and current_buys < min_buy_signals:
                    t = round(max(fallback_min_threshold, t - 0.02), 2)
                    current_buys = sum(1 for r in all_results if r.tp_probability >= t)
                effective_threshold = t
                print(
                    f"[FALLBACK][{group_name}] threshold={strategy.threshold:.2f}->{effective_threshold:.2f} "
                    f"buy_signals={current_buys}"
                )

        extra_symbol_set = set(extra_symbols)
        allowed_extra_symbols = set(extra_symbols)
        if sector_extra_enabled and extra_symbols:
            filtered_extra: list[SignalResult] = []
            for r in all_results:
                if r.symbol not in extra_symbol_set:
                    continue
                raw_df = symbol_data_norm.get(r.symbol)
                if raw_df is None or raw_df.empty or "Close" not in raw_df.columns:
                    continue
                close_latest = float(pd.to_numeric(raw_df["Close"], errors="coerce").iloc[-1])
                if "Volume" in raw_df.columns:
                    vol_latest = float(pd.to_numeric(raw_df["Volume"], errors="coerce").fillna(0.0).iloc[-1])
                else:
                    vol_latest = 0.0
                value_traded = close_latest * vol_latest
                if r.tp_probability < sector_extra_min_prob:
                    continue
                if value_traded < sector_extra_min_value:
                    continue
                filtered_extra.append(r)

            filtered_extra.sort(key=lambda x: x.tp_probability, reverse=True)
            if filtered_extra and 0.0 < sector_extra_top_percent < 100.0:
                keep_n = max(1, int(np.ceil(len(filtered_extra) * (sector_extra_top_percent / 100.0))))
                filtered_extra = filtered_extra[:keep_n]
            allowed_extra_symbols = {r.symbol for r in filtered_extra}
            print(
                f"[SECTOR_EXTRA_FILTER][{group_name}] before={len(extra_symbols)} "
                f"after={len(allowed_extra_symbols)} min_prob={sector_extra_min_prob:.2f} "
                f"min_value={sector_extra_min_value:.0f} top%={sector_extra_top_percent:.1f}"
            )

        for r in all_results:
            r.buy = r.tp_probability >= effective_threshold
            if r.symbol in extra_symbol_set and r.symbol not in allowed_extra_symbols:
                r.buy = False
            notify(r)

        buy_candidates = [r for r in all_results if r.buy]
        selected = select_diversified_buys(
            buy_candidates=buy_candidates,
            all_symbol_data=symbol_data_norm,
            max_corr=max_corr,
            corr_lookback_days=corr_lookback_days,
            max_positions=max_positions,
            max_per_sector=max_per_sector,
        )
        print(
            f"[DIV][{group_name}] candidates={len(buy_candidates)} selected={len(selected)} "
            f"max_corr={max_corr:.2f} lookback={corr_lookback_days}"
        )
        for r in selected:
            print(f"[PICK][{group_name}] {r.symbol} tp_prob={r.tp_probability:.2%} sector={r.sector}")

        sig_date = all_results[0].last_date.date() if all_results else "N/A"
        trade_date = all_results[0].trade_date if all_results else "N/A"
        discord_lines.append(
            f"{group_name}: signal_date={sig_date} trade_date={trade_date} "
            f"thr={strategy.threshold:.2f}->{effective_threshold:.2f} lmt={strategy.limit_entry_ratio:.2%} "
            f"sl={strategy.stop_loss_ratio:.2%} exp={strategy.avg_return_signal:.2%} "
            f"fill={strategy.fill_rate:.2%} mdd={strategy.max_drawdown:.2%} extra={len(extra_symbols)} "
            f"win={strategy.win_rate:.2%} picks={len(selected)}"
        )
        if selected:
            discord_lines.extend(
                [f"  - {r.symbol} tp_prob={r.tp_probability:.2%} sector={r.sector}" for r in selected]
            )
        else:
            discord_lines.append("  - No BUY picks for next business day")

    if discord_webhook_url:
        try:
            send_discord_webhook(discord_webhook_url, "\n".join(discord_lines))
            print("[DISCORD] notification sent.")
        except Exception as exc:
            print(f"[ERROR] Discord notification failed: {exc}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download JP stock closes, train a trend model, and emit buy notifications."
    )
    parser.add_argument(
        "--watchlist",
        type=Path,
        default=Path("watchlist.csv"),
        help="CSV file path. Must include a 'symbol' column.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.60,
        help="Buy threshold for P(up) from the model.",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="max",
        help="History period used for training data (e.g. 5y, 20y, max). Default: max.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "prices",
        help="Directory to save fetched OHLCV CSV files.",
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=20,
        help="Maximum holding days for take-profit evaluation.",
    )
    parser.add_argument(
        "--take-profit",
        type=float,
        default=0.05,
        help="Take-profit ratio (0.05 means +5%% within horizon days).",
    )
    parser.add_argument(
        "--max-corr",
        type=float,
        default=0.70,
        help="Maximum allowed absolute pairwise return correlation among selected buys.",
    )
    parser.add_argument(
        "--corr-lookback-days",
        type=int,
        default=252,
        help="Lookback window for correlation calculation (trading days).",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=20,
        help="Maximum number of diversified picks (0 means unlimited).",
    )
    parser.add_argument(
        "--max-per-sector",
        type=int,
        default=3,
        help="Maximum picks per sector (0 means unlimited).",
    )
    parser.add_argument(
        "--no-sector",
        action="store_true",
        help="Disable sector retrieval from Yahoo Finance.",
    )
    parser.add_argument(
        "--expected-jst-hour",
        type=int,
        default=23,
        help="Expected JST hour for running this script (for night-session workflow).",
    )
    parser.add_argument(
        "--discord-webhook-url",
        type=str,
        default=DEFAULT_DISCORD_WEBHOOK_URL,
        help="Discord webhook URL for sending next-day signal notifications.",
    )
    parser.add_argument(
        "--min-fill-rate",
        type=float,
        default=0.10,
        help="Minimum fill-rate constraint used during return optimization.",
    )
    parser.add_argument(
        "--min-monthly-fills",
        type=float,
        default=0.0,
        help="Minimum average monthly fills constraint used during return optimization.",
    )
    parser.add_argument(
        "--max-drawdown-limit",
        type=float,
        default=1.0,
        help="Maximum allowed drawdown in strategy optimization (1.0 means no limit).",
    )
    parser.add_argument(
        "--min-buy-signals",
        type=int,
        default=1,
        help="Minimum BUY signals per group. If not met, threshold is relaxed downward.",
    )
    parser.add_argument(
        "--fallback-min-threshold",
        type=float,
        default=0.50,
        help="Lower bound for threshold relaxation when enforcing minimum BUY signals.",
    )
    parser.add_argument(
        "--no-sector-extra",
        action="store_true",
        help="Disable dynamic symbol expansion from bullish sectors.",
    )
    parser.add_argument(
        "--sector-extra-max-symbols",
        type=int,
        default=10,
        help="Maximum number of extra symbols added from bullish sectors per group.",
    )
    parser.add_argument(
        "--sector-extra-universe",
        type=Path,
        default=None,
        help="Optional CSV with 'symbol' column used as expansion universe.",
    )
    parser.add_argument(
        "--sector-signal-short-window",
        type=int,
        default=5,
        help="Short MA/return window for sector bullish signal.",
    )
    parser.add_argument(
        "--sector-signal-long-window",
        type=int,
        default=20,
        help="Long MA window for sector bullish signal.",
    )
    parser.add_argument(
        "--sector-signal-source",
        type=str,
        choices=["local", "etf"],
        default="local",
        help="Source for sector bullish signal: local=from downloaded symbol universe, etf=US sector ETFs.",
    )
    parser.add_argument(
        "--sector-ret-source",
        type=str,
        choices=["local", "etf"],
        default="local",
        help="Source for sector return feature in model: local or etf.",
    )
    parser.add_argument(
        "--sector-master",
        type=Path,
        default=None,
        help="Optional CSV with columns symbol,sector used as local sector master.",
    )
    parser.add_argument(
        "--sector-extra-min-prob",
        type=float,
        default=0.62,
        help="Minimum TP probability for dynamically added sector symbols.",
    )
    parser.add_argument(
        "--sector-extra-min-value",
        type=float,
        default=100000000.0,
        help="Minimum latest traded value (Close*Volume, JPY) for dynamic sector symbols.",
    )
    parser.add_argument(
        "--sector-extra-top-percent",
        type=float,
        default=40.0,
        help="Keep top X%% of filtered dynamic sector symbols by TP probability.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run(
        watchlist_path=args.watchlist,
        threshold=args.threshold,
        period=args.period,
        output_dir=args.output_dir,
        horizon_days=args.horizon_days,
        take_profit=args.take_profit,
        max_corr=args.max_corr,
        corr_lookback_days=args.corr_lookback_days,
        max_positions=args.max_positions,
        max_per_sector=args.max_per_sector,
        include_sector=not args.no_sector,
        expected_jst_hour=args.expected_jst_hour,
        discord_webhook_url=args.discord_webhook_url,
        min_fill_rate=args.min_fill_rate,
        min_monthly_fills=args.min_monthly_fills,
        max_drawdown_limit=args.max_drawdown_limit,
        min_buy_signals=args.min_buy_signals,
        fallback_min_threshold=args.fallback_min_threshold,
        sector_extra_enabled=not args.no_sector_extra,
        sector_extra_max_symbols=args.sector_extra_max_symbols,
        sector_extra_universe_path=args.sector_extra_universe,
        sector_signal_short_window=args.sector_signal_short_window,
        sector_signal_long_window=args.sector_signal_long_window,
        sector_signal_source=args.sector_signal_source,
        sector_ret_source=args.sector_ret_source,
        sector_master_path=args.sector_master,
        sector_extra_min_prob=args.sector_extra_min_prob,
        sector_extra_min_value=args.sector_extra_min_value,
        sector_extra_top_percent=args.sector_extra_top_percent,
    )


if __name__ == "__main__":
    raise SystemExit(main())

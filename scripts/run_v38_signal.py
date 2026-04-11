from __future__ import annotations

import argparse
import io
import sys
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.shikiho_text_parser import backtest_phase_adaptive_practical_v35_alloc50_candidate as base_mod
from projects.shikiho_text_parser.compare_alloc50_post_major_multi_etf_expanded_candidate import (
    BASE_ETF_OUTPERFORM_MIN,
    BASE_ETF_TICKER,
    ETF_SL,
    ETF_SCORE_GAP_MIN,
    ETF_SCORE_MIN,
    ETF_TICKERS,
    ETF_TP,
    POST_MAJOR_PHASES,
    STOCK_BREAK_MULT,
    STOCK_SL,
    STOCK_TP,
    STOCK_TOP_N,
    build_post_major_state,
    classify_sector_state,
    pick_best_etf,
    prepare_earnings_map,
    prev_high_break_signal,
    q2026_2q_spec,
    q2_spec,
    q2025_2q_spec,
    q3_spec,
    q4_2_spec,
    q4_spec,
)
from projects.shikiho_text_parser.search_phase_method_optimization import PRICE_DIR
from src.stock_signal import download_daily_data, save_price_data


DATASET_SPECS = {
    "q2_2024": q2_spec,
    "2025_2q": q2025_2q_spec,
    "2026-2Q": q2026_2q_spec,
    "3Q": q3_spec,
    "4Q": q4_spec,
    "4Q-2": q4_2_spec,
}

DEFAULT_DATASET = "2026-2Q"
SSA_CSV = ROOT / "projects" / "shikiho_text_parser" / "output" / "nikkei_ssa_guard_candidate" / "nikkei_ssa_daily.csv"
SP500_RETURNS_CSV = (
    ROOT
    / "projects"
    / "shikiho_text_parser"
    / "output"
    / "backtest_post_crash_surge_daytrade_sp500_candidate"
    / "sp500_returns_shifted_for_jp.csv"
)
NO_TRADE_SUB_PHASES = {"surge", "downtrend", "reversal_down", "raw_post_crash_high_vol"}
V38_NO_ENTRY_PHASES = {"crash", "surge"}
CRASH_LATE_AFTER_N = 4
SUB_COOLDOWN_DAYS = 5
SUB_DAILY_TOP_N = 2
THEME_MIN_COUNT = 2
RET5_MIN = 0.12
VOL_RATIO_MIN = 2.0
CLOSE_POS_MIN = 0.60
UPPER_SHADOW_MAX = 0.25
SUB_TP = 0.15
SUB_SL = 0.08
V38_ETF_TP = 0.08
V38_ETF_SL = 0.04
V38_STOCK_TP_STRONG = 0.10
V38_STOCK_SL_STRONG = 0.05
V38_STOCK_TP_NORMAL = 0.06
V38_STOCK_SL_NORMAL = 0.03
V38_NORMAL_PRICE_TOP3_MAX = 0.42
V38_ETF_EXCLUDED_PHASES = {"capitulation_end", "settling", "surge"}
V38_ETF_SIGNAL_TICKERS = [ticker for ticker in ETF_TICKERS if ticker != "1328.T"]
V38_SURGE_DAYTRADE_SP500_MIN = 0.01
V38_SURGE_DAYTRADE_TOP_N = 2
V38_SURGE_DAYTRADE_RET1_MIN = 0.05
V38_SURGE_DAYTRADE_RET5_MIN = 0.12
V38_SURGE_DAYTRADE_VOL_RATIO_MIN = 2.0
V38_SURGE_DAYTRADE_PHASES = {"normal", "downtrend"}
V38_REBOUND_DAYTRADE_SP500_MIN = 0.01
V38_REBOUND_DAYTRADE_TOP_N = 1
V38_REBOUND_DAYTRADE_RET1_MIN = 0.05
V38_REBOUND_DAYTRADE_VOL_RATIO_MIN = 2.0
V38_REBOUND_DAYTRADE_CLOSE_POS_MIN = 0.55
V38_REBOUND_DAYTRADE_PHASES = {"normal", "downtrend"}
V38_REBOUND_DAYTRADE_ALLOC_YEN = 500_000.0


def compute_fundamental_proxy(df: pd.DataFrame) -> pd.Series:
    parts: list[pd.Series] = []
    for col in ["overall_score", "finance_quality_score", "growth_quality_score", "fundamental_score"]:
        if col in df.columns:
            parts.append(pd.to_numeric(df[col], errors="coerce"))
    for col in ["score_growth", "score_profitability", "score_safety", "shikiho_score_overall"]:
        if col in df.columns:
            parts.append(pd.to_numeric(df[col], errors="coerce") / 5.0)
    if not parts:
        return pd.Series(np.nan, index=df.index)
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True)


def compute_expensive_proxy(df: pd.DataFrame) -> pd.Series:
    parts: list[pd.Series] = []
    if "sector_adjusted_per_score" in df.columns:
        parts.append(pd.to_numeric(df["sector_adjusted_per_score"], errors="coerce"))
    if "forecast_per" in df.columns:
        parts.append((pd.to_numeric(df["forecast_per"], errors="coerce") >= 25).astype(float))
    if "ocr_per" in df.columns:
        parts.append((pd.to_numeric(df["ocr_per"], errors="coerce") >= 25).astype(float))
    if not parts:
        return pd.Series(np.nan, index=df.index)
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True)


def first_nonempty(row: dict | pd.Series, keys: list[str], default: str = "UNKNOWN") -> str:
    for key in keys:
        value = row.get(key) if hasattr(row, "get") else None
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return default


def effective_regime_name(phase_name: str, post_major: bool) -> str:
    if post_major and phase_name != "crash":
        return f"post_crash_{phase_name}"
    return phase_name


def next_business_day(ts: pd.Timestamp) -> pd.Timestamp:
    d = ts + pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d += pd.Timedelta(days=1)
    return d.normalize()


def refresh_prices(
    tickers: list[str],
    price_dir: Path,
    priority_tickers: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    ordered = list(dict.fromkeys((priority_tickers or []) + tickers))
    success_tickers: list[str] = []
    failed_tickers: list[str] = []
    for ticker in ordered:
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                df = download_daily_data(ticker, period="max")
            save_price_data(ticker, df, price_dir)
            success_tickers.append(ticker)
        except Exception as exc:
            failed_tickers.append(ticker)
            print(f"[REFRESH_FAILED_TICKER] {ticker}: {type(exc).__name__}: {exc}")
            continue
    print(
        f"[REFRESH] requested={len(ordered)} success={len(success_tickers)} failed={len(failed_tickers)}"
    )
    if failed_tickers:
        print(f"[REFRESH_FAILED] {', '.join(failed_tickers[:20])}")
    return success_tickers, failed_tickers


def summarize_price_freshness(
    price_map: dict[str, pd.DataFrame],
    target_tickers: list[str] | None = None,
    label: str = "PRICE",
) -> tuple[pd.Timestamp, list[str]]:
    latest_by_ticker: dict[str, pd.Timestamp] = {}
    ticker_filter = set(target_tickers) if target_tickers is not None else None
    for ticker, df in price_map.items():
        if ticker_filter is not None and ticker not in ticker_filter:
            continue
        if df.empty:
            continue
        latest_by_ticker[ticker] = df.index.max()
    if not latest_by_ticker:
        raise ValueError("No price data found for freshness check.")
    latest_signal = max(latest_by_ticker.values())
    stale = sorted(
        ticker for ticker, latest in latest_by_ticker.items()
        if latest < latest_signal
    )
    print(
        f"[{label}_STATUS] latest_signal_date={latest_signal.date().isoformat()} "
        f"up_to_date={len(latest_by_ticker) - len(stale)} stale={len(stale)} total={len(latest_by_ticker)}"
    )
    if stale:
        print(f"[{label}_STALE_SAMPLE] {', '.join(stale[:20])}")
    return latest_signal, stale


def stale_by_business_day_tolerance(
    price_map: dict[str, pd.DataFrame],
    latest_signal: pd.Timestamp,
    target_tickers: list[str],
    max_lag_bdays: int,
) -> list[str]:
    stale: list[str] = []
    for ticker in target_tickers:
        df = price_map.get(ticker)
        if df is None or df.empty:
            stale.append(ticker)
            continue
        last_date = pd.Timestamp(df.index.max()).normalize()
        bdays_lag = len(pd.bdate_range(last_date + pd.offsets.BDay(1), latest_signal))
        if bdays_lag > max_lag_bdays:
            stale.append(ticker)
    return sorted(set(stale))


def latest_signal_date(price_map: dict[str, pd.DataFrame]) -> pd.Timestamp:
    latest = [df.index.max() for df in price_map.values() if not df.empty]
    if not latest:
        raise ValueError("No price data found for v3.6 signal generation.")
    return max(latest)


def load_ssa_prior_state(trade_date: pd.Timestamp) -> tuple[bool, bool]:
    if not SSA_CSV.exists():
        return False, False
    ssa_df = pd.read_csv(SSA_CSV, usecols=["Date", "ssa_recovery_confirm"])
    ssa_df["Date"] = pd.to_datetime(ssa_df["Date"], errors="coerce")
    ssa_df["ssa_recovery_confirm"] = ssa_df["ssa_recovery_confirm"].astype(str).str.lower().eq("true")
    prior = ssa_df.loc[ssa_df["Date"] < trade_date].tail(1)
    if prior.empty:
        return False, False
    return bool(prior.iloc[0]["ssa_recovery_confirm"]), True


def load_ssa_prior_confirm(trade_date: pd.Timestamp) -> bool:
    confirm, _available = load_ssa_prior_state(trade_date)
    return confirm


def v38_etf_allowed(
    post_major_active: bool,
    sector_mode: str,
    phase_name: str,
    ssa_confirm_prior: bool,
    ssa_available_prior: bool,
) -> bool:
    if not post_major_active:
        return False
    if sector_mode != "concentrated":
        return False
    if phase_name in V38_NO_ENTRY_PHASES:
        return False
    if phase_name in V38_ETF_EXCLUDED_PHASES:
        return False
    if ssa_available_prior and not ssa_confirm_prior:
        return False
    return True


def v38_stock_allowed(
    post_major_active: bool,
    sector_state: dict,
    phase_name: str,
) -> tuple[bool, float, float]:
    if not post_major_active:
        return False, V38_STOCK_TP_STRONG, V38_STOCK_SL_STRONG
    if str(sector_state.get("mode")) != "dispersed":
        return False, V38_STOCK_TP_STRONG, V38_STOCK_SL_STRONG
    if phase_name in V38_NO_ENTRY_PHASES:
        return False, V38_STOCK_TP_STRONG, V38_STOCK_SL_STRONG
    if phase_name in {"capitulation_end", "downtrend"}:
        return True, V38_STOCK_TP_STRONG, V38_STOCK_SL_STRONG
    if phase_name == "normal" and float(sector_state.get("price_top3_share", 1.0)) <= V38_NORMAL_PRICE_TOP3_MAX:
        return True, V38_STOCK_TP_NORMAL, V38_STOCK_SL_NORMAL
    return False, V38_STOCK_TP_STRONG, V38_STOCK_SL_STRONG


def load_sp500_return_prior(trade_date: pd.Timestamp) -> tuple[float | None, bool]:
    if not SP500_RETURNS_CSV.exists():
        return None, False
    try:
        df = pd.read_csv(SP500_RETURNS_CSV)
    except Exception:
        return None, False
    if df.empty:
        return None, False
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    val_col = "ret_sp500" if "ret_sp500" in df.columns else df.columns[-1]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    row = df.loc[df[date_col].eq(pd.Timestamp(trade_date).normalize())].tail(1)
    if row.empty:
        prior = df.loc[df[date_col] < pd.Timestamp(trade_date).normalize()].tail(1)
        if prior.empty:
            return None, False
        lag_days = (pd.Timestamp(trade_date).normalize() - pd.Timestamp(prior.iloc[0][date_col]).normalize()).days
        if lag_days > 4:
            return None, False
        row = prior
    if row.empty:
        return None, False
    val = pd.to_numeric(row.iloc[0][val_col], errors="coerce")
    if pd.isna(val):
        return None, False
    return float(val), True


def compute_moneyflow_features(df: pd.DataFrame, signal_date: pd.Timestamp) -> dict | None:
    hist = df.loc[df.index <= signal_date].copy().sort_index()
    if len(hist) < 25 or signal_date not in hist.index:
        return None
    hist["ret1"] = hist["Close"] / hist["Close"].shift(1) - 1.0
    hist["ret5"] = hist["Close"] / hist["Close"].shift(5) - 1.0
    hist["vol_ratio20"] = hist["Volume"] / hist["Volume"].rolling(20).mean()
    hist["prev20_high"] = hist["High"].shift(1).rolling(20).max()
    price_range = (hist["High"] - hist["Low"]).replace(0, np.nan)
    hist["close_pos"] = (hist["Close"] - hist["Low"]) / price_range
    hist["upper_shadow_ratio"] = (hist["High"] - hist[["Close", "Open"]].max(axis=1)) / price_range
    hist["value_traded"] = hist["Close"] * hist["Volume"]
    hist["prev20_break"] = hist["Close"] > hist["prev20_high"]
    row = hist.loc[signal_date]
    needed = [
        "ret1",
        "ret5",
        "vol_ratio20",
        "close_pos",
        "upper_shadow_ratio",
        "value_traded",
        "prev20_break",
        "High",
        "Close",
    ]
    if any(pd.isna(row.get(col)) for col in needed):
        return None
    if float(row["ret1"]) < V38_SURGE_DAYTRADE_RET1_MIN:
        return None
    if float(row["ret5"]) < V38_SURGE_DAYTRADE_RET5_MIN:
        return None
    if float(row["vol_ratio20"]) < V38_SURGE_DAYTRADE_VOL_RATIO_MIN:
        return None
    if float(row["close_pos"]) < 0.60:
        return None
    if float(row["upper_shadow_ratio"]) > 0.25:
        return None
    if not bool(row["prev20_break"]):
        return None
    return {
        "ret1": float(row["ret1"]),
        "ret5": float(row["ret5"]),
        "vol_ratio20": float(row["vol_ratio20"]),
        "close_pos": float(row["close_pos"]),
        "upper_shadow_ratio": float(row["upper_shadow_ratio"]),
        "value_traded": float(row["value_traded"]),
        "trigger": float(row["High"]) * STOCK_BREAK_MULT,
        "close": float(row["Close"]),
    }


def compute_rebound_daytrade_features(df: pd.DataFrame, signal_date: pd.Timestamp) -> dict | None:
    hist = df.loc[df.index <= signal_date].copy().sort_index()
    if len(hist) < 25 or signal_date not in hist.index:
        return None
    hist["ret1"] = hist["Close"] / hist["Close"].shift(1) - 1.0
    hist["ret5"] = hist["Close"] / hist["Close"].shift(5) - 1.0
    hist["vol_ratio20"] = hist["Volume"] / hist["Volume"].rolling(20).mean()
    hist["prev20_high"] = hist["High"].shift(1).rolling(20).max()
    price_range = (hist["High"] - hist["Low"]).replace(0, np.nan)
    hist["close_pos"] = (hist["Close"] - hist["Low"]) / price_range
    hist["value_traded"] = hist["Close"] * hist["Volume"]
    hist["prev20_break"] = hist["Close"] > hist["prev20_high"]
    row = hist.loc[signal_date]
    needed = [
        "ret1",
        "vol_ratio20",
        "close_pos",
        "value_traded",
        "prev20_break",
        "High",
        "Close",
    ]
    if any(pd.isna(row.get(col)) for col in needed):
        return None
    if float(row["ret1"]) < V38_REBOUND_DAYTRADE_RET1_MIN:
        return None
    if float(row["vol_ratio20"]) < V38_REBOUND_DAYTRADE_VOL_RATIO_MIN:
        return None
    if float(row["close_pos"]) < V38_REBOUND_DAYTRADE_CLOSE_POS_MIN:
        return None
    if not bool(row["prev20_break"]):
        return None
    return {
        "ret1": float(row["ret1"]),
        "ret5": float(row["ret5"]) if pd.notna(row.get("ret5")) else np.nan,
        "vol_ratio20": float(row["vol_ratio20"]),
        "close_pos": float(row["close_pos"]),
        "value_traded": float(row["value_traded"]),
        "trigger": float(row["High"]),
        "close": float(row["Close"]),
    }


def build_rebound_daytrade_candidates(
    price_map_all: dict[str, pd.DataFrame],
    active_lookup: dict,
    signal_date: pd.Timestamp,
    prior_phase_name: str,
    post_major: bool,
    projected_phase_name: str,
    sp500_ret_prior: float | None,
    require_sp500: bool = True,
) -> list[tuple[str, float, dict]]:
    if not post_major:
        return []
    if projected_phase_name in V38_NO_ENTRY_PHASES:
        return []
    if prior_phase_name not in V38_REBOUND_DAYTRADE_PHASES:
        return []
    if require_sp500 and (sp500_ret_prior is None or sp500_ret_prior < V38_REBOUND_DAYTRADE_SP500_MIN):
        return []
    feature_rows: list[tuple[str, dict]] = []
    for ticker, df in price_map_all.items():
        if ticker in ETF_TICKERS or ticker not in active_lookup:
            continue
        features = compute_rebound_daytrade_features(df, signal_date)
        if not features:
            continue
        feature_rows.append((ticker, features))
    if not feature_rows:
        return []
    feat_df = pd.DataFrame([{**features, "ticker": ticker} for ticker, features in feature_rows])
    for col in ["ret1", "vol_ratio20", "value_traded", "close_pos"]:
        feat_df[f"{col}_rank"] = feat_df[col].rank(pct=True, method="average")
    feat_df["moneyflow_score"] = (
        feat_df["ret1_rank"] * 40.0
        + feat_df["vol_ratio20_rank"] * 25.0
        + feat_df["value_traded_rank"] * 25.0
        + feat_df["close_pos_rank"] * 10.0
    )
    out: list[tuple[str, float, dict]] = []
    for _, row in feat_df.sort_values("moneyflow_score", ascending=False).head(V38_REBOUND_DAYTRADE_TOP_N).iterrows():
        detail = row.to_dict()
        out.append((str(row["ticker"]), float(row["moneyflow_score"]), detail))
    return out


def build_surge_daytrade_candidates(
    price_map_all: dict[str, pd.DataFrame],
    active_lookup: dict,
    trade_date: pd.Timestamp,
    signal_date: pd.Timestamp,
    prior_phase_name: str,
    post_major: bool,
    sp500_ret_prior: float | None,
) -> list[tuple[str, float, dict]]:
    # v3.8 policy: crash/surge prediction is not reliable enough for live entry.
    # Keep the helper for historical research imports, but never emit live signals.
    return []
    if not post_major:
        return []
    if prior_phase_name not in V38_SURGE_DAYTRADE_PHASES:
        return []
    if sp500_ret_prior is None or sp500_ret_prior < V38_SURGE_DAYTRADE_SP500_MIN:
        return []
    feature_rows: list[tuple[str, dict]] = []
    for ticker, df in price_map_all.items():
        if ticker in ETF_TICKERS or ticker not in active_lookup:
            continue
        features = compute_moneyflow_features(df, signal_date)
        if not features:
            continue
        feature_rows.append((ticker, features))
    if not feature_rows:
        return []
    feat_df = pd.DataFrame([{**f, "ticker": t} for t, f in feature_rows])
    for col in ["ret1", "ret5", "vol_ratio20", "value_traded", "close_pos"]:
        feat_df[f"{col}_rank"] = feat_df[col].rank(pct=True, method="average")
    feat_df["moneyflow_score"] = (
        feat_df["ret1_rank"] * 15.0
        + feat_df["ret5_rank"] * 10.0
        + feat_df["vol_ratio20_rank"] * 35.0
        + feat_df["value_traded_rank"] * 30.0
        + feat_df["close_pos_rank"] * 10.0
    )
    out: list[tuple[str, float, dict]] = []
    for _, row in feat_df.sort_values("moneyflow_score", ascending=False).head(V38_SURGE_DAYTRADE_TOP_N).iterrows():
        detail = row.to_dict()
        out.append((str(row["ticker"]), float(row["moneyflow_score"]), detail))
    return out


def v38_live_etf_strength_score(
    df: pd.DataFrame,
    trade_date: pd.Timestamp,
) -> tuple[bool, float | None, float]:
    """Score ETF using only data available before the next trade date.

    The backtest helper checks the trade day's high to decide whether a stop
    trigger was hit. For live notifications that day does not exist yet, so we
    only emit tomorrow's trigger price and rank ETFs from the signal-day state.
    """
    hist = df.loc[df.index < trade_date].copy().sort_index()
    if len(hist) < 10:
        return False, None, -999.0
    basis = hist.iloc[-1]
    close_now = float(basis["Close"])
    close_5d = float(hist.iloc[-6]["Close"])
    low_10d = float(hist.tail(10)["Low"].min()) if "Low" in hist.columns else float(hist.tail(10)["Close"].min())
    high_now = float(basis["High"])
    day_range = float(basis["High"] - basis["Low"]) if "Low" in basis.index else 0.0
    close_pos = (close_now - float(basis["Low"])) / day_range if day_range > 0 else 0.5
    ret5 = close_now / close_5d - 1.0 if close_5d > 0 else 0.0
    reclaim10 = close_now / low_10d - 1.0 if low_10d > 0 else 0.0
    trigger = high_now * STOCK_BREAK_MULT
    score = ret5 + 0.5 * reclaim10 + 0.05 * close_pos
    ok = ret5 > 0 and reclaim10 > 0 and score >= ETF_SCORE_MIN
    return bool(ok), float(trigger), float(score)


def v38_pick_best_etf_live(
    price_map: dict[str, pd.DataFrame],
    trade_date: pd.Timestamp,
) -> tuple[str | None, float | None, dict[str, float]]:
    scored: list[tuple[str, float, float]] = []
    detail_scores: dict[str, float] = {}
    triggers: dict[str, float] = {}
    for ticker in V38_ETF_SIGNAL_TICKERS:
        df = price_map.get(ticker)
        if df is None or df.empty:
            detail_scores[ticker] = -999.0
            continue
        ok, trigger, score = v38_live_etf_strength_score(df, trade_date)
        detail_scores[ticker] = score
        if ok and trigger is not None:
            triggers[ticker] = trigger
            scored.append((ticker, trigger, score))
    if not scored:
        return None, None, detail_scores
    scored.sort(key=lambda x: x[2], reverse=True)
    top_ticker, top_trigger, top_score = scored[0]
    second_score = scored[1][2] if len(scored) > 1 else -999.0
    base_score = detail_scores.get(BASE_ETF_TICKER, -999.0)
    if top_score < ETF_SCORE_MIN:
        return None, None, detail_scores
    if len(scored) > 1 and (top_score - second_score) < ETF_SCORE_GAP_MIN:
        if BASE_ETF_TICKER in triggers:
            return BASE_ETF_TICKER, triggers[BASE_ETF_TICKER], detail_scores
        return top_ticker, top_trigger, detail_scores
    if top_ticker != BASE_ETF_TICKER:
        if base_score <= -900.0 or top_score < (base_score + BASE_ETF_OUTPERFORM_MIN):
            if BASE_ETF_TICKER in triggers:
                return BASE_ETF_TICKER, triggers[BASE_ETF_TICKER], detail_scores
    return top_ticker, top_trigger, detail_scores


def prev_high_break_trigger_live(
    df: pd.DataFrame,
    basis_date: pd.Timestamp,
    break_mult: float = STOCK_BREAK_MULT,
) -> float | None:
    hist = df.loc[:basis_date]
    if len(hist) < 3 or "High" not in hist.columns:
        return None
    prev_high = hist.iloc[-1]["High"]
    if pd.isna(prev_high):
        return None
    return float(prev_high) * break_mult


def format_buy(symbol: str, method: str, sector: str, company: str, close_price: float, entry_price: float, tp_prob: float, tp_ratio: float, sl_ratio: float) -> str:
    tp_price = entry_price * (1.0 + tp_ratio)
    sl_price = entry_price * (1.0 - sl_ratio)
    entry_ratio_pct = (entry_price / close_price - 1.0) * 100.0 if close_price > 0 else 0.0
    return (
        f"[BUY] {symbol} | method={method} close={close_price:.2f} "
        f"tp_prob={tp_prob:.2f}% lmt={entry_ratio_pct:.1f}% lmt_price={entry_price:.2f} "
        f"tp={tp_ratio * 100.0:.1f}% tp_price={tp_price:.2f} "
        f"sl={sl_ratio * 100.0:.1f}% sl_price={sl_price:.2f}"
    ), (
        f"[PICK][v38] {symbol} tp_prob={tp_prob:.2f}% "
        f"sector={sector} method={method} company={company}"
    )


def compute_no_trade_subsignal(
    df: pd.DataFrame,
    signal_date: pd.Timestamp,
) -> dict | None:
    hist = df.loc[df.index <= signal_date].copy().sort_index()
    if len(hist) < 25 or signal_date not in hist.index:
        return None
    hist["ret5"] = hist["Close"] / hist["Close"].shift(5) - 1.0
    hist["vol_ratio20"] = hist["Volume"] / hist["Volume"].rolling(20).mean()
    hist["prev20_high"] = hist["High"].shift(1).rolling(20).max()
    price_range = (hist["High"] - hist["Low"]).replace(0, np.nan)
    hist["close_pos"] = (hist["Close"] - hist["Low"]) / price_range
    hist["upper_shadow_ratio"] = (hist["High"] - hist[["Close", "Open"]].max(axis=1)) / price_range
    row = hist.loc[signal_date]
    if any(pd.isna(row.get(col)) for col in ["ret5", "vol_ratio20", "prev20_high", "close_pos", "upper_shadow_ratio"]):
        return None
    close_price = float(row["Close"])
    if close_price <= float(row["prev20_high"]):
        return None
    if float(row["ret5"]) < RET5_MIN:
        return None
    if float(row["vol_ratio20"]) < VOL_RATIO_MIN:
        return None
    if float(row["close_pos"]) < CLOSE_POS_MIN:
        return None
    if float(row["upper_shadow_ratio"]) > UPPER_SHADOW_MAX:
        return None
    rank_score = (
        float(row["ret5"]) * 100.0
        + min(float(row["vol_ratio20"]), 8.0) * 5.0
        + float(row["close_pos"]) * 10.0
        - float(row["upper_shadow_ratio"]) * 10.0
    )
    return {
        "close_price": close_price,
        "ret5": float(row["ret5"]),
        "vol_ratio20": float(row["vol_ratio20"]),
        "close_pos": float(row["close_pos"]),
        "upper_shadow_ratio": float(row["upper_shadow_ratio"]),
        "rank_score": rank_score,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate v3.8 live signal log for notification.")
    parser.add_argument("--dataset", choices=list(DATASET_SPECS.keys()), default=DEFAULT_DATASET)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--skip-refresh", action="store_true")
    parser.add_argument("--refresh-only", action="store_true")
    parser.add_argument("--max-stale-count", type=int, default=25)
    parser.add_argument("--max-selected-stale-count", type=int, default=0)
    parser.add_argument("--max-selected-lag-bdays", type=int, default=0)
    parser.add_argument("--daytrade-only", action="store_true", help="Emit only the v3.8 rebound daytrade trigger candidates.")
    parser.add_argument("--daytrade-precheck", action="store_true", help="Emit prior-night v3.8 rebound daytrade provisional candidates without the S&P500 gate.")
    parser.add_argument("--disable-daytrade", action="store_true", help="Disable v3.8 rebound daytrade trigger candidates.")
    args = parser.parse_args()

    spec = DATASET_SPECS[args.dataset]()
    detail_df = base_mod.normalize_static(pd.read_csv(spec.detail_csv)).drop_duplicates(subset=["ticker"], keep="first")
    selected_df = base_mod.normalize_static(pd.read_csv(spec.selected_csv)).drop_duplicates(subset=["ticker"], keep="first")
    detail_df["fundamental_proxy"] = compute_fundamental_proxy(detail_df)
    detail_df["expensive_proxy"] = compute_expensive_proxy(detail_df)
    detail_df["hard_detached_proxy"] = (
        detail_df["fundamental_proxy"].notna()
        & detail_df["expensive_proxy"].notna()
        & (detail_df["fundamental_proxy"] <= 0.40)
        & (detail_df["expensive_proxy"] >= 0.50)
    )
    tickers = list(dict.fromkeys(detail_df["ticker"].astype(str).tolist() + ETF_TICKERS))
    selected_tickers = list(dict.fromkeys(selected_df["ticker"].astype(str).tolist() + V38_ETF_SIGNAL_TICKERS))

    if not args.skip_refresh:
        priority_tickers = selected_tickers + ["^N225"]
        refresh_prices(tickers, PRICE_DIR, priority_tickers=priority_tickers)
    price_map_all = base_mod.load_price_map(tickers, pd.Timestamp.today())
    signal_date, stale_tickers = summarize_price_freshness(price_map_all, label="PRICE")
    _, selected_stale_raw = summarize_price_freshness(price_map_all, target_tickers=selected_tickers, label="SELECTED")
    selected_stale_tickers = stale_by_business_day_tolerance(
        price_map_all,
        signal_date,
        selected_tickers,
        args.max_selected_lag_bdays,
    )
    if selected_stale_raw and not selected_stale_tickers:
        print(
            f"[SELECTED_STALE_OK] tolerated within {args.max_selected_lag_bdays} business day(s): "
            f"{', '.join(selected_stale_raw[:20])}"
        )
    if not args.skip_refresh and selected_stale_tickers:
        print(f"[REFRESH_RETRY] retrying stale selected tickers: {', '.join(selected_stale_tickers[:20])}")
        refresh_prices(selected_stale_tickers, PRICE_DIR, priority_tickers=selected_stale_tickers)
        price_map_all = base_mod.load_price_map(tickers, pd.Timestamp.today())
        signal_date, stale_tickers = summarize_price_freshness(price_map_all, label="PRICE")
        _, selected_stale_raw = summarize_price_freshness(price_map_all, target_tickers=selected_tickers, label="SELECTED")
        selected_stale_tickers = stale_by_business_day_tolerance(
            price_map_all,
            signal_date,
            selected_tickers,
            args.max_selected_lag_bdays,
        )
        if selected_stale_raw and not selected_stale_tickers:
            print(
                f"[SELECTED_STALE_OK] tolerated within {args.max_selected_lag_bdays} business day(s): "
                f"{', '.join(selected_stale_raw[:20])}"
            )
    if len(stale_tickers) > args.max_stale_count:
        print(
            f"[PRICE_WARNING] stale tickers after refresh: {len(stale_tickers)} > {args.max_stale_count}"
        )
    if len(selected_stale_tickers) > args.max_selected_stale_count:
        raise SystemExit(
            f"Too many selected stale tickers after refresh: "
            f"{len(selected_stale_tickers)} > {args.max_selected_stale_count}"
        )
    if args.refresh_only:
        print(f"[REFRESH_ONLY] dataset={args.dataset} signal_date={signal_date.date().isoformat()}")
        return 0
    trade_date = next_business_day(signal_date)

    phase_df = build_post_major_state(pd.read_csv(base_mod.PHASE_CSV, skiprows=[1]).rename(columns={"phase": "phase_name"}))
    phase_df["Date"] = pd.to_datetime(phase_df["Date"])
    for col in ["ret5", "dd20", "vol10"]:
        phase_df[col] = pd.to_numeric(phase_df[col], errors="coerce")
    phase_state = phase_df.set_index("Date").sort_index()
    phase_map = base_mod.load_phase_map(base_mod.PHASE_CSV)
    earnings_map = prepare_earnings_map(spec)

    stock_price_map_all = {k: v for k, v in price_map_all.items() if k not in ETF_TICKERS}
    static_lookup = detail_df.drop_duplicates(subset=["ticker"], keep="first").set_index("ticker").to_dict("index")
    start_date = pd.Timestamp(spec.start_date)
    trade_dates = sorted(
        set().union(
            *[
                set(df[(df.index >= start_date) & (df.index <= signal_date)].index.tolist())
                for df in stock_price_map_all.values()
            ]
        )
    )
    if not trade_dates:
        raise SystemExit("No stock trade dates found for selected dataset.")

    standard_tables = base_mod.build_interval_standard_tables(spec.name, detail_df, stock_price_map_all, trade_dates, 10)
    crash_tables = base_mod.build_interval_crash_tables(detail_df, stock_price_map_all, trade_dates, 10)
    _, anchor_for_date = base_mod.build_rebalance_schedule(trade_dates, 10)
    anchor_key = anchor_for_date[signal_date]

    phase_name = base_mod.projected_phase_name(trade_date, phase_map, 1, "difficult_v11")
    rule_name = "crash_disabled" if phase_name == "crash" else base_mod.practical_v2_mapping().get(phase_name, "condition2")
    standard_table = standard_tables.get(anchor_key, pd.DataFrame())
    standard_lookup = standard_table.set_index("ticker").to_dict("index") if not standard_table.empty else {}
    active_lookup = crash_tables[anchor_key].set_index("ticker").to_dict("index") if phase_name == "crash" and anchor_key in crash_tables and not crash_tables[anchor_key].empty else standard_lookup

    prior_phase_row = phase_state.loc[phase_state.index < trade_date].tail(1)
    post_major = False if prior_phase_row.empty else bool(prior_phase_row.iloc[0]["post_major_crash_mode"])
    post_major_phase = "" if prior_phase_row.empty else str(prior_phase_row.iloc[0]["phase_name"])
    post_major_active = post_major and post_major_phase in POST_MAJOR_PHASES
    sector_state = classify_sector_state(standard_table, price_map_all, trade_date)
    ssa_confirm_prior, ssa_available_prior = load_ssa_prior_state(trade_date)
    if args.daytrade_precheck:
        sp500_ret_prior, sp500_available_prior = None, False
    else:
        sp500_ret_prior, sp500_available_prior = load_sp500_return_prior(trade_date)
    etf_post_major_active = v38_etf_allowed(
        post_major_active=post_major_active,
        sector_mode=str(sector_state["mode"]),
        phase_name=phase_name,
        ssa_confirm_prior=ssa_confirm_prior,
        ssa_available_prior=ssa_available_prior,
    )
    stock_post_major_active, v38_stock_tp, v38_stock_sl = v38_stock_allowed(
        post_major_active=post_major_active,
        sector_state=sector_state,
        phase_name=phase_name,
    )
    effective_regime = effective_regime_name(phase_name, post_major)

    display_method = (
        "v38_post_crash_rebound_daytrade_precheck"
        if args.daytrade_precheck
        else ("v38_post_crash_rebound_daytrade" if args.daytrade_only else ("crash_disabled" if phase_name == "crash" else rule_name))
    )
    print(
        f"[META] regime={phase_name} method={display_method} "
        f"signal_date={signal_date.date().isoformat()} trade_date={trade_date.date().isoformat()} "
        f"ssa_confirm_prior={ssa_confirm_prior} "
        f"ssa_available_prior={ssa_available_prior} "
        f"crash_late_active=False "
        f"crash_pos= "
        f"post_major_crash_mode={post_major} "
        f"post_major_phase={post_major_phase or 'none'} "
        f"sector_mode={sector_state['mode']} "
        f"effective_regime={effective_regime} "
        f"sp500_ret_prior={sp500_ret_prior if sp500_ret_prior is not None else 'NA'} "
        f"sp500_available_prior={sp500_available_prior}"
    )

    lines: list[str] = []
    pick_lines: list[str] = []

    surge_daytrade_candidates: list[tuple[str, float, dict]] = []
    rebound_daytrade_candidates = []
    if not args.disable_daytrade:
        rebound_daytrade_candidates = build_rebound_daytrade_candidates(
            price_map_all=price_map_all,
            active_lookup=static_lookup,
            signal_date=signal_date,
            prior_phase_name=post_major_phase,
            post_major=post_major,
            projected_phase_name=phase_name,
            sp500_ret_prior=sp500_ret_prior,
            require_sp500=not args.daytrade_precheck,
        )
    for ticker, moneyflow_score, detail in rebound_daytrade_candidates:
        df = price_map_all[ticker]
        close_price = float(detail.get("close", df.loc[signal_date, "Close"]))
        trigger = float(detail["trigger"])
        company = str(static_lookup[ticker].get("company_name") or static_lookup[ticker].get("company") or ticker.removesuffix(".T"))
        sector = str(static_lookup[ticker].get("simple_sector") or static_lookup[ticker].get("sector") or "UNKNOWN")
        buy_line, pick_line = format_buy(
            symbol=ticker,
            method="v38_post_crash_rebound_daytrade_precheck_entry"
            if args.daytrade_precheck
            else "v38_post_crash_rebound_daytrade_entry",
            sector=sector,
            company=company,
            close_price=close_price,
            entry_price=trigger,
            tp_prob=max(moneyflow_score, 0.0),
            tp_ratio=0.0,
            sl_ratio=0.0,
        )
        lines.append(buy_line)
        pick_lines.append(pick_line)

    if args.daytrade_only or args.daytrade_precheck:
        pass
    elif phase_name in V38_NO_ENTRY_PHASES:
        # v3.8 does not take new entries in crash or surge.
        pass
    elif etf_post_major_active:
        chosen_etf, trigger_price, etf_scores = v38_pick_best_etf_live(price_map_all, trade_date)
        if chosen_etf and trigger_price is not None:
            df = price_map_all[chosen_etf]
            basis_idx = df.index[df.index < trade_date]
            if len(basis_idx):
                close_price = float(df.loc[basis_idx[-1], "Close"])
                score = max(float(etf_scores.get(chosen_etf, 0.0)) * 100.0, 0.0)
                buy_line, pick_line = format_buy(
                    symbol=chosen_etf,
                    method="v38_post_crash_concentrated_etf_entry",
                    sector="ETF",
                    company=chosen_etf,
                    close_price=close_price,
                    entry_price=float(trigger_price),
                    tp_prob=score,
                    tp_ratio=V38_ETF_TP,
                    sl_ratio=V38_ETF_SL,
                )
                lines.append(buy_line)
                pick_lines.append(pick_line)
    elif stock_post_major_active:
        candidates = []
        for ticker in [t for t in price_map_all if t not in ETF_TICKERS]:
            if ticker not in active_lookup:
                continue
            df = price_map_all[ticker]
            basis_idx = df.index[df.index < trade_date]
            if not len(basis_idx):
                continue
            trigger = prev_high_break_trigger_live(df, basis_idx[-1])
            if trigger is None:
                continue
            metrics = base_mod.compute_metrics(df, basis_idx[-1])
            if not metrics:
                continue
            sig_rule, signal_score = base_mod.eval_signal(rule_name, metrics, active_lookup[ticker])
            if not sig_rule:
                continue
            if base_mod.trading_days_to_next_earnings(trade_date, df, earnings_map.get(ticker, [])) not in (None, 0):
                continue
            candidates.append((ticker, float(signal_score), float(trigger)))
        candidates.sort(key=lambda x: x[1], reverse=True)
        for ticker, signal_score, trigger in candidates[:1]:
            df = price_map_all[ticker]
            basis_idx = df.index[df.index < trade_date]
            close_price = float(df.loc[basis_idx[-1], "Close"])
            company = str(active_lookup[ticker].get("company_name") or active_lookup[ticker].get("company") or ticker.removesuffix(".T"))
            sector = str(active_lookup[ticker].get("simple_sector") or active_lookup[ticker].get("sector") or "UNKNOWN")
            buy_line, pick_line = format_buy(
                symbol=ticker,
                method="v38_post_crash_dispersed_prev_high_entry",
                sector=sector,
                company=company,
                close_price=close_price,
                entry_price=trigger,
                tp_prob=max(signal_score * 100.0, 0.0),
                tp_ratio=v38_stock_tp,
                sl_ratio=v38_stock_sl,
            )
            lines.append(buy_line)
            pick_lines.append(pick_line)
    else:
        candidates = []
        for ticker in [t for t in price_map_all if t not in ETF_TICKERS]:
            if ticker not in active_lookup:
                continue
            df = price_map_all[ticker]
            if signal_date not in df.index:
                continue
            metrics = base_mod.compute_metrics(df, signal_date)
            if not metrics:
                continue
            metrics["signal_basis_date"] = signal_date
            sig_rule, signal_score = base_mod.eval_signal(rule_name, metrics, active_lookup[ticker])
            if not sig_rule:
                continue
            if base_mod.trading_days_to_next_earnings(trade_date, df, earnings_map.get(ticker, [])) not in (None, 0):
                continue
            prev_close = float(df.loc[metrics["signal_basis_date"], "Close"])
            entry_limit = 1.5
            trigger = prev_close * (1.0 + entry_limit / 100.0)
            candidates.append((ticker, float(signal_score), float(trigger), metrics))
        candidates.sort(key=lambda x: x[1], reverse=True)
        for ticker, signal_score, trigger, metrics in candidates[:args.top_n]:
            df = price_map_all[ticker]
            close_price = float(df.loc[metrics["signal_basis_date"], "Close"])
            company = str(active_lookup[ticker].get("company_name") or active_lookup[ticker].get("company") or ticker.removesuffix(".T"))
            sector = str(active_lookup[ticker].get("simple_sector") or active_lookup[ticker].get("sector") or "UNKNOWN")
            buy_line, pick_line = format_buy(
                symbol=ticker,
                method=f"{rule_name}_entry",
                sector=sector,
                company=company,
                close_price=close_price,
                entry_price=trigger,
                tp_prob=max(signal_score * 100.0, 0.0),
                tp_ratio=0.08,
                sl_ratio=0.05,
            )
            lines.append(buy_line)
            pick_lines.append(pick_line)

    for line in lines:
        print(line)
    for line in pick_lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

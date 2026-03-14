from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.stock_signal import (
    FEATURE_COLS,
    build_features,
    download_daily_data,
    download_dow_futures_feature,
    download_index_returns,
    download_nikkei_futures_night_feature,
    download_sector_index_returns,
    fetch_symbol_sectors,
    load_sector_master,
    normalize_symbol,
    read_watchlist_groups,
    train_global_model,
)


@dataclass
class FoldResult:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    final_capital: float
    profit: float
    orders: int
    fills: int
    fill_rate: float
    win_rate: float
    avg_trade_ret: float
    max_drawdown: float
    objective_score: float


@dataclass
class Position:
    symbol: str
    exit_date: pd.Timestamp
    exit_value: float


def _parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _default_group_schedule() -> list[dict[str, Any]]:
    return [
        {"group": "group5", "start": "2025-01-01", "end": "2025-04-01"},
        {"group": "group4", "start": "2025-04-01", "end": "2025-07-01"},
        {"group": "group3", "start": "2025-07-01", "end": "2025-10-01"},
        {"group": "group2", "start": "2025-10-01", "end": "2025-12-31"},
    ]


def _load_group_schedule(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return _default_group_schedule()
    df = pd.read_csv(path)
    required = {"group", "start", "end"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"group schedule must include columns: {required}")
    out: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        out.append(
            {
                "group": str(row["group"]).strip(),
                "start": pd.Timestamp(str(row["start"])).strftime("%Y-%m-%d"),
                "end": pd.Timestamp(str(row["end"])).strftime("%Y-%m-%d"),
            }
        )
    return out


def _build_day_group_map(schedule: list[dict[str, Any]]) -> dict[pd.Timestamp, str]:
    day_group: dict[pd.Timestamp, str] = {}
    for row in schedule:
        g = str(row["group"])
        st = pd.Timestamp(row["start"]).normalize()
        en = pd.Timestamp(row["end"]).normalize()
        for d in pd.bdate_range(st, en):
            day_group[d.normalize()] = g
    return day_group


def _load_prices(symbols: list[str], period: str) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    price_dir = Path("data/prices")
    for sym in symbols:
        csv_path = price_dir / f"{sym.replace('.', '_')}.csv"
        df: pd.DataFrame | None = None
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                if "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"])
                    df = df.set_index("Date").sort_index()
            except Exception:
                df = None
        if df is None or df.empty or "Close" not in df.columns:
            try:
                df = download_daily_data(sym, period=period)
            except Exception:
                continue
        if df is not None and not df.empty and "Close" in df.columns:
            out[sym] = df.sort_index()
    return out


def _generate_folds(
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_months: int,
    test_months: int,
    step_months: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    folds: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    test_start = (start + pd.DateOffset(months=train_months)).normalize()
    fold_id = 0
    while test_start <= end:
        train_start = (test_start - pd.DateOffset(months=train_months)).normalize()
        train_end = (test_start - pd.Timedelta(days=1)).normalize()
        test_end = min((test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)).normalize(), end)
        if train_start >= start and train_start <= train_end and test_start <= test_end:
            folds.append((train_start, train_end, test_start, test_end))
        test_start = (test_start + pd.DateOffset(months=step_months)).normalize()
        fold_id += 1
        if fold_id > 120:
            break
    return folds


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _normal_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _expected_improvement(mu: np.ndarray, sigma: np.ndarray, best: float, xi: float = 0.01) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-9)
    imp = mu - best - xi
    z = imp / sigma
    cdf = np.vectorize(_normal_cdf)(z)
    pdf = np.vectorize(_normal_pdf)(z)
    return imp * cdf + sigma * pdf


def _build_regime_map(index_returns: pd.DataFrame) -> pd.Series:
    if "ret_n225" not in index_returns.columns:
        return pd.Series(dtype="object")
    ret = pd.to_numeric(index_returns["ret_n225"], errors="coerce")
    mom20 = ret.rolling(20).sum()
    vol20 = ret.rolling(20).std()
    regime = pd.Series("neutral", index=index_returns.index, dtype="object")
    regime[(mom20 > 0.01) & (vol20 < 0.02)] = "bull"
    regime[mom20 < -0.01] = "bear"
    return regime


def _build_local_sector_return_map(raw_map: dict[str, pd.DataFrame], sectors: dict[str, str]) -> dict[str, pd.Series]:
    ret_cols: dict[str, pd.Series] = {}
    for sym, df in raw_map.items():
        if "Close" not in df.columns:
            continue
        ret = pd.to_numeric(df["Close"], errors="coerce").pct_change()
        ret_cols[sym] = ret
    if not ret_cols:
        return {}
    ret_df = pd.DataFrame(ret_cols).sort_index()
    sec_map: dict[str, list[str]] = {}
    for sym in ret_df.columns:
        sec = str(sectors.get(sym, "UNKNOWN"))
        if not sec or sec == "UNKNOWN" or sec == "-":
            continue
        sec_map.setdefault(sec, []).append(sym)
    out: dict[str, pd.Series] = {}
    for sec, cols in sec_map.items():
        if len(cols) < 3:
            continue
        out[sec] = ret_df[cols].mean(axis=1).rename("sector_ret_1")
    return out


def _build_bullish_sectors_by_day(
    sector_ret_map: dict[str, pd.Series],
    short_window: int,
    long_window: int,
    ret_min: float,
) -> dict[pd.Timestamp, set[str]]:
    bullish_by_day: dict[pd.Timestamp, set[str]] = {}
    for sec, ret in sector_ret_map.items():
        s = pd.to_numeric(ret, errors="coerce").dropna()
        if len(s) < max(short_window, long_window) + 5:
            continue
        c = (1.0 + s).cumprod()
        ma_s = c.rolling(short_window).mean()
        ma_l = c.rolling(long_window).mean()
        ret_s = c.pct_change(short_window)
        # Relaxed sector bullish condition so signals are not too sparse.
        is_bull = (ma_s > ma_l) & (ret_s > ret_min)
        for d, v in is_bull.dropna().items():
            if bool(v):
                dn = pd.Timestamp(d).normalize()
                bullish_by_day.setdefault(dn, set()).add(sec)
    return bullish_by_day


def _dynamic_params_by_regime(
    day: pd.Timestamp,
    base_threshold: float,
    base_tp: float,
    base_sl: float,
    base_horizon: int,
    regime_map: pd.Series,
    regime_enabled: bool,
    bull_hold_bonus: int,
    bear_hold_penalty: int,
    bull_threshold_shift: float,
    bear_threshold_shift: float,
    bull_tp_shift: float,
    bear_tp_shift: float,
    bull_sl_shift: float,
    bear_sl_shift: float,
    min_horizon: int,
    max_horizon: int,
) -> tuple[float, float, float, int]:
    if not regime_enabled or regime_map.empty:
        return base_threshold, base_tp, base_sl, base_horizon
    regime = regime_map.get(day, "neutral")
    threshold = base_threshold
    tp = base_tp
    sl = base_sl
    horizon = base_horizon
    if regime == "bull":
        threshold = max(0.0, min(1.0, threshold + bull_threshold_shift))
        tp = max(0.01, tp + bull_tp_shift)
        sl = max(0.01, sl + bull_sl_shift)
        horizon = min(max_horizon, horizon + bull_hold_bonus)
    elif regime == "bear":
        threshold = max(0.0, min(1.0, threshold + bear_threshold_shift))
        tp = max(0.01, tp + bear_tp_shift)
        sl = max(0.01, sl + bear_sl_shift)
        horizon = max(min_horizon, horizon - bear_hold_penalty)
    horizon = max(min_horizon, min(max_horizon, int(horizon)))
    return threshold, tp, sl, horizon


def _simulate_trade_path(
    df: pd.DataFrame,
    ni: int,
    end_i: int,
    entry: float,
    shares: int,
    tp: float,
    sl: float,
    trailing_enabled: bool,
    trailing_ratio: float,
    trailing_activate: float,
    partial_tp_enabled: bool,
    partial_ratio: float,
) -> tuple[float, pd.Timestamp]:
    base_tp_price = entry * (1.0 + tp)
    sl_price = entry * (1.0 - sl)
    peak = entry
    remaining = int(shares)
    realized = 0.0
    partial_done = False
    exit_date = pd.Timestamp(df.index[end_i]).normalize()

    for j in range(ni, end_i + 1):
        hi = float(df["High"].iloc[j]) if "High" in df.columns else float(df["Close"].iloc[j])
        lo = float(df["Low"].iloc[j]) if "Low" in df.columns else float(df["Close"].iloc[j])
        d = pd.Timestamp(df.index[j]).normalize()
        peak = max(peak, hi)

        if trailing_enabled and peak >= entry * (1.0 + trailing_activate):
            trail_stop = peak * (1.0 - trailing_ratio)
            sl_price = max(sl_price, trail_stop)

        # Conservative order: stop hit first if both touched intraday.
        if lo <= sl_price:
            realized += remaining * sl_price
            remaining = 0
            exit_date = d
            break

        if partial_tp_enabled and (not partial_done) and hi >= base_tp_price:
            sell_shares = max(1, int(round(shares * partial_ratio)))
            sell_shares = min(sell_shares, remaining)
            realized += sell_shares * base_tp_price
            remaining -= sell_shares
            partial_done = True
            sl_price = max(sl_price, entry)
        elif (not partial_tp_enabled) and hi >= base_tp_price:
            realized += remaining * base_tp_price
            remaining = 0
            exit_date = d
            break

    if remaining > 0:
        realized += remaining * float(df["Close"].iloc[end_i])
    return realized, exit_date


def _simulate_fold(
    prob_maps: dict[str, dict[str, pd.Series]],
    raw_map: dict[str, pd.DataFrame],
    day_group: dict[pd.Timestamp, str],
    regime_map: pd.Series,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    threshold: float,
    limit_ratio: float,
    take_profit: float,
    stop_loss: float,
    horizon_days: int,
    initial_capital: float,
    max_per_symbol: float,
    lot_size: int,
    position_sizing: str,
    daily_capital_ratio: float,
    min_score_eps: float,
    min_score_frac_of_max: float,
    max_daily_trades: int,
    allow_skip_small_target: bool,
    regime_enabled: bool,
    bull_hold_bonus: int,
    bear_hold_penalty: int,
    bull_threshold_shift: float,
    bear_threshold_shift: float,
    bull_tp_shift: float,
    bear_tp_shift: float,
    bull_sl_shift: float,
    bear_sl_shift: float,
    min_horizon: int,
    max_horizon: int,
    trailing_enabled: bool,
    trailing_ratio: float,
    trailing_activate: float,
    partial_tp_enabled: bool,
    partial_tp_ratio: float,
    sector_extra_enabled: bool,
    sector_extra_max_symbols: int,
    sector_extra_min_prob: float,
    sectors: dict[str, str],
    group_base_symbols: dict[str, set[str]],
    bullish_sectors_by_day: dict[pd.Timestamp, set[str]],
    fee_bps: float,
    entry_slippage_bps: float,
    exit_slippage_bps: float,
    lambda_dd: float,
    fold_id: int,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
) -> FoldResult:
    cash = initial_capital
    open_pos: list[Position] = []
    equity_rows: list[tuple[pd.Timestamp, float]] = []
    trade_rets: list[float] = []
    orders = 0
    fills = 0

    for day in pd.bdate_range(test_start, test_end):
        day = day.normalize()
        still_open: list[Position] = []
        for pos in open_pos:
            if pos.exit_date == day:
                cash += pos.exit_value
            else:
                still_open.append(pos)
        open_pos = still_open

        group = day_group.get(day)
        if group is None:
            equity_rows.append((day, cash + sum(p.exit_value for p in open_pos)))
            continue

        held = {p.symbol for p in open_pos}
        day_threshold, day_tp, day_sl, day_horizon = _dynamic_params_by_regime(
            day=day,
            base_threshold=threshold,
            base_tp=take_profit,
            base_sl=stop_loss,
            base_horizon=horizon_days,
            regime_map=regime_map,
            regime_enabled=regime_enabled,
            bull_hold_bonus=bull_hold_bonus,
            bear_hold_penalty=bear_hold_penalty,
            bull_threshold_shift=bull_threshold_shift,
            bear_threshold_shift=bear_threshold_shift,
            bull_tp_shift=bull_tp_shift,
            bear_tp_shift=bear_tp_shift,
            bull_sl_shift=bull_sl_shift,
            bear_sl_shift=bear_sl_shift,
            min_horizon=min_horizon,
            max_horizon=max_horizon,
        )
        group_probs = prob_maps.get(group, {})
        base_syms = group_base_symbols.get(group, set())
        candidate_symbols = [s for s in group_probs.keys() if s in base_syms]
        if sector_extra_enabled and sector_extra_max_symbols > 0:
            bull = bullish_sectors_by_day.get(day, set())
            if bull:
                extra_ranked = sorted(
                    [
                        (s, float(group_probs[s].get(day)))
                        for s in group_probs.keys()
                        if s not in base_syms
                        and str(sectors.get(s, "UNKNOWN")) in bull
                        and s in group_probs
                        and np.isfinite(float(group_probs[s].get(day)))
                        and float(group_probs[s].get(day)) >= sector_extra_min_prob
                    ],
                    key=lambda x: x[1],
                    reverse=True,
                )
                candidate_symbols.extend([s for s, _ in extra_ranked[:sector_extra_max_symbols]])
        candidates: list[dict[str, Any]] = []
        for sym in candidate_symbols:
            ser = group_probs.get(sym)
            if ser is None:
                continue
            if sym in held:
                continue
            prob = ser.get(day)
            if pd.isna(prob) or float(prob) < day_threshold:
                continue
            df = raw_map.get(sym)
            if df is None or day not in df.index:
                continue
            i = df.index.get_loc(day)
            if isinstance(i, slice):
                i = i.stop - 1
            if i + 1 >= len(df):
                continue

            close = float(df["Close"].iloc[i])
            entry = close * (1.0 - limit_ratio)
            low_n = float(df["Low"].iloc[i + 1]) if "Low" in df.columns else float(df["Close"].iloc[i + 1])
            high_n = float(df["High"].iloc[i + 1]) if "High" in df.columns else float(df["Close"].iloc[i + 1])
            orders += 1
            if not (low_n <= entry <= high_n):
                continue

            lot_cost = entry * lot_size
            max_lots = int(max_per_symbol // lot_cost)
            if max_lots < 1:
                max_lots = 1
            vol20 = float(df["Close"].pct_change().rolling(20).std().iloc[i])
            if not np.isfinite(vol20) or vol20 <= 0:
                vol20 = 0.02
            score_prob = max(float(prob) - day_threshold, min_score_eps)
            score = score_prob / max(vol20, 1e-4)
            candidates.append(
                {
                    "prob": float(prob),
                    "sym": sym,
                    "i": int(i),
                    "entry": float(entry),
                    "lot_cost": float(lot_cost),
                    "max_lots": int(max_lots),
                    "score": float(score),
                }
            )

        if position_sizing == "prob_vol":
            candidates.sort(key=lambda x: x["score"], reverse=True)
            if candidates and min_score_frac_of_max > 0:
                max_score = max(c["score"] for c in candidates)
                score_floor = max_score * min_score_frac_of_max
                candidates = [c for c in candidates if c["score"] >= score_floor]
                candidates.sort(key=lambda x: x["score"], reverse=True)
            if max_daily_trades > 0:
                candidates = candidates[:max_daily_trades]
            total_score = sum(c["score"] for c in candidates)
            daily_budget = min(cash, max(0.0, cash * daily_capital_ratio))
        else:
            candidates.sort(key=lambda x: x["prob"], reverse=True)
            if max_daily_trades > 0:
                candidates = candidates[:max_daily_trades]
            total_score = 0.0
            daily_budget = cash

        for c in candidates:
            sym = str(c["sym"])
            i = int(c["i"])
            entry = float(c["entry"])
            lot_cost = float(c["lot_cost"])
            max_lots = int(c["max_lots"])
            if position_sizing == "prob_vol" and total_score > 0:
                target_cost = min(max_per_symbol, daily_budget * (c["score"] / total_score))
                lots = int(target_cost // lot_cost)
                if lots < 1 and not allow_skip_small_target:
                    lots = 1
                if lots < 1 and allow_skip_small_target:
                    continue
                lots = min(lots, max_lots)
            else:
                lots = max_lots
            if lots < 1:
                continue
            shares = lots * lot_size
            exec_entry = entry * (1.0 + (entry_slippage_bps / 10_000.0))
            cost = exec_entry * shares
            entry_fee = cost * (fee_bps / 10_000.0)
            if cost > cash:
                continue
            df = raw_map[sym]
            ni = i + 1
            end_i = min(len(df) - 1, ni + day_horizon - 1)
            exit_value, exit_date = _simulate_trade_path(
                df=df,
                ni=ni,
                end_i=end_i,
                entry=exec_entry,
                shares=shares,
                tp=day_tp,
                sl=day_sl,
                trailing_enabled=trailing_enabled,
                trailing_ratio=trailing_ratio,
                trailing_activate=trailing_activate,
                partial_tp_enabled=partial_tp_enabled,
                partial_ratio=partial_tp_ratio,
            )
            exit_value = exit_value * (1.0 - (exit_slippage_bps / 10_000.0))
            exit_fee = exit_value * (fee_bps / 10_000.0)

            cash -= (cost + entry_fee)
            open_pos.append(Position(symbol=sym, exit_date=exit_date, exit_value=(exit_value - exit_fee)))
            fills += 1
            trade_rets.append((exit_value - exit_fee) / (cost + entry_fee) - 1.0)

        equity_rows.append((day, cash + sum(p.exit_value for p in open_pos)))

    for pos in open_pos:
        cash += pos.exit_value

    eq = pd.DataFrame(equity_rows, columns=["date", "equity"]).drop_duplicates("date").sort_values("date")
    if len(eq) > 0:
        peak = eq["equity"].cummax()
        max_dd = float(((peak - eq["equity"]) / peak).max())
    else:
        max_dd = 0.0

    tr = pd.Series(trade_rets, dtype=float)
    fill_rate = (fills / orders) if orders > 0 else 0.0
    win_rate = float((tr > 0).mean()) if len(tr) else 0.0
    avg_trade_ret = float(tr.mean()) if len(tr) else 0.0
    final_cap = float(cash)
    score = final_cap - (lambda_dd * initial_capital * max_dd)
    return FoldResult(
        fold_id=fold_id,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        final_capital=final_cap,
        profit=final_cap - initial_capital,
        orders=orders,
        fills=fills,
        fill_rate=fill_rate,
        win_rate=win_rate,
        avg_trade_ret=avg_trade_ret,
        max_drawdown=max_dd,
        objective_score=score,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Walk-forward + Bayesian optimization for stock strategy.")
    parser.add_argument("--watchlist", type=Path, default=Path("watchlist.csv"))
    parser.add_argument("--group-schedule", type=Path, default=None)
    parser.add_argument("--period", type=str, default="max")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--start", type=str, default="2025-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    parser.add_argument("--train-months", type=int, default=24)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--step-months", type=int, default=3)

    parser.add_argument("--horizon-values", type=str, default="5,10,15,20,25,30,35,40")
    parser.add_argument("--threshold-values", type=str, default="0.60,0.61,0.62,0.63,0.64")
    parser.add_argument("--limit-values", type=str, default="-0.01,-0.0075,-0.005,-0.0025,0.0,0.002")
    parser.add_argument("--take-profit-values", type=str, default="0.05,0.06,0.07")
    parser.add_argument("--stop-loss-values", type=str, default="0.10,0.12,0.14")

    parser.add_argument("--initial-capital", type=float, default=3_000_000.0)
    parser.add_argument("--max-per-symbol", type=float, default=500_000.0)
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument("--lambda-dd", type=float, default=1.0)
    parser.add_argument("--lambda-values", type=str, default="", help="Comma list. If set, overrides --lambda-dd.")
    parser.add_argument(
        "--position-sizing",
        type=str,
        choices=["fixed", "prob_vol"],
        default="prob_vol",
        help="Position sizing mode. prob_vol uses probability and 20-day volatility.",
    )
    parser.add_argument(
        "--daily-capital-ratio",
        type=float,
        default=1.0,
        help="Daily deployable capital ratio for prob_vol sizing.",
    )
    parser.add_argument(
        "--min-score-eps",
        type=float,
        default=0.005,
        help="Minimum excess probability used in prob_vol score.",
    )
    parser.add_argument(
        "--min-score-frac-of-max",
        type=float,
        default=0.20,
        help="In prob_vol, keep candidates with score >= max_score * this value.",
    )
    parser.add_argument(
        "--max-daily-trades",
        type=int,
        default=5,
        help="Maximum new entries per day (0 means unlimited).",
    )
    parser.add_argument(
        "--allow-skip-small-target",
        action="store_true",
        default=True,
        help="Skip candidate when target allocation is below 1 lot in prob_vol mode.",
    )
    parser.add_argument(
        "--no-allow-skip-small-target",
        action="store_true",
        help="Force minimum 1 lot even when target allocation is below 1 lot.",
    )
    parser.add_argument("--regime-enabled", action="store_true", default=True)
    parser.add_argument("--bull-hold-bonus", type=int, default=5)
    parser.add_argument("--bear-hold-penalty", type=int, default=5)
    parser.add_argument("--bull-threshold-shift", type=float, default=-0.005)
    parser.add_argument("--bear-threshold-shift", type=float, default=0.01)
    parser.add_argument("--bull-tp-shift", type=float, default=0.005)
    parser.add_argument("--bear-tp-shift", type=float, default=-0.005)
    parser.add_argument("--bull-sl-shift", type=float, default=0.0)
    parser.add_argument("--bear-sl-shift", type=float, default=-0.02)
    parser.add_argument("--min-horizon", type=int, default=5)
    parser.add_argument("--max-horizon", type=int, default=40)
    parser.add_argument("--trailing-enabled", action="store_true", default=True)
    parser.add_argument("--trailing-ratio", type=float, default=0.04)
    parser.add_argument("--trailing-activate", type=float, default=0.03)
    parser.add_argument("--partial-tp-enabled", action="store_true", default=True)
    parser.add_argument("--partial-tp-ratio", type=float, default=0.5)
    parser.add_argument("--fee-bps", type=float, default=10.0, help="Per-side fee in bps.")
    parser.add_argument("--entry-slippage-bps", type=float, default=5.0, help="Entry adverse slippage in bps.")
    parser.add_argument("--exit-slippage-bps", type=float, default=5.0, help="Exit adverse slippage in bps.")
    parser.add_argument("--no-sector-extra", action="store_true", help="Disable sector-signal based extra symbols.")
    parser.add_argument("--sector-extra-max-symbols", type=int, default=10)
    parser.add_argument("--sector-extra-min-prob", type=float, default=0.56)
    parser.add_argument("--sector-signal-short-window", type=int, default=5)
    parser.add_argument("--sector-signal-long-window", type=int, default=20)
    parser.add_argument("--sector-signal-ret-min", type=float, default=-0.004)
    parser.add_argument("--sector-master", type=Path, default=Path("data/sector_master_template.csv"))

    parser.add_argument("--bo-init", type=int, default=12)
    parser.add_argument("--bo-iter", type=int, default=40)
    parser.add_argument("--bo-candidates", type=int, default=500)

    parser.add_argument("--out-trials", type=Path, default=Path("data/walkforward_bo_trials.csv"))
    parser.add_argument("--out-folds", type=Path, default=Path("data/walkforward_bo_best_folds.csv"))
    args = parser.parse_args()
    if args.no_allow_skip_small_target:
        args.allow_skip_small_target = False

    random.seed(args.seed)
    np.random.seed(args.seed)

    schedule = _load_group_schedule(args.group_schedule)
    day_group = _build_day_group_map(schedule)
    grouped = read_watchlist_groups(args.watchlist)
    used_symbols = sorted(
        {
            normalize_symbol(s)
            for row in schedule
            for s in grouped.get(str(row["group"]), [])
        }
    )
    raw_map = _load_prices(used_symbols, period=args.period)
    print(f"[INFO] symbols={len(used_symbols)} loaded={len(raw_map)}")
    if not raw_map:
        raise ValueError("no symbols loaded")

    index_returns = download_index_returns(period=args.period)
    futures_feature = download_nikkei_futures_night_feature()
    dow_futures_feature = download_dow_futures_feature(period=args.period)
    sector_master = load_sector_master(args.sector_master) if args.sector_master else {}
    sectors = {s: sector_master.get(s, "UNKNOWN") for s in raw_map.keys()}
    unresolved = [s for s, sec in sectors.items() if sec in ("UNKNOWN", "", "-")]
    if unresolved:
        fetched = fetch_symbol_sectors(unresolved)
        for s in unresolved:
            v = str(fetched.get(s, "UNKNOWN"))
            if v not in ("UNKNOWN", "", "-"):
                sectors[s] = v
    unknown_n = sum(1 for sec in sectors.values() if str(sec) in ("UNKNOWN", "", "-"))
    print(f"[INFO] sector_master={len(sector_master)} unknown_after_merge={unknown_n}")
    sector_ret_map = download_sector_index_returns(period=args.period, sectors=set(sectors.values()))
    local_sector_ret_map = _build_local_sector_return_map(raw_map=raw_map, sectors=sectors)
    bullish_sectors_by_day = _build_bullish_sectors_by_day(
        sector_ret_map=local_sector_ret_map,
        short_window=args.sector_signal_short_window,
        long_window=args.sector_signal_long_window,
        ret_min=args.sector_signal_ret_min,
    )
    print(f"[INFO] local_sector_series={len(local_sector_ret_map)} bullish_days={len(bullish_sectors_by_day)}")
    regime_map = _build_regime_map(index_returns=index_returns)
    group_base_symbols: dict[str, set[str]] = {}
    for row in schedule:
        g = str(row["group"])
        group_base_symbols[g] = {normalize_symbol(s) for s in grouped.get(g, []) if normalize_symbol(s) in raw_map}

    global_start = pd.Timestamp(args.start).normalize()
    global_end = pd.Timestamp(args.end).normalize()
    folds = _generate_folds(
        start=global_start,
        end=global_end,
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=args.step_months,
    )
    if not folds:
        raise ValueError("no folds generated; adjust train/test window")
    print(f"[INFO] folds={len(folds)} window={global_start.date()}..{global_end.date()}")

    horizon_vals = _parse_int_list(args.horizon_values)
    threshold_vals = _parse_float_list(args.threshold_values)
    limit_vals = _parse_float_list(args.limit_values)
    tp_vals = _parse_float_list(args.take_profit_values)
    sl_vals = _parse_float_list(args.stop_loss_values)

    all_combos: list[tuple[int, float, float, float, float]] = [
        (h, th, lim, tp, sl)
        for h in horizon_vals
        for th in threshold_vals
        for lim in limit_vals
        for tp in tp_vals
        for sl in sl_vals
    ]
    print(f"[INFO] search_space={len(all_combos)}")
    if args.lambda_values.strip():
        lambda_values = _parse_float_list(args.lambda_values)
    else:
        lambda_values = [args.lambda_dd]
    print(
        f"[INFO] lambdas={lambda_values} position_sizing={args.position_sizing} "
        f"regime={args.regime_enabled} trailing={args.trailing_enabled} partial_tp={args.partial_tp_enabled}"
    )

    feature_cache: dict[tuple[int, float], dict[str, pd.DataFrame]] = {}

    def get_feature_map(h: int, tp: float) -> dict[str, pd.DataFrame]:
        key = (h, tp)
        if key in feature_cache:
            return feature_cache[key]
        fmap: dict[str, pd.DataFrame] = {}
        for sym, raw_df in raw_map.items():
            sector = sectors.get(sym, "UNKNOWN")
            feat = build_features(
                df=raw_df,
                index_returns=index_returns,
                futures_feature=futures_feature,
                dow_futures_feature=dow_futures_feature,
                sector=sector,
                sector_ret_map=sector_ret_map,
                horizon_days=h,
                take_profit=tp,
            )
            fmap[sym] = feat
        feature_cache[key] = fmap
        return fmap

    all_trials_frames: list[pd.DataFrame] = []
    best_summaries: list[dict[str, Any]] = []
    best_fold_rows_global: list[dict[str, Any]] = []

    for lambda_dd in lambda_values:
        def evaluate_combo(combo: tuple[int, float, float, float, float]) -> tuple[float, dict[str, float], list[FoldResult]]:
            h, th, lim, tp, sl = combo
            fmap = get_feature_map(h, tp)
            fold_rows: list[FoldResult] = []
            for idx, (tr_st, tr_en, te_st, te_en) in enumerate(folds, start=1):
                models: dict[str, Any] = {}
                prob_maps: dict[str, dict[str, pd.Series]] = {}
                for row in schedule:
                    group = str(row["group"])
                    syms = [normalize_symbol(s) for s in grouped.get(group, []) if normalize_symbol(s) in raw_map]
                    eval_syms = sorted(set(raw_map.keys()) if not args.no_sector_extra else set(syms))
                    train_frames: list[pd.DataFrame] = []
                    for sym in syms:
                        feat = fmap[sym]
                        labeled = feat.dropna(subset=FEATURE_COLS + ["target"]).copy()
                        labeled = labeled[(labeled.index >= tr_st) & (labeled.index <= tr_en)]
                        if not labeled.empty:
                            labeled["symbol"] = sym
                            train_frames.append(labeled)
                    if not train_frames:
                        continue
                    train_df = pd.concat(train_frames, axis=0)
                    try:
                        model = train_global_model(train_df)
                    except Exception:
                        continue
                    models[group] = model
                    pmap: dict[str, pd.Series] = {}
                    for sym in eval_syms:
                        feat = fmap[sym].dropna(subset=FEATURE_COLS).copy()
                        feat = feat[(feat.index >= te_st) & (feat.index <= te_en)]
                        if feat.empty:
                            continue
                        probs = model.predict_proba(feat[FEATURE_COLS])[:, 1]
                        pmap[sym] = pd.Series(probs, index=feat.index)
                    prob_maps[group] = pmap

                fold_result = _simulate_fold(
                    prob_maps=prob_maps,
                    raw_map=raw_map,
                    day_group=day_group,
                    regime_map=regime_map,
                    test_start=te_st,
                    test_end=te_en,
                    threshold=th,
                    limit_ratio=lim,
                    take_profit=tp,
                    stop_loss=sl,
                    horizon_days=h,
                    initial_capital=args.initial_capital,
                    max_per_symbol=args.max_per_symbol,
                    lot_size=args.lot_size,
                    position_sizing=args.position_sizing,
                    daily_capital_ratio=args.daily_capital_ratio,
                    min_score_eps=args.min_score_eps,
                    min_score_frac_of_max=args.min_score_frac_of_max,
                    max_daily_trades=args.max_daily_trades,
                    allow_skip_small_target=args.allow_skip_small_target,
                    regime_enabled=args.regime_enabled,
                    bull_hold_bonus=args.bull_hold_bonus,
                    bear_hold_penalty=args.bear_hold_penalty,
                    bull_threshold_shift=args.bull_threshold_shift,
                    bear_threshold_shift=args.bear_threshold_shift,
                    bull_tp_shift=args.bull_tp_shift,
                    bear_tp_shift=args.bear_tp_shift,
                    bull_sl_shift=args.bull_sl_shift,
                    bear_sl_shift=args.bear_sl_shift,
                    min_horizon=args.min_horizon,
                    max_horizon=args.max_horizon,
                    trailing_enabled=args.trailing_enabled,
                    trailing_ratio=args.trailing_ratio,
                    trailing_activate=args.trailing_activate,
                    partial_tp_enabled=args.partial_tp_enabled,
                    partial_tp_ratio=args.partial_tp_ratio,
                    sector_extra_enabled=not args.no_sector_extra,
                    sector_extra_max_symbols=args.sector_extra_max_symbols,
                    sector_extra_min_prob=args.sector_extra_min_prob,
                    sectors=sectors,
                    group_base_symbols=group_base_symbols,
                    bullish_sectors_by_day=bullish_sectors_by_day,
                    fee_bps=args.fee_bps,
                    entry_slippage_bps=args.entry_slippage_bps,
                    exit_slippage_bps=args.exit_slippage_bps,
                    lambda_dd=lambda_dd,
                    fold_id=idx,
                    train_start=tr_st,
                    train_end=tr_en,
                )
                fold_rows.append(fold_result)

            if not fold_rows:
                return -1e18, {"final_capital_mean": 0.0, "max_drawdown_mean": 1.0, "objective_mean": -1e18}, []

            final_cap_mean = float(np.mean([f.final_capital for f in fold_rows]))
            dd_mean = float(np.mean([f.max_drawdown for f in fold_rows]))
            score_mean = float(np.mean([f.objective_score for f in fold_rows]))
            summary = {
                "final_capital_mean": final_cap_mean,
                "max_drawdown_mean": dd_mean,
                "objective_mean": score_mean,
            }
            return score_mean, summary, fold_rows

        index_map = {c: i for i, c in enumerate(all_combos)}
        tried: set[int] = set()
        trials: list[dict[str, Any]] = []
        best_score = -1e18
        best_combo: tuple[int, float, float, float, float] | None = None
        best_folds: list[FoldResult] = []

        def run_one(combo: tuple[int, float, float, float, float], phase: str) -> None:
            nonlocal best_score, best_combo, best_folds
            idx = index_map[combo]
            if idx in tried:
                return
            tried.add(idx)
            score, summary, folds_rows = evaluate_combo(combo)
            h, th, lim, tp, sl = combo
            row = {
                "lambda_dd": lambda_dd,
                "phase": phase,
                "horizon": h,
                "threshold": th,
                "limit": lim,
                "take_profit": tp,
                "stop_loss": sl,
                "objective_mean": score,
                "final_capital_mean": summary["final_capital_mean"],
                "max_drawdown_mean": summary["max_drawdown_mean"],
                "folds": len(folds_rows),
            }
            trials.append(row)
            print(
                f"[lambda={lambda_dd:.3f}][{phase}] h={h} th={th:.2f} lim={lim:.2%} tp={tp:.2%} sl={sl:.2%} "
                f"obj={score:,.0f} cap_mean={summary['final_capital_mean']:,.0f} dd_mean={summary['max_drawdown_mean']:.2%}"
            )
            if score > best_score:
                best_score = score
                best_combo = combo
                best_folds = folds_rows

        init_n = min(args.bo_init, len(all_combos))
        for combo in random.sample(all_combos, k=init_n):
            run_one(combo, phase="init")

        for it in range(1, args.bo_iter + 1):
            if len(tried) >= len(all_combos):
                break
            x_train = []
            y_train = []
            for row in trials:
                c = (
                    int(row["horizon"]),
                    float(row["threshold"]),
                    float(row["limit"]),
                    float(row["take_profit"]),
                    float(row["stop_loss"]),
                )
                x_train.append(
                    [
                        horizon_vals.index(c[0]) / max(1, len(horizon_vals) - 1),
                        threshold_vals.index(c[1]) / max(1, len(threshold_vals) - 1),
                        limit_vals.index(c[2]) / max(1, len(limit_vals) - 1),
                        tp_vals.index(c[3]) / max(1, len(tp_vals) - 1),
                        sl_vals.index(c[4]) / max(1, len(sl_vals) - 1),
                    ]
                )
                y_train.append(float(row["objective_mean"]))
            x_train_arr = np.array(x_train, dtype=float)
            y_train_arr = np.array(y_train, dtype=float)

            kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-5)
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=args.seed)
            gp.fit(x_train_arr, y_train_arr)

            remaining = [c for c in all_combos if index_map[c] not in tried]
            pool = random.sample(remaining, k=min(args.bo_candidates, len(remaining)))
            x_pool = np.array(
                [
                    [
                        horizon_vals.index(c[0]) / max(1, len(horizon_vals) - 1),
                        threshold_vals.index(c[1]) / max(1, len(threshold_vals) - 1),
                        limit_vals.index(c[2]) / max(1, len(limit_vals) - 1),
                        tp_vals.index(c[3]) / max(1, len(tp_vals) - 1),
                        sl_vals.index(c[4]) / max(1, len(sl_vals) - 1),
                    ]
                    for c in pool
                ],
                dtype=float,
            )
            mu, sigma = gp.predict(x_pool, return_std=True)
            ei = _expected_improvement(mu=mu, sigma=sigma, best=best_score)
            chosen = pool[int(np.argmax(ei))]
            run_one(chosen, phase=f"bo{it:02d}")

        trials_df = pd.DataFrame(trials).sort_values("objective_mean", ascending=False)
        all_trials_frames.append(trials_df)

        if best_combo is None:
            continue
        h, th, lim, tp, sl = best_combo
        best_summaries.append(
            {
                "lambda_dd": lambda_dd,
                "horizon": h,
                "threshold": th,
                "limit": lim,
                "take_profit": tp,
                "stop_loss": sl,
                "objective_mean": best_score,
                "final_capital_mean": float(np.mean([f.final_capital for f in best_folds])) if best_folds else 0.0,
                "max_drawdown_mean": float(np.mean([f.max_drawdown for f in best_folds])) if best_folds else 0.0,
                "folds": len(best_folds),
            }
        )
        for f in best_folds:
            row = dict(f.__dict__)
            row["lambda_dd"] = lambda_dd
            row["best_horizon"] = h
            row["best_threshold"] = th
            row["best_limit"] = lim
            row["best_take_profit"] = tp
            row["best_stop_loss"] = sl
            best_fold_rows_global.append(row)

        print("\n[BEST]")
        print(
            json.dumps(
                {
                    "lambda_dd": lambda_dd,
                    "horizon": h,
                    "threshold": th,
                    "limit": lim,
                    "take_profit": tp,
                    "stop_loss": sl,
                    "objective_mean": best_score,
                    "folds": len(best_folds),
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    if not all_trials_frames:
        raise ValueError("no valid optimization result")
    out_trials_df = pd.concat(all_trials_frames, axis=0).sort_values(
        ["lambda_dd", "objective_mean"], ascending=[True, False]
    )
    args.out_trials.parent.mkdir(parents=True, exist_ok=True)
    out_trials_df.to_csv(args.out_trials, index=False)

    if best_fold_rows_global:
        out_fold_df = pd.DataFrame(best_fold_rows_global)
    else:
        out_fold_df = pd.DataFrame()
    args.out_folds.parent.mkdir(parents=True, exist_ok=True)
    out_fold_df.to_csv(args.out_folds, index=False)

    if best_summaries:
        summary_df = pd.DataFrame(best_summaries).sort_values(
            ["final_capital_mean", "max_drawdown_mean"], ascending=[False, True]
        )
        summary_path = args.out_trials.with_name(args.out_trials.stem + "_best_by_lambda.csv")
        summary_df.to_csv(summary_path, index=False)
        print("\n[BEST_BY_LAMBDA]")
        print(summary_df.to_string(index=False))
        print(f"[OUT] summary={summary_path}")
    print(f"[OUT] trials={args.out_trials}")
    print(f"[OUT] folds={args.out_folds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

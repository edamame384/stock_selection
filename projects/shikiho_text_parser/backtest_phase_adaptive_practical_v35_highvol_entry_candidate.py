from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.shikiho_text_parser.backtest_phase_adaptive import compute_metrics, eval_signal  # noqa: E402
from projects.shikiho_text_parser.backtest_phase_adaptive_practical_v2 import practical_v2_mapping  # noqa: E402
from projects.shikiho_text_parser.compare_phase_adaptive_practical_v3 import TableSpec, build_snapshot, load_price_map, normalize_static  # noqa: E402
from projects.quarterly_ranker.select_2024_q2_pre_jul_candidates import select_candidates as select_q2_candidates  # noqa: E402
from projects.quarterly_ranker.select_4q_pre_oct_candidates import select_candidates as select_q4_candidates  # noqa: E402
from projects.shikiho_text_parser.select_promising_4q2 import score_universe as score_4q2_universe  # noqa: E402
from projects.shikiho_text_parser.methods.method_post_crash_broad import SELECTION_RULE, TRADING_RULE  # noqa: E402
from projects.shikiho_text_parser.search_phase_method_optimization import PHASE_CSV, load_phase_map  # noqa: E402


EARNINGS_CACHE_CSV = ROOT / "data" / "earnings_cache" / "yf_earnings_dates.csv"
IRBANK_EARNINGS_CACHE_CSV = ROOT / "data" / "earnings_cache" / "irbank_earnings_dates.csv"
RAW_4Q2_DIR = ROOT / "projects" / "shikiho_text_parser" / "data" / "raw" / "4Q-2"
PARSED_SUMMARY_CSV = ROOT / "projects" / "shikiho_text_parser" / "output" / "parsed_summary.csv"


def _normalize_earnings_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    idx = pd.to_datetime(df.index, errors="coerce")
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("Asia/Tokyo").tz_localize(None)
    return pd.DatetimeIndex(idx.normalize())


def fetch_earnings_cache(tickers: list[str], cache_csv: Path = EARNINGS_CACHE_CSV) -> pd.DataFrame:
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    if cache_csv.exists():
        cache = pd.read_csv(cache_csv)
    else:
        cache = pd.DataFrame(columns=["ticker", "earnings_date"])

    have = set(cache["ticker"].astype(str)) if not cache.empty else set()
    missing = [ticker for ticker in sorted(set(tickers)) if ticker not in have]
    if missing:
        cache_dir = ROOT / "data" / "yf_cache_earnings"
        cache_dir.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(cache_dir))
        for ticker in missing:
            try:
                ed = yf.Ticker(ticker).get_earnings_dates(limit=16)
                if ed is None or ed.empty:
                    continue
                rows: list[dict[str, str]] = []
                for dt in _normalize_earnings_index(ed):
                    if pd.isna(dt):
                        continue
                    rows.append({"ticker": ticker, "earnings_date": dt.strftime("%Y-%m-%d")})
                if rows:
                    cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
                    cache = cache.drop_duplicates(subset=["ticker", "earnings_date"]).sort_values(["ticker", "earnings_date"])
                    cache.to_csv(cache_csv, index=False, encoding="utf-8-sig")
            except Exception:
                continue
            time.sleep(0.1)

    if cache.empty:
        return cache
    cache["earnings_date"] = pd.to_datetime(cache["earnings_date"])
    return cache


def fetch_irbank_earnings_cache(tickers: list[str], cache_csv: Path = IRBANK_EARNINGS_CACHE_CSV) -> pd.DataFrame:
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    if cache_csv.exists():
        cache = pd.read_csv(cache_csv)
    else:
        cache = pd.DataFrame(columns=["ticker", "earnings_date", "source"])

    have = set(cache["ticker"].astype(str)) if not cache.empty else set()
    missing = [ticker for ticker in sorted(set(tickers)) if ticker not in have]
    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0"}

    for ticker in missing:
        code = ticker.replace(".T", "")
        url = f"https://irbank.net/{code}"
        try:
            resp = session.get(url, timeout=20, headers=headers)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            h2 = next((node for node in soup.find_all("h2") if "決算発表資料" in node.get_text(" ", strip=True)), None)
            if h2 is None:
                continue
            table = h2.find_next("table")
            if table is None:
                continue
            rows: list[dict[str, str]] = []
            for tr in table.find_all("tr"):
                cells = [c.get_text(" ", strip=True) for c in tr.find_all("td")]
                if len(cells) < 4:
                    continue
                date_text = cells[0].strip()
                title = cells[3].strip()
                if "決算短信" not in title:
                    continue
                dt = pd.to_datetime(date_text, errors="coerce")
                if pd.isna(dt):
                    continue
                rows.append(
                    {
                        "ticker": ticker,
                        "earnings_date": dt.strftime("%Y-%m-%d"),
                        "source": "irbank",
                    }
                )
            if rows:
                cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
                cache = cache.drop_duplicates(subset=["ticker", "earnings_date"]).sort_values(["ticker", "earnings_date"])
                cache.to_csv(cache_csv, index=False, encoding="utf-8-sig")
        except Exception:
            continue
        time.sleep(0.1)

    if cache.empty:
        return pd.DataFrame(columns=["ticker", "earnings_date", "source"])
    cache["earnings_date"] = pd.to_datetime(cache["earnings_date"], errors="coerce")
    cache = cache.dropna(subset=["earnings_date"]).sort_values(["ticker", "earnings_date"])
    return cache


def load_local_disclosure_dates(tickers: list[str]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    ticker_set = set(map(str, tickers))

    if PARSED_SUMMARY_CSV.exists():
        try:
            df = pd.read_csv(PARSED_SUMMARY_CSV, usecols=["ticker_code", "disclosure_date"])
            df = df.dropna(subset=["ticker_code", "disclosure_date"])
            for _, row in df.iterrows():
                code = str(row["ticker_code"]).strip()
                ticker = f"{code}.T" if code and code != "nan" else ""
                if ticker and ticker in ticker_set:
                    rows.append({"ticker": ticker, "earnings_date": str(row["disclosure_date"]).strip()})
        except Exception:
            pass

    if RAW_4Q2_DIR.exists():
        for path in RAW_4Q2_DIR.glob("*.txt"):
            code = path.stem.strip()
            ticker = f"{code}.T"
            if ticker not in ticker_set:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for line in text.splitlines():
                if "直近決算発表日" not in line:
                    continue
                dt = line.split("：", 1)[-1].replace("NEW!", "").strip()
                rows.append({"ticker": ticker, "earnings_date": dt})
                break

    if not rows:
        return pd.DataFrame(columns=["ticker", "earnings_date"])
    out = pd.DataFrame(rows).drop_duplicates(subset=["ticker", "earnings_date"])
    out["earnings_date"] = pd.to_datetime(out["earnings_date"], errors="coerce")
    out = out.dropna(subset=["earnings_date"]).sort_values(["ticker", "earnings_date"])
    return out


def build_earnings_map(cache: pd.DataFrame) -> dict[str, list[pd.Timestamp]]:
    if cache.empty:
        return {}
    out: dict[str, list[pd.Timestamp]] = {}
    for ticker, g in cache.groupby("ticker"):
        out[str(ticker)] = sorted(pd.to_datetime(g["earnings_date"]).dt.normalize().tolist())
    return out


def load_cached_earnings_only(tickers: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    ticker_set = set(map(str, tickers))

    local_df = load_local_disclosure_dates(tickers)
    if not local_df.empty:
        frames.append(local_df[["ticker", "earnings_date"]])

    if IRBANK_EARNINGS_CACHE_CSV.exists():
        try:
            irbank_df = pd.read_csv(IRBANK_EARNINGS_CACHE_CSV, usecols=["ticker", "earnings_date"])
            irbank_df = irbank_df[irbank_df["ticker"].astype(str).isin(ticker_set)]
            irbank_df["earnings_date"] = pd.to_datetime(irbank_df["earnings_date"], errors="coerce")
            irbank_df = irbank_df.dropna(subset=["earnings_date"])
            if not irbank_df.empty:
                frames.append(irbank_df)
        except Exception:
            pass

    if EARNINGS_CACHE_CSV.exists():
        try:
            yf_df = pd.read_csv(EARNINGS_CACHE_CSV, usecols=["ticker", "earnings_date"])
            yf_df = yf_df[yf_df["ticker"].astype(str).isin(ticker_set)]
            yf_df["earnings_date"] = pd.to_datetime(yf_df["earnings_date"], errors="coerce")
            yf_df = yf_df.dropna(subset=["earnings_date"])
            if not yf_df.empty:
                frames.append(yf_df)
        except Exception:
            pass

    if not frames:
        return pd.DataFrame(columns=["ticker", "earnings_date"])
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["ticker", "earnings_date"]).sort_values(["ticker", "earnings_date"])
    return merged


def prepare_metric_cache(price_map: dict[str, pd.DataFrame], dates: list[pd.Timestamp]) -> dict[str, dict[pd.Timestamp, dict]]:
    cache: dict[str, dict[pd.Timestamp, dict]] = {}
    for ticker, df in price_map.items():
        per_ticker: dict[pd.Timestamp, dict] = {}
        for date in dates:
            if date not in df.index:
                continue
            prev_idx = df.index[df.index < date]
            if len(prev_idx) == 0:
                continue
            signal_date = prev_idx[-1]
            metrics = compute_metrics(df, signal_date)
            if metrics:
                metrics["signal_basis_date"] = signal_date
            per_ticker[date] = metrics
        cache[ticker] = per_ticker
    return cache


def build_post_crash_broad_table(snapshot: pd.DataFrame) -> pd.DataFrame:
    if snapshot.empty:
        return snapshot
    x = snapshot.copy()
    x = x[x["trend_r2"].fillna(0) >= SELECTION_RULE["trend_r2_min"]]
    x = x[x["annual_return_pct"].fillna(-999) >= SELECTION_RULE["annual_return_min"]]
    x = x[x["quarter_return_pct"].fillna(-999) >= SELECTION_RULE["quarter_return_min"]]
    x = x[x["positive_month_ratio_pct"].fillna(0) >= SELECTION_RULE["positive_month_ratio_min"]]
    x = x[x["persistence_20d_pct"].fillna(0) >= SELECTION_RULE["persistence_20d_min"]]
    x = x[x["sector_adjusted_per_score"].fillna(0) >= SELECTION_RULE["sector_adjusted_per_score_min"]]
    x = x[x["ocr_per"].fillna(999) <= SELECTION_RULE["ocr_per_max"]]
    return x.sort_values(["sector_adjusted_per_score", "trend_r2", "annual_return_pct"], ascending=[False, False, False]).reset_index(drop=True)


def build_monthly_crash_tables(detail_df: pd.DataFrame, price_map: dict[str, pd.DataFrame], trade_dates: list[pd.Timestamp]) -> dict[pd.Timestamp, pd.DataFrame]:
    month_starts = sorted({pd.Timestamp(d.year, d.month, 1) for d in trade_dates})
    tables: dict[pd.Timestamp, pd.DataFrame] = {}
    for month_key in month_starts:
        prior_trade_dates = [d for d in trade_dates if d < month_key]
        basis_date = prior_trade_dates[-1] if prior_trade_dates else trade_dates[0] - pd.Timedelta(days=1)
        tables[month_key] = build_post_crash_broad_table(build_snapshot(detail_df, price_map, basis_date))
    return tables


def build_rebalance_schedule(trade_dates: list[pd.Timestamp], interval_days: int) -> tuple[list[pd.Timestamp], dict[pd.Timestamp, pd.Timestamp]]:
    if not trade_dates:
        return [], {}
    if interval_days <= 0:
        raise ValueError("interval_days must be positive")
    anchors = [trade_dates[i] for i in range(0, len(trade_dates), interval_days)]
    if anchors[-1] != trade_dates[-1]:
        anchors.append(trade_dates[-1])
    anchor_for_date: dict[pd.Timestamp, pd.Timestamp] = {}
    anchor_idx = 0
    current_anchor = anchors[anchor_idx]
    for date in trade_dates:
        while anchor_idx + 1 < len(anchors) and anchors[anchor_idx + 1] <= date:
            anchor_idx += 1
            current_anchor = anchors[anchor_idx]
        anchor_for_date[date] = current_anchor
    return anchors, anchor_for_date


def build_standard_table_for_dataset(spec_name: str, snapshot: pd.DataFrame) -> pd.DataFrame:
    if snapshot.empty:
        return snapshot
    if spec_name == "q2_2024":
        return select_q2_candidates(snapshot.copy()).drop_duplicates(subset=["ticker"], keep="first").reset_index(drop=True)
    if spec_name == "q3":
        return build_post_crash_broad_table(snapshot.copy()).drop_duplicates(subset=["ticker"], keep="first").reset_index(drop=True)
    if spec_name == "q4":
        return select_q4_candidates(snapshot.copy()).drop_duplicates(subset=["ticker"], keep="first").reset_index(drop=True)
    if spec_name == "4q2":
        scored = score_4q2_universe(snapshot.copy())
        return scored[scored["selected"] == True].copy().drop_duplicates(subset=["ticker"], keep="first").reset_index(drop=True)
    raise ValueError(f"Unsupported dataset for monthly standard rebuild: {spec_name}")


def build_interval_standard_tables(
    spec_name: str,
    detail_df: pd.DataFrame,
    price_map: dict[str, pd.DataFrame],
    trade_dates: list[pd.Timestamp],
    interval_days: int,
) -> dict[pd.Timestamp, pd.DataFrame]:
    anchors, _ = build_rebalance_schedule(trade_dates, interval_days)
    tables: dict[pd.Timestamp, pd.DataFrame] = {}
    for anchor_key in anchors:
        prior_trade_dates = [d for d in trade_dates if d < anchor_key]
        basis_date = prior_trade_dates[-1] if prior_trade_dates else trade_dates[0] - pd.Timedelta(days=1)
        snapshot = build_snapshot(detail_df, price_map, basis_date)
        tables[anchor_key] = build_standard_table_for_dataset(spec_name, snapshot)
    return tables


def build_interval_crash_tables(detail_df: pd.DataFrame, price_map: dict[str, pd.DataFrame], trade_dates: list[pd.Timestamp], interval_days: int) -> dict[pd.Timestamp, pd.DataFrame]:
    anchors, _ = build_rebalance_schedule(trade_dates, interval_days)
    tables: dict[pd.Timestamp, pd.DataFrame] = {}
    for anchor_key in anchors:
        prior_trade_dates = [d for d in trade_dates if d < anchor_key]
        basis_date = prior_trade_dates[-1] if prior_trade_dates else trade_dates[0] - pd.Timedelta(days=1)
        tables[anchor_key] = build_post_crash_broad_table(build_snapshot(detail_df, price_map, basis_date))
    return tables


def effective_phase_name(date: pd.Timestamp, phase_map: pd.Series, phase_shift_days: int) -> str:
    if phase_shift_days <= 0:
        phase = phase_map.loc[phase_map.index <= date].tail(1)
    else:
        phase = phase_map.loc[phase_map.index < date].tail(phase_shift_days).tail(1)
    if phase.empty:
        return "normal"
    return str(phase.iloc[0])


def lagged_phase_history(date: pd.Timestamp, phase_map: pd.Series, phase_shift_days: int) -> pd.Series:
    if phase_shift_days <= 0:
        return phase_map.loc[phase_map.index <= date]
    return phase_map.loc[phase_map.index < date]


def classify_selection_difficult_high_vol(
    date: pd.Timestamp,
    phase_map: pd.Series,
    phase_shift_days: int,
    lookback_days: int,
    rebound_phases: set[str],
) -> str | None:
    hist = lagged_phase_history(date, phase_map, phase_shift_days)
    if hist.empty:
        return None
    base = str(hist.iloc[-1])
    if base != "high_vol":
        return None

    recent = hist.tail(lookback_days)
    last_crash = recent[recent == "crash"]
    last_surge = recent[recent == "surge"]

    if not last_crash.empty:
        crash_date = last_crash.index[-1]
        after_crash = recent.loc[recent.index > crash_date]
        if any(str(x) in rebound_phases for x in after_crash.tolist()):
            return "rebound_confirmed_post_crash_high_vol"
        return "raw_post_crash_high_vol"

    if not last_surge.empty:
        surge_date = last_surge.index[-1]
        after_surge = recent.loc[recent.index > surge_date]
        if any(str(x) in rebound_phases for x in after_surge.tolist()):
            return "rebound_confirmed_post_surge_high_vol"
        return "post_surge_high_vol"

    return "generic_high_vol"


def classify_weak_uptrend(date: pd.Timestamp, phase_map: pd.Series, phase_shift_days: int, lookback_days: int = 5) -> bool:
    hist = lagged_phase_history(date, phase_map, phase_shift_days)
    if hist.empty:
        return False
    base = str(hist.iloc[-1])
    if base != "uptrend":
        return False

    streak = 0
    for value in reversed(hist.tolist()):
        if str(value) == "uptrend":
            streak += 1
        else:
            break
    if streak > 3:
        return False

    recent = hist.tail(max(lookback_days + streak, streak + 1))
    before_streak = recent.iloc[:-streak] if streak < len(recent) else recent.iloc[0:0]
    if before_streak.empty:
        return False
    return any(str(x) in {"high_vol", "crash", "surge", "settling"} for x in before_streak.tolist())


def weak_uptrend_flag(date: pd.Timestamp, phase_map: pd.Series, phase_shift_days: int, phase_proxy_mode: str) -> bool:
    return phase_proxy_mode in {"difficult_v8", "difficult_v9", "difficult_v10", "difficult_v11", "difficult_v12"} and classify_weak_uptrend(date, phase_map, phase_shift_days)


def projected_phase_name(
    date: pd.Timestamp,
    phase_map: pd.Series,
    phase_shift_days: int,
    phase_proxy_mode: str,
) -> str:
    base = effective_phase_name(date, phase_map, phase_shift_days)
    if phase_shift_days <= 0 or phase_proxy_mode == "lagged":
        return base
    if phase_proxy_mode == "crash_to_high_vol":
        if base == "crash":
            return "high_vol"
        return base
    if phase_proxy_mode == "crash_to_high_vol_surge_to_normal":
        if base == "crash":
            return "high_vol"
        if base == "surge":
            return "normal"
        return base
    if phase_proxy_mode in {"difficult_v1", "difficult_v2", "difficult_v3", "difficult_v4", "difficult_v5", "difficult_v6", "difficult_v7", "difficult_v8", "difficult_v9", "difficult_v10", "difficult_v11", "difficult_v12"}:
        if phase_proxy_mode == "difficult_v1":
            difficulty = classify_selection_difficult_high_vol(
                date,
                phase_map,
                phase_shift_days,
                lookback_days=5,
                rebound_phases={"reversal_up", "capitulation_end", "normal", "uptrend"},
            )
        elif phase_proxy_mode == "difficult_v2":
            difficulty = classify_selection_difficult_high_vol(
                date,
                phase_map,
                phase_shift_days,
                lookback_days=10,
                rebound_phases={"reversal_up", "capitulation_end", "normal", "uptrend"},
            )
        elif phase_proxy_mode == "difficult_v3":
            difficulty = classify_selection_difficult_high_vol(
                date,
                phase_map,
                phase_shift_days,
                lookback_days=10,
                rebound_phases={"reversal_up", "capitulation_end", "normal", "uptrend", "settling"},
            )
        elif phase_proxy_mode == "difficult_v4":
            difficulty = classify_selection_difficult_high_vol(
                date,
                phase_map,
                phase_shift_days,
                lookback_days=5,
                rebound_phases={"reversal_up", "capitulation_end", "normal", "uptrend"},
            )
        elif phase_proxy_mode == "difficult_v5":
            difficulty = classify_selection_difficult_high_vol(
                date,
                phase_map,
                phase_shift_days,
                lookback_days=10,
                rebound_phases={"reversal_up", "capitulation_end", "normal", "uptrend"},
            )
        elif phase_proxy_mode == "difficult_v6":
            difficulty = classify_selection_difficult_high_vol(
                date,
                phase_map,
                phase_shift_days,
                lookback_days=5,
                rebound_phases={"reversal_up", "capitulation_end", "normal", "uptrend"},
            )
        else:
            difficulty = classify_selection_difficult_high_vol(
                date,
                phase_map,
                phase_shift_days,
                lookback_days=10,
                rebound_phases={"reversal_up", "capitulation_end", "normal", "uptrend", "settling"},
            )
        if difficulty == "raw_post_crash_high_vol":
            return "downtrend"
        if phase_proxy_mode == "difficult_v6" and difficulty == "rebound_confirmed_post_crash_high_vol":
            return "normal"
        if phase_proxy_mode in {"difficult_v7", "difficult_v11", "difficult_v12"} and difficulty in {"rebound_confirmed_post_crash_high_vol", "generic_high_vol"}:
            return "normal"
        if phase_proxy_mode in {"difficult_v1", "difficult_v2", "difficult_v3"} and difficulty == "post_surge_high_vol":
            return "surge"
        if phase_proxy_mode in {"difficult_v8", "difficult_v9", "difficult_v10", "difficult_v11", "difficult_v12"} and classify_weak_uptrend(date, phase_map, phase_shift_days):
            if phase_proxy_mode in {"difficult_v8", "difficult_v12"}:
                return "downtrend"
            return "settling"
        return base
    return base


def post_crash_broad_signal(metrics: dict, static_row: dict) -> tuple[bool, float]:
    if not metrics:
        return False, 0.0
    signal = (
        metrics["trend_r2"] >= SELECTION_RULE["trend_r2_min"]
        and metrics["annual_return_pct"] >= SELECTION_RULE["annual_return_min"]
        and metrics["quarter_return_pct"] >= SELECTION_RULE["quarter_return_min"]
        and metrics["positive_month_ratio_pct"] >= SELECTION_RULE["positive_month_ratio_min"]
        and metrics["persistence_20d_pct"] >= SELECTION_RULE["persistence_20d_min"]
        and float(static_row.get("sector_adjusted_per_score", 0) if pd.notna(static_row.get("sector_adjusted_per_score")) else 0) >= SELECTION_RULE["sector_adjusted_per_score_min"]
        and (float(static_row.get("forecast_per")) <= SELECTION_RULE["ocr_per_max"] if pd.notna(static_row.get("forecast_per")) else False)
    )
    score = (
        0.35 * min(max(static_row.get("sector_adjusted_per_score", 0) or 0, 0.0), 1.0)
        + 0.20 * min(max(metrics["trend_r2"], 0.0), 1.0)
        + 0.15 * min(max(metrics["annual_return_pct"] / 100.0, 0.0), 1.0)
        + 0.10 * min(max(metrics["quarter_return_pct"] / 40.0, 0.0), 1.0)
        + 0.10 * min(max(metrics["persistence_20d_pct"] / 100.0, 0.0), 1.0)
        + 0.10 * min(max(metrics["positive_month_ratio_pct"] / 100.0, 0.0), 1.0)
    )
    return signal, score


def trading_days_to_next_earnings(current_date: pd.Timestamp, df: pd.DataFrame, earnings_dates: list[pd.Timestamp]) -> int | None:
    future_dates = [d for d in earnings_dates if d > current_date.normalize()]
    if not future_dates:
        return None
    next_dt = future_dates[0]
    trade_days = [d for d in df.index if current_date.normalize() < d.normalize() <= next_dt]
    return len(trade_days) if trade_days else None


def trading_days_since_prev_earnings(current_date: pd.Timestamp, df: pd.DataFrame, earnings_dates: list[pd.Timestamp]) -> int | None:
    past_dates = [d for d in earnings_dates if d < current_date.normalize()]
    if not past_dates:
        return None
    prev_dt = past_dates[-1]
    trade_days = [d for d in df.index if prev_dt < d.normalize() <= current_date.normalize()]
    return len(trade_days) if trade_days else None


def run_dataset(
    spec: TableSpec,
    phase_map: pd.Series,
    earnings_map: dict[str, list[pd.Timestamp]],
    earnings_pre_days: int,
    earnings_post_days: int,
    phase_shift_days: int,
    phase_proxy_mode: str,
    rebuild_interval_trading_days: int,
    uptrend_rebuild_interval_trading_days: int | None = None,
    high_vol_take_profit_pct: float | None = None,
    weak_uptrend_take_profit_pct: float | None = None,
    high_vol_min_score: float | None = None,
    high_vol_max_new_buys_per_day: int | None = None,
) -> dict:
    detail_df = normalize_static(pd.read_csv(spec.detail_csv)).drop_duplicates(subset=["ticker"], keep="first")

    start_date = pd.Timestamp(spec.start_date)
    end_date = pd.Timestamp(spec.end_date)
    price_map_all = load_price_map(detail_df["ticker"].astype(str).tolist(), end_date)
    trade_dates = sorted(set().union(*[set(df[(df.index >= start_date) & (df.index <= end_date)].index.tolist()) for df in price_map_all.values()]))
    if not trade_dates:
        return {
            "dataset": spec.name,
            "final_capital": float(spec.initial_capital),
            "total_return_pct": 0.0,
            "num_buys": 0,
            "max_drawdown_pct": 0.0,
            "earnings_pre_days": earnings_pre_days,
            "earnings_post_days": earnings_post_days,
            "phase_shift_days": phase_shift_days,
            "phase_proxy_mode": phase_proxy_mode,
            "rebuild_interval_trading_days": rebuild_interval_trading_days,
            "uptrend_rebuild_interval_trading_days": uptrend_rebuild_interval_trading_days,
            "high_vol_min_score": high_vol_min_score,
            "high_vol_max_new_buys_per_day": high_vol_max_new_buys_per_day,
            "phase_pnl": {},
        }

    standard_tables = build_interval_standard_tables(spec.name, detail_df, price_map_all, trade_dates, rebuild_interval_trading_days)
    crash_tables = build_interval_crash_tables(detail_df, price_map_all, trade_dates, rebuild_interval_trading_days)
    _, anchor_for_date = build_rebalance_schedule(trade_dates, rebuild_interval_trading_days)
    if uptrend_rebuild_interval_trading_days and uptrend_rebuild_interval_trading_days != rebuild_interval_trading_days:
        uptrend_standard_tables = build_interval_standard_tables(
            spec.name,
            detail_df,
            price_map_all,
            trade_dates,
            uptrend_rebuild_interval_trading_days,
        )
        _, uptrend_anchor_for_date = build_rebalance_schedule(trade_dates, uptrend_rebuild_interval_trading_days)
    else:
        uptrend_standard_tables = standard_tables
        uptrend_anchor_for_date = anchor_for_date
    union_tickers: set[str] = set()
    for tbl in standard_tables.values():
        if not tbl.empty:
            union_tickers.update(tbl["ticker"].astype(str).tolist())
    for tbl in uptrend_standard_tables.values():
        if not tbl.empty:
            union_tickers.update(tbl["ticker"].astype(str).tolist())
    for tbl in crash_tables.values():
        if not tbl.empty:
            union_tickers.update(tbl["ticker"].astype(str).tolist())
    price_map = {ticker: df for ticker, df in price_map_all.items() if ticker in union_tickers}
    dates = sorted(set().union(*[set(df[(df.index >= start_date) & (df.index <= end_date)].index.tolist()) for df in price_map.values()]))
    metrics_cache = prepare_metric_cache(price_map, dates)
    mapping = practical_v2_mapping()

    cash = float(spec.initial_capital)
    positions: dict[str, dict] = {}
    prev_signal = {ticker: False for ticker in price_map}
    equity_rows = []
    trade_rows: list[dict] = []
    buy_count = 0

    for date in dates:
        phase_name = projected_phase_name(date, phase_map, phase_shift_days, phase_proxy_mode)
        is_weak_uptrend = weak_uptrend_flag(date, phase_map, phase_shift_days, phase_proxy_mode)
        anchor_key = anchor_for_date[date]
        uptrend_anchor_key = uptrend_anchor_for_date[date]
        standard_lookup = standard_tables[anchor_key].set_index("ticker").to_dict("index") if anchor_key in standard_tables and not standard_tables[anchor_key].empty else {}
        uptrend_standard_lookup = (
            uptrend_standard_tables[uptrend_anchor_key].set_index("ticker").to_dict("index")
            if uptrend_anchor_key in uptrend_standard_tables and not uptrend_standard_tables[uptrend_anchor_key].empty
            else standard_lookup
        )
        active_lookup = crash_tables[anchor_key].set_index("ticker").to_dict("index") if phase_name == "crash" and not crash_tables[anchor_key].empty else standard_lookup
        if phase_name == "uptrend":
            active_lookup = uptrend_standard_lookup
        rule_name = "q2_defensive" if phase_name == "crash" else mapping.get(phase_name, "condition2")

        signal_today: dict[str, bool] = {}
        score_today: dict[str, float] = {}
        basis_date_today: dict[str, pd.Timestamp] = {}
        earnings_block_today: dict[str, bool] = {}
        earnings_post_block_today: dict[str, bool] = {}

        for ticker in price_map:
            df = price_map[ticker]
            dte = trading_days_to_next_earnings(date, df, earnings_map.get(ticker, []))
            earnings_block_today[ticker] = dte is not None and 1 <= dte <= earnings_pre_days
            dse = trading_days_since_prev_earnings(date, df, earnings_map.get(ticker, []))
            earnings_post_block_today[ticker] = dse is not None and 1 <= dse <= earnings_post_days
            if ticker not in active_lookup:
                signal_today[ticker] = False
                continue
            metrics = metrics_cache.get(ticker, {}).get(date, {})
            if not metrics:
                signal_today[ticker] = prev_signal.get(ticker, False) if date not in df.index else False
                continue
            if rule_name == "q2_defensive":
                sig, signal_score = post_crash_broad_signal(metrics, active_lookup[ticker])
            else:
                sig, signal_score = eval_signal(rule_name, metrics, active_lookup[ticker])
            if phase_name == "crash":
                earnings_block_today[ticker] = False
                earnings_post_block_today[ticker] = False
            if earnings_block_today[ticker] or earnings_post_block_today[ticker]:
                sig = False
            signal_today[ticker] = sig
            score_today[ticker] = signal_score
            basis_date_today[ticker] = metrics["signal_basis_date"]

        for ticker in list(positions.keys()):
            df = price_map[ticker]
            if date not in df.index:
                continue
            day = df.loc[date]
            close_price = float(day["Close"])
            low_price = float(day["Low"]) if "Low" in day.index and not pd.isna(day["Low"]) else close_price
            open_price = float(day["Open"]) if "Open" in day.index and not pd.isna(day["Open"]) else close_price
            entry_price = positions[ticker]["entry_price"]
            rule_for_pos = positions[ticker]["rule_name"]
            entry_phase = positions[ticker].get("entry_phase", "")
            entry_weak_uptrend = bool(positions[ticker].get("entry_weak_uptrend", False))
            if rule_for_pos == "q2_defensive":
                tp = TRADING_RULE["take_profit_pct"] / 100.0
            elif weak_uptrend_take_profit_pct is not None and entry_weak_uptrend:
                tp = weak_uptrend_take_profit_pct / 100.0
            elif high_vol_take_profit_pct is not None and entry_phase == "high_vol":
                tp = high_vol_take_profit_pct / 100.0
            else:
                tp = 0.08
            sl = TRADING_RULE["stop_loss_pct"] / 100.0 if rule_for_pos == "q2_defensive" else 0.05
            ret = close_price / entry_price - 1.0

            exit_reason = None
            exit_price = None
            if earnings_block_today.get(ticker, False):
                exit_reason = "pre_earnings_exit"
                exit_price = open_price
            elif phase_name == "crash":
                prev_idx = df.index[df.index < date]
                if len(prev_idx):
                    prev_date = prev_idx[-1]
                    prev_close = float(df.loc[prev_date, "Close"])
                    early_trigger = prev_close * 0.98
                    if low_price <= early_trigger:
                        exit_reason = "crash_early_exit"
                        exit_price = min(open_price, early_trigger)
            if exit_reason is None and ret >= tp:
                exit_reason = "take_profit"
                exit_price = close_price
            elif exit_reason is None and ret <= -sl:
                exit_reason = "stop_loss"
                exit_price = close_price
            elif exit_reason is None and not signal_today.get(ticker, False):
                exit_reason = "sell_signal"
                exit_price = close_price
            if exit_reason is not None:
                shares = positions[ticker]["shares"]
                cash += shares * float(exit_price)
                trade_rows.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "action": "SELL",
                    "price": float(exit_price),
                    "shares": int(shares),
                    "reason": exit_reason,
                    "nikkei_phase": phase_name,
                })
                del positions[ticker]

        candidates = []
        for ticker in price_map:
            if ticker in positions:
                continue
            if signal_today.get(ticker, False) and not prev_signal.get(ticker, False):
                candidates.append((ticker, score_today.get(ticker, 0.0)))
        candidates.sort(key=lambda x: x[1], reverse=True)
        if phase_name == "high_vol":
            if high_vol_min_score is not None:
                candidates = [(ticker, score) for ticker, score in candidates if score >= high_vol_min_score]
            if high_vol_max_new_buys_per_day is not None and high_vol_max_new_buys_per_day >= 0:
                candidates = candidates[:high_vol_max_new_buys_per_day]

        remaining = len(candidates)
        for ticker, _score in candidates:
            if earnings_block_today.get(ticker, False) or earnings_post_block_today.get(ticker, False):
                remaining -= 1
                continue
            df = price_map[ticker]
            day = df.loc[date]
            prev_close = float(df.loc[basis_date_today[ticker], "Close"])
            entry_limit = TRADING_RULE["entry_limit_pct"] if rule_name == "q2_defensive" else 1.5
            trigger_price = prev_close * (1.0 + entry_limit / 100.0)
            day_open = float(day["Open"]) if "Open" in day.index and not pd.isna(day["Open"]) else float(day["Close"])
            day_high = float(day["High"]) if "High" in day.index and not pd.isna(day["High"]) else float(day["Close"])
            if day_high < trigger_price:
                remaining -= 1
                continue
            fill_price = max(day_open, trigger_price)
            alloc = cash / remaining if remaining > 0 else 0.0
            lot_cost = fill_price * 100.0
            shares = int(alloc // lot_cost) * 100
            if shares >= 100 and shares * fill_price <= cash:
                cash -= shares * fill_price
                positions[ticker] = {
                    "shares": shares,
                    "entry_price": fill_price,
                    "rule_name": rule_name,
                    "entry_phase": phase_name,
                    "entry_weak_uptrend": is_weak_uptrend,
                }
                trade_rows.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "action": "BUY",
                    "price": float(fill_price),
                    "shares": int(shares),
                    "reason": f"{rule_name}_entry",
                    "nikkei_phase": phase_name,
                })
                buy_count += 1
            remaining -= 1

        market_value = 0.0
        for ticker, pos in positions.items():
            df = price_map[ticker]
            usable_idx = df.index[df.index <= date]
            if len(usable_idx):
                market_value += pos["shares"] * float(df.loc[usable_idx[-1], "Close"])
        equity_rows.append({"date": date, "nikkei_phase": phase_name, "equity": cash + market_value})
        prev_signal = signal_today

    latest_date = max(dates)
    for ticker in list(positions.keys()):
        df = price_map[ticker]
        usable_idx = df.index[df.index <= latest_date]
        if len(usable_idx):
            px = float(df.loc[usable_idx[-1], "Close"])
            cash += positions[ticker]["shares"] * px
            trade_rows.append({
                "date": latest_date.strftime("%Y-%m-%d"),
                "ticker": ticker,
                "action": "SELL",
                "price": px,
                "shares": int(positions[ticker]["shares"]),
                "reason": "end_of_backtest",
                "nikkei_phase": projected_phase_name(latest_date, phase_map, phase_shift_days, phase_proxy_mode),
            })
        del positions[ticker]

    equity_df = pd.DataFrame(equity_rows)
    if not equity_df.empty:
        equity_df["pnl_day"] = equity_df["equity"].diff().fillna(equity_df["equity"] - spec.initial_capital)
        phase_pnl = equity_df.groupby("nikkei_phase")["pnl_day"].sum().to_dict()
        drawdown = equity_df["equity"].astype(float) / equity_df["equity"].astype(float).cummax() - 1.0
        max_drawdown_pct = float(drawdown.min()) * 100.0
    else:
        phase_pnl = {}
        max_drawdown_pct = 0.0

    return {
        "dataset": spec.name,
        "final_capital": float(cash),
        "total_return_pct": (float(cash) / float(spec.initial_capital) - 1.0) * 100.0,
        "num_buys": int(buy_count),
        "max_drawdown_pct": max_drawdown_pct,
        "earnings_pre_days": earnings_pre_days,
        "earnings_post_days": earnings_post_days,
        "phase_shift_days": phase_shift_days,
        "phase_proxy_mode": phase_proxy_mode,
        "rebuild_interval_trading_days": rebuild_interval_trading_days,
        "uptrend_rebuild_interval_trading_days": uptrend_rebuild_interval_trading_days,
        "high_vol_take_profit_pct": high_vol_take_profit_pct,
        "weak_uptrend_take_profit_pct": weak_uptrend_take_profit_pct,
        "high_vol_min_score": high_vol_min_score,
        "high_vol_max_new_buys_per_day": high_vol_max_new_buys_per_day,
        "phase_pnl": phase_pnl,
        "trade_log": pd.DataFrame(trade_rows),
        "equity_curve": equity_df,
    }


def batch_specs() -> list[TableSpec]:
    return [
        TableSpec(
            "q2_2024",
            ROOT / "projects" / "quarterly_ranker" / "output" / "q2_2024_pre_analysis_20240630_aligned" / "q2_2024_pre_shikiho_feature_ranking.csv",
            ROOT / "projects" / "quarterly_ranker" / "output" / "q2_2024_pre_analysis_20240630_aligned" / "operational" / "q2_2024_pre_selected_candidates_operational.csv",
            "2024-07-01",
            "2024-09-30",
        ),
        TableSpec(
            "q3",
            ROOT / "projects" / "quarterly_ranker" / "output" / "q3_pre_analysis_20250630_aligned" / "q3_pre_shikiho_feature_ranking.csv",
            ROOT / "projects" / "quarterly_ranker" / "output" / "q3_pre_analysis_20250630_aligned" / "threshold_search_post_high_vol" / "operational" / "best_selected_candidates_operational.csv",
            "2025-07-01",
            "2025-09-30",
        ),
        TableSpec(
            "q4",
            ROOT / "projects" / "quarterly_ranker" / "output" / "q4_pre_analysis_20250930_full" / "q4_pre_shikiho_feature_ranking.csv",
            ROOT / "projects" / "quarterly_ranker" / "output" / "q4_pre_analysis_20250930_full" / "operational" / "q4_pre_selected_candidates_operational.csv",
            "2025-10-01",
            "2025-12-31",
        ),
        TableSpec(
            "4q2",
            ROOT / "projects" / "shikiho_text_parser" / "output" / "4q2_selection" / "4q2_scored_universe.csv",
            ROOT / "projects" / "shikiho_text_parser" / "output" / "4q2_selection" / "operational" / "4q2_selected_candidates_operational.csv",
            "2026-01-01",
            "2026-03-10",
        ),
    ]


def collect_relevant_tickers(spec: TableSpec) -> list[str]:
    detail_df = normalize_static(pd.read_csv(spec.detail_csv)).drop_duplicates(subset=["ticker"], keep="first")
    start_date = pd.Timestamp(spec.start_date)
    end_date = pd.Timestamp(spec.end_date)
    price_map_all = load_price_map(detail_df["ticker"].astype(str).tolist(), end_date)
    trade_dates = sorted(set().union(*[set(df[(df.index >= start_date) & (df.index <= end_date)].index.tolist()) for df in price_map_all.values()]))
    if not trade_dates:
        return []
    standard_tables = build_interval_standard_tables(spec.name, detail_df, price_map_all, trade_dates, 20)
    crash_tables = build_interval_crash_tables(detail_df, price_map_all, trade_dates, 20)
    union_tickers: set[str] = set()
    for tbl in standard_tables.values():
        if not tbl.empty:
            union_tickers.update(tbl["ticker"].astype(str).tolist())
    for tbl in crash_tables.values():
        if not tbl.empty:
            union_tickers.update(tbl["ticker"].astype(str).tolist())
    return sorted(union_tickers)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run practical phase-adaptive v3.1 with pre-earnings exit and post-earnings cooldown, except on crash table.")
    parser.add_argument("--detail-csv", type=Path)
    parser.add_argument("--selected-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-date", type=str)
    parser.add_argument("--end-date", type=str)
    parser.add_argument("--dataset-name", type=str)
    parser.add_argument("--initial-capital", type=float, default=3_000_000.0)
    parser.add_argument("--earnings-pre-days", type=int, default=1)
    parser.add_argument("--earnings-post-days", type=int, default=5)
    parser.add_argument("--phase-shift-days", type=int, default=0)
    parser.add_argument("--rebuild-interval-trading-days", type=int, default=20)
    parser.add_argument("--uptrend-rebuild-interval-trading-days", type=int, default=None)
    parser.add_argument("--high-vol-take-profit-pct", type=float, default=None)
    parser.add_argument("--weak-uptrend-take-profit-pct", type=float, default=None)
    parser.add_argument("--high-vol-min-score", type=float, default=None)
    parser.add_argument("--high-vol-max-new-buys-per-day", type=int, default=None)
    parser.add_argument(
        "--phase-proxy-mode",
        type=str,
        default="lagged",
        choices=[
            "lagged",
            "crash_to_high_vol",
            "crash_to_high_vol_surge_to_normal",
            "difficult_v1",
            "difficult_v2",
            "difficult_v3",
            "difficult_v4",
            "difficult_v5",
            "difficult_v6",
            "difficult_v7",
            "difficult_v8",
            "difficult_v9",
            "difficult_v10",
            "difficult_v11",
            "difficult_v12",
        ],
    )
    parser.add_argument("--batch", action="store_true")
    args = parser.parse_args()

    phase_map = load_phase_map(PHASE_CSV)
    specs = batch_specs() if args.batch else [
        TableSpec(
            args.dataset_name,
            args.detail_csv,
            args.selected_csv,
            args.start_date,
            args.end_date,
            args.initial_capital,
        )
    ]

    all_tickers: list[str] = []
    for spec in specs:
        all_tickers.extend(collect_relevant_tickers(spec))
    earnings_map = build_earnings_map(load_cached_earnings_only(all_tickers))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in specs:
        result = run_dataset(
            spec,
            phase_map,
            earnings_map,
            args.earnings_pre_days,
            args.earnings_post_days,
            args.phase_shift_days,
            args.phase_proxy_mode,
            args.rebuild_interval_trading_days,
            args.uptrend_rebuild_interval_trading_days,
            args.high_vol_take_profit_pct,
            args.weak_uptrend_take_profit_pct,
            args.high_vol_min_score,
            args.high_vol_max_new_buys_per_day,
        )
        out_dir = args.output_dir / spec.name if args.batch else args.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "mapping_name": f"practical_phase_adaptive_v31_pre{args.earnings_pre_days}_post{args.earnings_post_days}_except_crash",
            "mapping_ja": f"実務向け局面切替v3.1_決算前{args.earnings_pre_days}営業日_決算後{args.earnings_post_days}営業日見送り_crash除外",
            "earnings_pre_days": args.earnings_pre_days,
            "earnings_post_days": args.earnings_post_days,
            "phase_shift_days": args.phase_shift_days,
            "phase_proxy_mode": args.phase_proxy_mode,
            "rebuild_interval_trading_days": args.rebuild_interval_trading_days,
            "uptrend_rebuild_interval_trading_days": args.uptrend_rebuild_interval_trading_days,
            "high_vol_take_profit_pct": args.high_vol_take_profit_pct,
            "weak_uptrend_take_profit_pct": args.weak_uptrend_take_profit_pct,
            "high_vol_min_score": args.high_vol_min_score,
            "high_vol_max_new_buys_per_day": args.high_vol_max_new_buys_per_day,
            **{k: v for k, v in result.items() if k not in {"trade_log", "equity_curve"}},
        }
        (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        result["trade_log"].to_csv(out_dir / "trade_log.csv", index=False, encoding="utf-8-sig")
        result["equity_curve"].to_csv(out_dir / "equity_curve.csv", index=False, encoding="utf-8-sig")
        rows.append(payload)

    if args.batch:
        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(args.output_dir / "summary_all.csv", index=False, encoding="utf-8-sig")
        print(summary_df.to_string(index=False))
    else:
        print(json.dumps(rows[0], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

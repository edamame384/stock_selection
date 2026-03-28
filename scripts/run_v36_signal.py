from __future__ import annotations

import argparse
import io
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.shikiho_text_parser import backtest_phase_adaptive_practical_v35_alloc50_candidate as base_mod
from projects.shikiho_text_parser.compare_alloc50_post_major_multi_etf_expanded_candidate import (
    ETF_SL,
    ETF_TICKERS,
    ETF_TP,
    POST_MAJOR_PHASES,
    STOCK_SL,
    STOCK_TP,
    STOCK_TOP_N,
    build_post_major_state,
    classify_sector_state,
    pick_best_etf,
    prepare_earnings_map,
    prev_high_break_signal,
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
    "3Q": q3_spec,
    "4Q": q4_spec,
    "4Q-2": q4_2_spec,
}

DEFAULT_DATASET = "4Q-2"


def next_business_day(ts: pd.Timestamp) -> pd.Timestamp:
    d = ts + pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d += pd.Timedelta(days=1)
    return d.normalize()


def refresh_prices(tickers: list[str], price_dir: Path) -> None:
    for ticker in tickers:
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                df = download_daily_data(ticker, period="max")
            save_price_data(ticker, df, price_dir)
        except Exception:
            continue


def latest_signal_date(price_map: dict[str, pd.DataFrame]) -> pd.Timestamp:
    latest = [df.index.max() for df in price_map.values() if not df.empty]
    if not latest:
        raise ValueError("No price data found for v3.6 signal generation.")
    return max(latest)


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
        f"[PICK][v36] {symbol} tp_prob={tp_prob:.2f}% "
        f"sector={sector} method={method} company={company}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate v3.6 live signal log for Discord notification.")
    parser.add_argument("--dataset", choices=list(DATASET_SPECS.keys()), default=DEFAULT_DATASET)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--skip-refresh", action="store_true")
    args = parser.parse_args()

    spec = DATASET_SPECS[args.dataset]()
    detail_df = base_mod.normalize_static(pd.read_csv(spec.detail_csv)).drop_duplicates(subset=["ticker"], keep="first")
    tickers = list(dict.fromkeys(detail_df["ticker"].astype(str).tolist() + ETF_TICKERS))

    if not args.skip_refresh:
        refresh_prices(tickers, PRICE_DIR)
    price_map_all = base_mod.load_price_map(tickers, pd.Timestamp.today())
    signal_date = latest_signal_date(price_map_all)
    trade_date = next_business_day(signal_date)

    phase_df = build_post_major_state(pd.read_csv(base_mod.PHASE_CSV, skiprows=[1]).rename(columns={"phase": "phase_name"}))
    phase_df["Date"] = pd.to_datetime(phase_df["Date"])
    for col in ["ret5", "dd20", "vol10"]:
        phase_df[col] = pd.to_numeric(phase_df[col], errors="coerce")
    phase_state = phase_df.set_index("Date").sort_index()
    phase_map = base_mod.load_phase_map(base_mod.PHASE_CSV)
    earnings_map = prepare_earnings_map(spec)

    stock_price_map_all = {k: v for k, v in price_map_all.items() if k not in ETF_TICKERS}
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
    rule_name = "q2_defensive" if phase_name == "crash" else base_mod.practical_v2_mapping().get(phase_name, "condition2")

    standard_table = standard_tables.get(anchor_key, pd.DataFrame())
    standard_lookup = standard_table.set_index("ticker").to_dict("index") if not standard_table.empty else {}
    active_lookup = crash_tables[anchor_key].set_index("ticker").to_dict("index") if phase_name == "crash" and anchor_key in crash_tables and not crash_tables[anchor_key].empty else standard_lookup

    prior_phase_row = phase_state.loc[phase_state.index < trade_date].tail(1)
    post_major = False if prior_phase_row.empty else bool(prior_phase_row.iloc[0]["post_major_crash_mode"])
    post_major_phase = "" if prior_phase_row.empty else str(prior_phase_row.iloc[0]["phase_name"])
    post_major_active = post_major and post_major_phase in POST_MAJOR_PHASES
    sector_state = classify_sector_state(standard_table, price_map_all, trade_date)

    print(
        f"[META] regime={phase_name} method={rule_name} "
        f"signal_date={signal_date.date().isoformat()} trade_date={trade_date.date().isoformat()}"
    )

    lines: list[str] = []
    pick_lines: list[str] = []

    if post_major_active and sector_state["mode"] == "concentrated":
        chosen_etf, trigger_price, etf_scores = pick_best_etf(price_map_all, trade_date)
        if chosen_etf and trigger_price is not None:
            df = price_map_all[chosen_etf]
            basis_idx = df.index[df.index < trade_date]
            if len(basis_idx):
                close_price = float(df.loc[basis_idx[-1], "Close"])
                score = max(float(etf_scores.get(chosen_etf, 0.0)) * 100.0, 0.0)
                buy_line, pick_line = format_buy(
                    symbol=chosen_etf,
                    method="post_major_multi_etf_entry",
                    sector="ETF",
                    company=chosen_etf,
                    close_price=close_price,
                    entry_price=float(trigger_price),
                    tp_prob=score,
                    tp_ratio=ETF_TP,
                    sl_ratio=ETF_SL,
                )
                lines.append(buy_line)
                pick_lines.append(pick_line)
    elif post_major_active and sector_state["mode"] == "dispersed":
        candidates = []
        for ticker in [t for t in price_map_all if t not in ETF_TICKERS]:
            if ticker not in active_lookup:
                continue
            df = price_map_all[ticker]
            basis_idx = df.index[df.index < trade_date]
            if not len(basis_idx):
                continue
            sig, trigger = prev_high_break_signal(df, basis_idx[-1], trade_date)
            if not sig or trigger is None:
                continue
            metrics = base_mod.compute_metrics(df, basis_idx[-1])
            if not metrics:
                continue
            if rule_name == "q2_defensive":
                sig_rule, signal_score = base_mod.post_crash_broad_signal(metrics, active_lookup[ticker])
            else:
                sig_rule, signal_score = base_mod.eval_signal(rule_name, metrics, active_lookup[ticker])
            if not sig_rule:
                continue
            if base_mod.trading_days_to_next_earnings(trade_date, df, earnings_map.get(ticker, [])) not in (None, 0):
                continue
            candidates.append((ticker, float(signal_score), float(trigger)))
        candidates.sort(key=lambda x: x[1], reverse=True)
        for ticker, signal_score, trigger in candidates[:STOCK_TOP_N]:
            df = price_map_all[ticker]
            basis_idx = df.index[df.index < trade_date]
            close_price = float(df.loc[basis_idx[-1], "Close"])
            company = str(active_lookup[ticker].get("company_name") or active_lookup[ticker].get("company") or ticker.removesuffix(".T"))
            sector = str(active_lookup[ticker].get("simple_sector") or active_lookup[ticker].get("sector") or "UNKNOWN")
            buy_line, pick_line = format_buy(
                symbol=ticker,
                method="post_major_prev_high_break_entry",
                sector=sector,
                company=company,
                close_price=close_price,
                entry_price=trigger,
                tp_prob=max(signal_score * 100.0, 0.0),
                tp_ratio=STOCK_TP,
                sl_ratio=STOCK_SL,
            )
            lines.append(buy_line)
            pick_lines.append(pick_line)
    else:
        metrics_cache = base_mod.prepare_metric_cache(stock_price_map_all, [trade_date])
        candidates = []
        for ticker in [t for t in price_map_all if t not in ETF_TICKERS]:
            if ticker not in active_lookup:
                continue
            df = price_map_all[ticker]
            metrics = metrics_cache.get(ticker, {}).get(trade_date, {})
            if not metrics:
                continue
            if rule_name == "q2_defensive":
                sig_rule, signal_score = base_mod.post_crash_broad_signal(metrics, active_lookup[ticker])
            else:
                sig_rule, signal_score = base_mod.eval_signal(rule_name, metrics, active_lookup[ticker])
            if not sig_rule:
                continue
            if base_mod.trading_days_to_next_earnings(trade_date, df, earnings_map.get(ticker, [])) not in (None, 0):
                continue
            prev_close = float(df.loc[metrics["signal_basis_date"], "Close"])
            entry_limit = base_mod.TRADING_RULE["entry_limit_pct"] if rule_name == "q2_defensive" else 1.5
            trigger = prev_close * (1.0 + entry_limit / 100.0)
            day = df.loc[trade_date]
            day_high = float(day["High"]) if "High" in day.index and not pd.isna(day["High"]) else float(day["Close"])
            if day_high < trigger:
                continue
            candidates.append((ticker, float(signal_score), float(trigger)))
        candidates.sort(key=lambda x: x[1], reverse=True)
        for ticker, signal_score, trigger in candidates[:args.top_n]:
            df = price_map_all[ticker]
            metrics = metrics_cache[ticker][trade_date]
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
                tp_ratio=0.05 if rule_name == "q2_defensive" else 0.08,
                sl_ratio=0.05 if rule_name == "q2_defensive" else 0.05,
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

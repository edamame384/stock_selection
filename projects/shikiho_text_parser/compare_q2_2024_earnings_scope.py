from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.shikiho_text_parser.backtest_phase_adaptive import compute_metrics, eval_signal
from projects.shikiho_text_parser.backtest_phase_adaptive_practical_v2 import practical_v2_mapping
from projects.shikiho_text_parser.backtest_phase_adaptive_practical_v31_pre_earnings5 import (
    batch_specs,
    build_earnings_map,
    build_monthly_crash_tables,
    fetch_earnings_cache,
    fetch_irbank_earnings_cache,
    load_local_disclosure_dates,
    load_phase_map,
    load_price_map,
    normalize_static,
    PHASE_CSV,
    post_crash_broad_signal,
    prepare_metric_cache,
    trading_days_to_next_earnings,
    TRADING_RULE,
)


def trading_days_since_prev_earnings(current_date: pd.Timestamp, df: pd.DataFrame, earnings_dates: list[pd.Timestamp]) -> int | None:
    past_dates = [d for d in earnings_dates if d < current_date.normalize()]
    if not past_dates:
        return None
    prev_dt = past_dates[-1]
    trade_days = [d for d in df.index if prev_dt < d.normalize() <= current_date.normalize()]
    return len(trade_days) if trade_days else None

def run_variant(spec, phase_map, earnings_map, ignore_pre_phases: set[str], ignore_post_phases: set[str], post_days: int) -> dict:
    detail_df = normalize_static(pd.read_csv(spec.detail_csv)).drop_duplicates(subset=["ticker"], keep="first")
    selected_df = pd.read_csv(spec.selected_csv).drop_duplicates(subset=["ticker"], keep="first")
    standard_tickers = set(selected_df["ticker"].astype(str))
    standard_table = detail_df[detail_df["ticker"].astype(str).isin(standard_tickers)].copy().drop_duplicates(subset=["ticker"], keep="first")

    start_date = pd.Timestamp(spec.start_date)
    end_date = pd.Timestamp(spec.end_date)
    price_map_all = load_price_map(detail_df["ticker"].astype(str).tolist(), end_date)
    trade_dates = sorted(set().union(*[set(df[(df.index >= start_date) & (df.index <= end_date)].index.tolist()) for df in price_map_all.values()]))
    crash_tables = build_monthly_crash_tables(detail_df, price_map_all, trade_dates)
    union_tickers = set(standard_table["ticker"].astype(str))
    for tbl in crash_tables.values():
        if not tbl.empty:
            union_tickers.update(tbl["ticker"].astype(str).tolist())
    price_map = {ticker: df for ticker, df in price_map_all.items() if ticker in union_tickers}
    dates = sorted(set().union(*[set(df[(df.index >= start_date) & (df.index <= end_date)].index.tolist()) for df in price_map.values()]))
    metrics_cache = prepare_metric_cache(price_map, dates)
    standard_lookup = standard_table.set_index("ticker").to_dict("index")
    mapping = practical_v2_mapping()

    cash = float(spec.initial_capital)
    positions: dict[str, dict] = {}
    prev_signal = {ticker: False for ticker in price_map}
    equity_rows = []
    buy_count = 0

    for date in dates:
        phase = phase_map.loc[phase_map.index <= date].tail(1)
        phase_name = str(phase.iloc[0]) if not phase.empty else "normal"
        month_key = pd.Timestamp(date.year, date.month, 1)
        active_lookup = crash_tables[month_key].set_index("ticker").to_dict("index") if phase_name == "crash" and not crash_tables[month_key].empty else standard_lookup
        rule_name = "q2_defensive" if phase_name == "crash" else mapping.get(phase_name, "condition2")

        signal_today: dict[str, bool] = {}
        score_today: dict[str, float] = {}
        basis_date_today: dict[str, pd.Timestamp] = {}
        earnings_block_today: dict[str, bool] = {}
        earnings_post_block_today: dict[str, bool] = {}

        for ticker in price_map:
            df = price_map[ticker]
            dte = trading_days_to_next_earnings(date, df, earnings_map.get(ticker, [])) if phase_name not in ignore_pre_phases else None
            dse = trading_days_since_prev_earnings(date, df, earnings_map.get(ticker, [])) if phase_name not in ignore_post_phases else None
            earnings_block_today[ticker] = dte is not None and 1 <= dte <= 1
            earnings_post_block_today[ticker] = dse is not None and 1 <= dse <= post_days
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
            tp = TRADING_RULE["take_profit_pct"] / 100.0 if rule_for_pos == "q2_defensive" else 0.08
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
                cash += positions[ticker]["shares"] * float(exit_price)
                del positions[ticker]

        candidates = []
        for ticker in price_map:
            if ticker in positions:
                continue
            if signal_today.get(ticker, False) and not prev_signal.get(ticker, False):
                candidates.append((ticker, score_today.get(ticker, 0.0)))
        candidates.sort(key=lambda x: x[1], reverse=True)

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
                positions[ticker] = {"shares": shares, "entry_price": fill_price, "rule_name": rule_name}
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

    equity_df = pd.DataFrame(equity_rows)
    drawdown = equity_df["equity"].astype(float) / equity_df["equity"].astype(float).cummax() - 1.0
    return {
        "final_capital": float(cash if not positions else equity_df.iloc[-1]["equity"]),
        "total_return_pct": (float((cash if not positions else equity_df.iloc[-1]["equity"])) / float(spec.initial_capital) - 1.0) * 100.0,
        "num_buys": int(buy_count),
        "max_drawdown_pct": float(drawdown.min()) * 100.0 if not equity_df.empty else 0.0,
    }


def main() -> int:
    spec = next(s for s in batch_specs() if s.name == "q2_2024")
    phase_map = load_phase_map(PHASE_CSV)
    sel = pd.read_csv(spec.selected_csv).drop_duplicates(subset=["ticker"], keep="first")
    tickers = sel["ticker"].astype(str).tolist()
    local_df = load_local_disclosure_dates(tickers)
    irbank_df = fetch_irbank_earnings_cache(tickers)
    have_local = set(local_df["ticker"].astype(str)) if not local_df.empty else set()
    have_irbank = set(irbank_df["ticker"].astype(str)) if not irbank_df.empty else set()
    yf_df = fetch_earnings_cache([t for t in tickers if t not in have_local and t not in have_irbank])
    merged = pd.concat(
        [
            local_df[["ticker", "earnings_date"]] if not local_df.empty else pd.DataFrame(columns=["ticker", "earnings_date"]),
            irbank_df[["ticker", "earnings_date"]] if not irbank_df.empty else pd.DataFrame(columns=["ticker", "earnings_date"]),
            yf_df[["ticker", "earnings_date"]] if not yf_df.empty else pd.DataFrame(columns=["ticker", "earnings_date"]),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["ticker", "earnings_date"])
    earnings_map = build_earnings_map(merged)

    variants = {
        "baseline_pre1_post5": (set(), set(), 5),
        "ignore_crash_both": ({"crash"}, {"crash"}, 5),
        "ignore_crash_highvol_both": ({"crash", "high_vol"}, {"crash", "high_vol"}, 5),
        "pre1_only": (set(), {"crash", "high_vol", "normal", "stable", "settling", "uptrend", "downtrend", "surge", "reversal_up", "reversal_down", "capitulation_end", "overheated_range"}, 0),
        "ignore_crash_post_only": (set(), {"crash"}, 5),
        "ignore_crash_highvol_post_only": (set(), {"crash", "high_vol"}, 5),
    }
    rows = []
    for name, cfg in variants.items():
        ignore_pre_phases, ignore_post_phases, post_days = cfg
        result = run_variant(spec, phase_map, earnings_map, ignore_pre_phases, ignore_post_phases, post_days)
        result["variant"] = name
        rows.append(result)
    out = pd.DataFrame(rows)[["variant", "final_capital", "total_return_pct", "num_buys", "max_drawdown_pct"]]
    out_dir = ROOT / "projects" / "shikiho_text_parser" / "output" / "q2_2024_earnings_scope_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    (out_dir / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

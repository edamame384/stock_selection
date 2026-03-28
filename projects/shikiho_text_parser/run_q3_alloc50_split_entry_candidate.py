from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.shikiho_text_parser import backtest_phase_adaptive_practical_v35_alloc50_candidate as mod


OUT_DIR = ROOT / "projects" / "shikiho_text_parser" / "output" / "q3_alloc50_split_entry_candidate"


def q3_spec() -> mod.TableSpec:
    return mod.TableSpec(
        "q3",
        ROOT / "projects" / "quarterly_ranker" / "output" / "q3_pre_analysis_20250630_aligned" / "q3_pre_shikiho_feature_ranking.csv",
        ROOT / "projects" / "quarterly_ranker" / "output" / "q3_pre_analysis_20250630_aligned" / "threshold_search_post_high_vol" / "operational" / "best_selected_candidates_operational.csv",
        "2025-07-01",
        "2025-09-30",
        3_000_000.0,
    )


def load_cached_earnings_only(tickers: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    local_df = mod.load_local_disclosure_dates(tickers)
    if not local_df.empty:
        frames.append(local_df[["ticker", "earnings_date"]])
    if mod.IRBANK_EARNINGS_CACHE_CSV.exists():
        irbank_df = pd.read_csv(mod.IRBANK_EARNINGS_CACHE_CSV, usecols=["ticker", "earnings_date"])
        irbank_df = irbank_df[irbank_df["ticker"].astype(str).isin(set(tickers))]
        irbank_df["earnings_date"] = pd.to_datetime(irbank_df["earnings_date"], errors="coerce")
        irbank_df = irbank_df.dropna(subset=["earnings_date"])
        if not irbank_df.empty:
            frames.append(irbank_df)
    if mod.EARNINGS_CACHE_CSV.exists():
        yf_df = pd.read_csv(mod.EARNINGS_CACHE_CSV, usecols=["ticker", "earnings_date"])
        yf_df = yf_df[yf_df["ticker"].astype(str).isin(set(tickers))]
        yf_df["earnings_date"] = pd.to_datetime(yf_df["earnings_date"], errors="coerce")
        yf_df = yf_df.dropna(subset=["earnings_date"])
        if not yf_df.empty:
            frames.append(yf_df)
    if not frames:
        return pd.DataFrame(columns=["ticker", "earnings_date"])
    merged = pd.concat(frames, ignore_index=True)
    return merged.drop_duplicates(subset=["ticker", "earnings_date"]).sort_values(["ticker", "earnings_date"])


def run_split_dataset(
    spec: mod.TableSpec,
    phase_map: pd.Series,
    earnings_map: dict[str, list[pd.Timestamp]],
) -> dict:
    detail_df = mod.normalize_static(pd.read_csv(spec.detail_csv)).drop_duplicates(subset=["ticker"], keep="first")

    start_date = pd.Timestamp(spec.start_date)
    end_date = pd.Timestamp(spec.end_date)
    price_map_all = mod.load_price_map(detail_df["ticker"].astype(str).tolist(), end_date)
    trade_dates = sorted(
        set().union(*[set(df[(df.index >= start_date) & (df.index <= end_date)].index.tolist()) for df in price_map_all.values()])
    )
    if not trade_dates:
        return {
            "dataset": spec.name,
            "final_capital": float(spec.initial_capital),
            "total_return_pct": 0.0,
            "num_buys": 0,
            "max_drawdown_pct": 0.0,
            "phase_pnl": {},
            "split_add_count": 0,
        }

    standard_tables = mod.build_interval_standard_tables(spec.name, detail_df, price_map_all, trade_dates, 10)
    crash_tables = mod.build_interval_crash_tables(detail_df, price_map_all, trade_dates, 10)
    _, anchor_for_date = mod.build_rebalance_schedule(trade_dates, 10)
    union_tickers: set[str] = set()
    for tbl in standard_tables.values():
        if not tbl.empty:
            union_tickers.update(tbl["ticker"].astype(str).tolist())
    for tbl in crash_tables.values():
        if not tbl.empty:
            union_tickers.update(tbl["ticker"].astype(str).tolist())
    price_map = {ticker: df for ticker, df in price_map_all.items() if ticker in union_tickers}
    dates = sorted(
        set().union(*[set(df[(df.index >= start_date) & (df.index <= end_date)].index.tolist()) for df in price_map.values()])
    )
    metrics_cache = mod.prepare_metric_cache(price_map, dates)
    mapping = mod.practical_v2_mapping()
    next_trade_date = {dates[i]: dates[i + 1] for i in range(len(dates) - 1)}

    cash = float(spec.initial_capital)
    positions: dict[str, dict] = {}
    prev_signal = {ticker: False for ticker in price_map}
    equity_rows = []
    trade_rows: list[dict] = []
    buy_count = 0
    split_add_count = 0

    for date in dates:
        phase_name = mod.projected_phase_name(date, phase_map, 1, "difficult_v11")
        is_weak_uptrend = mod.weak_uptrend_flag(date, phase_map, 1, "difficult_v11")
        anchor_key = anchor_for_date[date]
        standard_lookup = (
            standard_tables[anchor_key].set_index("ticker").to_dict("index")
            if anchor_key in standard_tables and not standard_tables[anchor_key].empty
            else {}
        )
        active_lookup = (
            crash_tables[anchor_key].set_index("ticker").to_dict("index")
            if phase_name == "crash" and not crash_tables[anchor_key].empty
            else standard_lookup
        )
        rule_name = "q2_defensive" if phase_name == "crash" else mapping.get(phase_name, "condition2")

        signal_today: dict[str, bool] = {}
        score_today: dict[str, float] = {}
        basis_date_today: dict[str, pd.Timestamp] = {}
        earnings_block_today: dict[str, bool] = {}
        earnings_post_block_today: dict[str, bool] = {}

        for ticker in price_map:
            df = price_map[ticker]
            dte = mod.trading_days_to_next_earnings(date, df, earnings_map.get(ticker, []))
            earnings_block_today[ticker] = dte is not None and 1 <= dte <= 1
            dse = mod.trading_days_since_prev_earnings(date, df, earnings_map.get(ticker, []))
            earnings_post_block_today[ticker] = dse is not None and 1 <= dse <= 5
            if ticker not in active_lookup:
                signal_today[ticker] = False
                continue
            metrics = metrics_cache.get(ticker, {}).get(date, {})
            if not metrics:
                signal_today[ticker] = prev_signal.get(ticker, False) if date not in df.index else False
                continue
            if rule_name == "q2_defensive":
                sig, signal_score = mod.post_crash_broad_signal(metrics, active_lookup[ticker])
            else:
                sig, signal_score = mod.eval_signal(rule_name, metrics, active_lookup[ticker])
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
                tp = mod.TRADING_RULE["take_profit_pct"] / 100.0
            elif entry_weak_uptrend:
                tp = 0.08
            elif entry_phase == "high_vol":
                tp = 0.08
            else:
                tp = 0.08
            sl = mod.TRADING_RULE["stop_loss_pct"] / 100.0 if rule_for_pos == "q2_defensive" else 0.05
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
                trade_rows.append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "ticker": ticker,
                        "action": "SELL",
                        "price": float(exit_price),
                        "shares": int(shares),
                        "reason": exit_reason,
                        "nikkei_phase": phase_name,
                    }
                )
                del positions[ticker]

        # one-time next-day add for split targets
        for ticker, pos in list(positions.items()):
            if pos.get("pending_add_date") != date or pos.get("split_completed", False):
                continue
            if not signal_today.get(ticker, False):
                pos["split_completed"] = True
                continue
            if earnings_block_today.get(ticker, False) or earnings_post_block_today.get(ticker, False):
                pos["split_completed"] = True
                continue
            df = price_map[ticker]
            day = df.loc[date]
            prev_close = float(df.loc[basis_date_today[ticker], "Close"])
            entry_limit = mod.TRADING_RULE["entry_limit_pct"] if pos["rule_name"] == "q2_defensive" else 1.5
            trigger_price = prev_close * (1.0 + entry_limit / 100.0)
            day_open = float(day["Open"]) if "Open" in day.index and not pd.isna(day["Open"]) else float(day["Close"])
            day_high = float(day["High"]) if "High" in day.index and not pd.isna(day["High"]) else float(day["Close"])
            if day_high < trigger_price:
                pos["split_completed"] = True
                continue
            fill_price = max(day_open, trigger_price)
            lot_cost = fill_price * 100.0
            if cash < lot_cost or pos.get("lot_cost_at_entry", lot_cost) > 300000.0:
                pos["split_completed"] = True
                continue
            cash -= lot_cost
            prev_shares = pos["shares"]
            prev_cost = pos["entry_price"] * prev_shares
            pos["shares"] = prev_shares + 100
            pos["entry_price"] = (prev_cost + lot_cost) / pos["shares"]
            pos["split_completed"] = True
            trade_rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "action": "BUY",
                    "price": float(fill_price),
                    "shares": 100,
                    "reason": f"{pos['rule_name']}_split_add",
                    "nikkei_phase": phase_name,
                }
            )
            buy_count += 1
            split_add_count += 1

        candidates = []
        for ticker in price_map:
            if ticker in positions:
                continue
            if signal_today.get(ticker, False) and not prev_signal.get(ticker, False):
                df = price_map[ticker]
                if date not in df.index or ticker not in basis_date_today:
                    continue
                day = df.loc[date]
                prev_close = float(df.loc[basis_date_today[ticker], "Close"])
                entry_limit = mod.TRADING_RULE["entry_limit_pct"] if rule_name == "q2_defensive" else 1.5
                trigger_price = prev_close * (1.0 + entry_limit / 100.0)
                day_open = float(day["Open"]) if "Open" in day.index and not pd.isna(day["Open"]) else float(day["Close"])
                fill_price = max(day_open, trigger_price)
                lot_cost = fill_price * 100.0
                candidates.append((ticker, score_today.get(ticker, 0.0), lot_cost))
        candidates.sort(key=lambda x: x[1], reverse=True)

        split_targets = {ticker for ticker, _score, lot_cost in candidates[:2] if lot_cost <= 300000.0}

        for ticker, _score, _lot_cost in candidates:
            if earnings_block_today.get(ticker, False) or earnings_post_block_today.get(ticker, False):
                continue
            df = price_map[ticker]
            day = df.loc[date]
            prev_close = float(df.loc[basis_date_today[ticker], "Close"])
            entry_limit = mod.TRADING_RULE["entry_limit_pct"] if rule_name == "q2_defensive" else 1.5
            trigger_price = prev_close * (1.0 + entry_limit / 100.0)
            day_open = float(day["Open"]) if "Open" in day.index and not pd.isna(day["Open"]) else float(day["Close"])
            day_high = float(day["High"]) if "High" in day.index and not pd.isna(day["High"]) else float(day["Close"])
            if day_high < trigger_price:
                continue
            fill_price = max(day_open, trigger_price)
            lot_cost = fill_price * 100.0

            if ticker in split_targets:
                shares = 100 if cash >= lot_cost else 0
                pending_add_date = next_trade_date.get(date)
                split_completed = False
            else:
                if lot_cost < 500000.0:
                    shares = min(int(500000.0 // lot_cost), int(cash // lot_cost)) * 100
                else:
                    shares = 100 if cash >= lot_cost else 0
                pending_add_date = None
                split_completed = True

            if shares >= 100 and shares * fill_price <= cash:
                cash -= shares * fill_price
                positions[ticker] = {
                    "shares": shares,
                    "entry_price": fill_price,
                    "rule_name": rule_name,
                    "entry_phase": phase_name,
                    "entry_weak_uptrend": is_weak_uptrend,
                    "pending_add_date": pending_add_date,
                    "split_completed": split_completed,
                    "lot_cost_at_entry": lot_cost,
                }
                trade_rows.append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "ticker": ticker,
                        "action": "BUY",
                        "price": float(fill_price),
                        "shares": int(shares),
                        "reason": f"{rule_name}_entry",
                        "nikkei_phase": phase_name,
                    }
                )
                buy_count += 1

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
            trade_rows.append(
                {
                    "date": latest_date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "action": "SELL",
                    "price": px,
                    "shares": int(positions[ticker]["shares"]),
                    "reason": "end_of_backtest",
                    "nikkei_phase": mod.projected_phase_name(latest_date, phase_map, 1, "difficult_v11"),
                }
            )
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
        "phase_pnl": phase_pnl,
        "split_add_count": int(split_add_count),
        "trade_log": pd.DataFrame(trade_rows),
        "equity_curve": equity_df,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    spec = q3_spec()
    phase_map = mod.load_phase_map(mod.PHASE_CSV)
    tickers = mod.collect_relevant_tickers(spec)
    earnings_map = mod.build_earnings_map(load_cached_earnings_only(tickers))
    result = run_split_dataset(spec, phase_map, earnings_map)

    payload = {k: v for k, v in result.items() if k not in {"trade_log", "equity_curve"}}
    payload["variant"] = "alloc50_split_top2_under300k_q3"
    payload["split_rule"] = "initial 100 shares, add 100 shares next day if signal continues"
    payload["split_target_rule"] = "100-share lot cost <= 300000 and top 2 candidates of the day"
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    result["trade_log"].to_csv(OUT_DIR / "trade_log.csv", index=False, encoding="utf-8-sig")
    result["equity_curve"].to_csv(OUT_DIR / "equity_curve.csv", index=False, encoding="utf-8-sig")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

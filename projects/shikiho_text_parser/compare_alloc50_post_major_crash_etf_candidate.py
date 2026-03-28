from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.shikiho_text_parser import backtest_phase_adaptive_practical_v35_alloc50_candidate as base_mod  # noqa: E402
from projects.shikiho_text_parser.compare_phase_adaptive_practical_v3 import TableSpec  # noqa: E402


OUT_DIR = ROOT / "projects" / "shikiho_text_parser" / "output" / "compare_alloc50_post_major_crash_etf_candidate"
ETF_TICKERS = ["1306.T", "1321.T", "2516.T"]
ETF_PHASES = {"high_vol", "capitulation_end", "settling", "normal"}
ETF_TP = 0.06
ETF_SL = 0.03


def build_post_major_state(phase_df: pd.DataFrame) -> pd.DataFrame:
    x = phase_df.copy().reset_index(drop=True)
    x["major_crash"] = (x["phase_name"].astype(str) == "crash") & ((x["ret5"] <= -0.08) | (x["dd20"] <= -0.14))
    mode = False
    recovery_streak = 0
    states = []
    for _, row in x.iterrows():
        phase = str(row["phase_name"])
        dd20 = float(row["dd20"]) if pd.notna(row["dd20"]) else None
        if bool(row["major_crash"]):
            mode = True
            recovery_streak = 0
        elif mode:
            if phase in {"stable", "uptrend"} and dd20 is not None and dd20 >= -0.05:
                recovery_streak += 1
            else:
                recovery_streak = 0
            if recovery_streak >= 5:
                mode = False
        states.append(bool(mode))
    x["post_major_crash_mode"] = states
    return x


def prev_high_break_signal(df: pd.DataFrame, basis_date: pd.Timestamp, trade_date: pd.Timestamp) -> tuple[bool, float | None]:
    hist = df.loc[:basis_date]
    if len(hist) < 3 or trade_date not in df.index:
        return False, None
    prev_high = float(hist.iloc[-1]["High"])
    trigger = prev_high * 1.002
    day_high = float(df.loc[trade_date, "High"]) if not pd.isna(df.loc[trade_date, "High"]) else float(df.loc[trade_date, "Close"])
    return day_high >= trigger, trigger


def q2_spec() -> TableSpec:
    return TableSpec(
        "q2_2024",
        ROOT / "projects" / "quarterly_ranker" / "output" / "q2_2024_pre_analysis_20240630_aligned" / "q2_2024_pre_shikiho_feature_ranking.csv",
        ROOT / "projects" / "quarterly_ranker" / "output" / "q2_2024_pre_analysis_20240630_aligned" / "operational" / "q2_2024_pre_selected_candidates_operational.csv",
        "2024-07-01",
        "2024-09-30",
        3_000_000.0,
    )


def q2025_2q_spec() -> TableSpec:
    return TableSpec(
        "q2_2024",
        ROOT / "projects" / "quarterly_ranker" / "output" / "2025_2q_pre_analysis_20250331_candidate" / "pre_shikiho_feature_ranking.csv",
        ROOT / "projects" / "quarterly_ranker" / "output" / "2025_2q_pre_analysis_20250331_candidate" / "operational" / "pre_selected_candidates_operational.csv",
        "2025-04-01",
        "2025-06-30",
        3_000_000.0,
    )


def prepare_earnings_map(spec: TableSpec) -> dict[str, list[pd.Timestamp]]:
    detail_df = pd.read_csv(spec.detail_csv)
    tickers = set(detail_df["ticker"].astype(str).tolist())
    sources: list[pd.DataFrame] = []
    for path in [
        ROOT / "data" / "earnings_cache" / "irbank_earnings_dates.csv",
        ROOT / "data" / "earnings_cache" / "yf_earnings_dates.csv",
        ROOT / "projects" / "shikiho_text_parser" / "output" / "parsed_summary.csv",
    ]:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "ticker" not in df.columns and "ticker_code" in df.columns:
            df["ticker"] = df["ticker_code"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True) + ".T"
        if "earnings_date" not in df.columns and "disclosure_date" in df.columns:
            df["earnings_date"] = df["disclosure_date"]
        if "ticker" not in df.columns or "earnings_date" not in df.columns:
            continue
        df = df[["ticker", "earnings_date"]].copy()
        df["ticker"] = df["ticker"].astype(str).str.strip()
        df = df[df["ticker"].isin(tickers)]
        sources.append(df)
    merged = pd.concat(sources, ignore_index=True) if sources else pd.DataFrame(columns=["ticker", "earnings_date"])
    if not merged.empty:
        merged["earnings_date"] = pd.to_datetime(merged["earnings_date"], errors="coerce")
        merged = merged.dropna(subset=["earnings_date"]).drop_duplicates(subset=["ticker", "earnings_date"]).sort_values(["ticker", "earnings_date"])
    return base_mod.build_earnings_map(merged)


def run_dataset_with_etf(spec: TableSpec, phase_df: pd.DataFrame, earnings_map: dict[str, list[pd.Timestamp]], etf_ticker: str) -> dict:
    detail_df = base_mod.normalize_static(pd.read_csv(spec.detail_csv)).drop_duplicates(subset=["ticker"], keep="first")
    start_date = pd.Timestamp(spec.start_date)
    end_date = pd.Timestamp(spec.end_date)
    tickers = detail_df["ticker"].astype(str).tolist()
    if etf_ticker not in tickers:
        tickers.append(etf_ticker)
    price_map_all = base_mod.load_price_map(tickers, end_date)
    trade_dates = sorted(set().union(*[set(df[(df.index >= start_date) & (df.index <= end_date)].index.tolist()) for df in price_map_all.values()]))
    if not trade_dates:
        return {"dataset": spec.name, "final_capital": float(spec.initial_capital), "total_return_pct": 0.0, "num_buys": 0, "max_drawdown_pct": 0.0, "phase_pnl": {}, "trade_log": pd.DataFrame(), "equity_curve": pd.DataFrame()}

    stock_price_map_all = {k: v for k, v in price_map_all.items() if k != etf_ticker}
    standard_tables = base_mod.build_interval_standard_tables(spec.name, detail_df, stock_price_map_all, trade_dates, 10)
    crash_tables = base_mod.build_interval_crash_tables(detail_df, stock_price_map_all, trade_dates, 10)
    _, anchor_for_date = base_mod.build_rebalance_schedule(trade_dates, 10)
    union_tickers: set[str] = set()
    for tbl in standard_tables.values():
        if not tbl.empty:
            union_tickers.update(tbl["ticker"].astype(str).tolist())
    for tbl in crash_tables.values():
        if not tbl.empty:
            union_tickers.update(tbl["ticker"].astype(str).tolist())
    union_tickers.add(etf_ticker)
    price_map = {ticker: df for ticker, df in price_map_all.items() if ticker in union_tickers}
    dates = sorted(set().union(*[set(df[(df.index >= start_date) & (df.index <= end_date)].index.tolist()) for df in price_map.values()]))
    metrics_cache = base_mod.prepare_metric_cache({k: v for k, v in price_map.items() if k != etf_ticker}, dates)
    mapping = base_mod.practical_v2_mapping()
    phase_map = base_mod.load_phase_map(base_mod.PHASE_CSV)
    phase_state = phase_df.set_index("Date").sort_index()

    cash = float(spec.initial_capital)
    positions: dict[str, dict] = {}
    prev_signal = {ticker: False for ticker in price_map if ticker != etf_ticker}
    equity_rows = []
    trade_rows: list[dict] = []
    buy_count = 0

    for date in dates:
        phase_name = base_mod.projected_phase_name(date, phase_map, 1, "difficult_v11")
        is_weak_uptrend = base_mod.weak_uptrend_flag(date, phase_map, 1, "difficult_v11")
        anchor_key = anchor_for_date[date]
        standard_lookup = standard_tables[anchor_key].set_index("ticker").to_dict("index") if anchor_key in standard_tables and not standard_tables[anchor_key].empty else {}
        active_lookup = crash_tables[anchor_key].set_index("ticker").to_dict("index") if phase_name == "crash" and not crash_tables[anchor_key].empty else standard_lookup
        rule_name = "q2_defensive" if phase_name == "crash" else mapping.get(phase_name, "condition2")

        prior_phase_row = phase_state.loc[phase_state.index < date].tail(1)
        post_major = False if prior_phase_row.empty else bool(prior_phase_row.iloc[0]["post_major_crash_mode"])
        post_major_phase = "" if prior_phase_row.empty else str(prior_phase_row.iloc[0]["phase_name"])
        etf_active = post_major and post_major_phase in ETF_PHASES

        signal_today: dict[str, bool] = {}
        score_today: dict[str, float] = {}
        basis_date_today: dict[str, pd.Timestamp] = {}
        earnings_block_today: dict[str, bool] = {}
        earnings_post_block_today: dict[str, bool] = {}

        for ticker in [t for t in price_map if t != etf_ticker]:
            df = price_map[ticker]
            dte = base_mod.trading_days_to_next_earnings(date, df, earnings_map.get(ticker, []))
            earnings_block_today[ticker] = dte is not None and 1 <= dte <= 1
            dse = base_mod.trading_days_since_prev_earnings(date, df, earnings_map.get(ticker, []))
            earnings_post_block_today[ticker] = dse is not None and 1 <= dse <= 5
            if ticker not in active_lookup:
                signal_today[ticker] = False
                continue
            metrics = metrics_cache.get(ticker, {}).get(date, {})
            if not metrics:
                signal_today[ticker] = prev_signal.get(ticker, False) if date not in df.index else False
                continue
            if rule_name == "q2_defensive":
                sig, signal_score = base_mod.post_crash_broad_signal(metrics, active_lookup[ticker])
            else:
                sig, signal_score = base_mod.eval_signal(rule_name, metrics, active_lookup[ticker])
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
            if ticker == etf_ticker:
                etf_basis_idx = df.index[df.index < date]
                etf_sig = False
                if len(etf_basis_idx):
                    etf_sig, _trigger = prev_high_break_signal(df, etf_basis_idx[-1], date)
                ret = close_price / entry_price - 1.0
                exit_reason = None
                if ret >= ETF_TP:
                    exit_reason = "take_profit"
                elif ret <= -ETF_SL:
                    exit_reason = "stop_loss"
                elif not etf_active or not etf_sig:
                    exit_reason = "sell_signal"
                if exit_reason:
                    cash += positions[ticker]["shares"] * close_price
                    trade_rows.append({"date": date.strftime("%Y-%m-%d"), "ticker": ticker, "action": "SELL", "price": close_price, "shares": int(positions[ticker]["shares"]), "reason": exit_reason, "nikkei_phase": phase_name})
                    del positions[ticker]
                continue

            rule_for_pos = positions[ticker]["rule_name"]
            entry_phase = positions[ticker].get("entry_phase", "")
            entry_weak_uptrend = bool(positions[ticker].get("entry_weak_uptrend", False))
            if rule_for_pos == "q2_defensive":
                tp = base_mod.TRADING_RULE["take_profit_pct"] / 100.0
            elif entry_weak_uptrend:
                tp = 0.08
            elif entry_phase == "high_vol":
                tp = 0.08
            else:
                tp = 0.08
            sl = base_mod.TRADING_RULE["stop_loss_pct"] / 100.0 if rule_for_pos == "q2_defensive" else 0.05
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
                trade_rows.append({"date": date.strftime("%Y-%m-%d"), "ticker": ticker, "action": "SELL", "price": float(exit_price), "shares": int(shares), "reason": exit_reason, "nikkei_phase": phase_name})
                del positions[ticker]

        if etf_ticker not in positions and etf_active:
            etf_df = price_map[etf_ticker]
            basis_idx = etf_df.index[etf_df.index < date]
            if len(basis_idx):
                etf_ok, trigger = prev_high_break_signal(etf_df, basis_idx[-1], date)
                if etf_ok:
                    day = etf_df.loc[date]
                    day_open = float(day["Open"]) if "Open" in day.index and not pd.isna(day["Open"]) else float(day["Close"])
                    fill_price = max(day_open, float(trigger))
                    shares = int((cash // (fill_price * 100)) * 100)
                    if shares >= 100 and shares * fill_price <= cash:
                        cash -= shares * fill_price
                        positions[etf_ticker] = {"shares": shares, "entry_price": fill_price, "rule_name": "post_major_crash_etf"}
                        trade_rows.append({"date": date.strftime("%Y-%m-%d"), "ticker": etf_ticker, "action": "BUY", "price": float(fill_price), "shares": int(shares), "reason": "post_major_crash_etf_entry", "nikkei_phase": phase_name})
                        buy_count += 1

        if not etf_active:
            candidates = []
            for ticker in [t for t in price_map if t != etf_ticker]:
                if ticker in positions:
                    continue
                if signal_today.get(ticker, False) and not prev_signal.get(ticker, False):
                    candidates.append((ticker, score_today.get(ticker, 0.0)))
            candidates.sort(key=lambda x: x[1], reverse=True)

            for ticker, _score in candidates:
                if earnings_block_today.get(ticker, False) or earnings_post_block_today.get(ticker, False):
                    continue
                df = price_map[ticker]
                day = df.loc[date]
                prev_close = float(df.loc[basis_date_today[ticker], "Close"])
                entry_limit = base_mod.TRADING_RULE["entry_limit_pct"] if rule_name == "q2_defensive" else 1.5
                trigger_price = prev_close * (1.0 + entry_limit / 100.0)
                day_open = float(day["Open"]) if "Open" in day.index and not pd.isna(day["Open"]) else float(day["Close"])
                day_high = float(day["High"]) if "High" in day.index and not pd.isna(day["High"]) else float(day["Close"])
                if day_high < trigger_price:
                    continue
                fill_price = max(day_open, trigger_price)
                lot_cost = fill_price * 100.0
                if lot_cost < 500000.0:
                    shares = min(int(500000.0 // lot_cost), int(cash // lot_cost)) * 100
                else:
                    shares = 100 if cash >= lot_cost else 0
                if shares >= 100 and shares * fill_price <= cash:
                    cash -= shares * fill_price
                    positions[ticker] = {"shares": shares, "entry_price": fill_price, "rule_name": rule_name, "entry_phase": phase_name, "entry_weak_uptrend": is_weak_uptrend}
                    trade_rows.append({"date": date.strftime("%Y-%m-%d"), "ticker": ticker, "action": "BUY", "price": float(fill_price), "shares": int(shares), "reason": f"{rule_name}_entry", "nikkei_phase": phase_name})
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
            trade_rows.append({"date": latest_date.strftime("%Y-%m-%d"), "ticker": ticker, "action": "SELL", "price": px, "shares": int(positions[ticker]["shares"]), "reason": "end_of_backtest", "nikkei_phase": base_mod.projected_phase_name(latest_date, phase_map, 1, "difficult_v11")})
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
        "dataset": spec.start_date,
        "final_capital": float(cash),
        "total_return_pct": (float(cash) / float(spec.initial_capital) - 1.0) * 100.0,
        "num_buys": int(buy_count),
        "max_drawdown_pct": max_drawdown_pct,
        "phase_pnl": phase_pnl,
        "trade_log": pd.DataFrame(trade_rows),
        "equity_curve": equity_df,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--etf-tickers", nargs="*", default=None)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase_df = build_post_major_state(pd.read_csv(base_mod.PHASE_CSV, skiprows=[1]).rename(columns={"phase": "phase_name"}))
    phase_df["Date"] = pd.to_datetime(phase_df["Date"])
    for col in ["ret5", "dd20", "vol10"]:
        phase_df[col] = pd.to_numeric(phase_df[col], errors="coerce")

    specs = {
        "q2_2024": q2_spec(),
        "2025_2q": q2025_2q_spec(),
    }
    if args.datasets:
        wanted = set(args.datasets)
        specs = {k: v for k, v in specs.items() if k in wanted}
    etf_tickers = ETF_TICKERS if not args.etf_tickers else args.etf_tickers
    rows = []
    for name, spec in specs.items():
        earnings_map = prepare_earnings_map(spec)
        for etf_ticker in etf_tickers:
            result = run_dataset_with_etf(spec, phase_df, earnings_map, etf_ticker)
            out_dir = OUT_DIR / name / etf_ticker.replace(".", "_")
            out_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "variant": f"alloc50_post_major_crash_etf_{etf_ticker}_prev_high_break_tp6_sl3",
                "dataset": name,
                "etf_ticker": etf_ticker,
                "etf_rule": "post_major_crash_mode + prev_high_break + tp6/sl3",
                **{k: v for k, v in result.items() if k not in {"trade_log", "equity_curve"}},
            }
            (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            result["trade_log"].to_csv(out_dir / "trade_log.csv", index=False, encoding="utf-8-sig")
            result["equity_curve"].to_csv(out_dir / "equity_curve.csv", index=False, encoding="utf-8-sig")
            rows.append(payload)

    pd.DataFrame(rows).to_csv(OUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows)[["dataset", "total_return_pct", "max_drawdown_pct", "num_buys"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

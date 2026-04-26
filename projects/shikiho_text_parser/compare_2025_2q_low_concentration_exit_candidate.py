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


OUT_DIR = ROOT / "projects" / "shikiho_text_parser" / "output" / "compare_2025_2q_low_concentration_exit_candidate"
DETAIL_CSV = ROOT / "projects" / "quarterly_ranker" / "output" / "2025_2q_low_concentration_sector_candidate" / "pre_shikiho_feature_ranking_filtered.csv"
SELECTED_CSV = ROOT / "projects" / "quarterly_ranker" / "output" / "2025_2q_low_concentration_sector_candidate" / "operational" / "pre_selected_candidates_operational.csv"


def spec() -> TableSpec:
    return TableSpec(
        "q2_2024",
        DETAIL_CSV,
        SELECTED_CSV,
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


def run_dataset_custom(spec: TableSpec, earnings_map: dict[str, list[pd.Timestamp]], take_profit_pct: float, stop_loss_pct: float) -> dict:
    detail_df = base_mod.normalize_static(pd.read_csv(spec.detail_csv)).drop_duplicates(subset=["ticker"], keep="first")
    start_date = pd.Timestamp(spec.start_date)
    end_date = pd.Timestamp(spec.end_date)
    price_map_all = base_mod.load_price_map(detail_df["ticker"].astype(str).tolist(), end_date)
    trade_dates = sorted(set().union(*[set(df[(df.index >= start_date) & (df.index <= end_date)].index.tolist()) for df in price_map_all.values()]))
    if not trade_dates:
        return {
            "dataset": spec.name,
            "final_capital": float(spec.initial_capital),
            "total_return_pct": 0.0,
            "num_buys": 0,
            "max_drawdown_pct": 0.0,
            "phase_pnl": {},
            "trade_log": pd.DataFrame(),
            "equity_curve": pd.DataFrame(),
        }

    standard_tables = base_mod.build_interval_standard_tables(spec.name, detail_df, price_map_all, trade_dates, 10)
    crash_tables = base_mod.build_interval_crash_tables(detail_df, price_map_all, trade_dates, 10)
    _, anchor_for_date = base_mod.build_rebalance_schedule(trade_dates, 10)
    union_tickers: set[str] = set()
    for tbl in standard_tables.values():
        if not tbl.empty:
            union_tickers.update(tbl["ticker"].astype(str).tolist())
    for tbl in crash_tables.values():
        if not tbl.empty:
            union_tickers.update(tbl["ticker"].astype(str).tolist())
    price_map = {ticker: df for ticker, df in price_map_all.items() if ticker in union_tickers}
    dates = sorted(set().union(*[set(df[(df.index >= start_date) & (df.index <= end_date)].index.tolist()) for df in price_map.values()]))
    metrics_cache = base_mod.prepare_metric_cache(price_map, dates)
    mapping = base_mod.practical_v2_mapping()
    phase_map = base_mod.load_phase_map(base_mod.PHASE_CSV)

    cash = float(spec.initial_capital)
    positions: dict[str, dict] = {}
    prev_signal = {ticker: False for ticker in price_map}
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

        signal_today: dict[str, bool] = {}
        score_today: dict[str, float] = {}
        basis_date_today: dict[str, pd.Timestamp] = {}
        earnings_block_today: dict[str, bool] = {}
        earnings_post_block_today: dict[str, bool] = {}

        for ticker in price_map:
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
            rule_for_pos = positions[ticker]["rule_name"]
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
            if exit_reason is None and ret >= take_profit_pct / 100.0:
                exit_reason = "take_profit"
                exit_price = close_price
            elif exit_reason is None and ret <= -stop_loss_pct / 100.0:
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

        candidates: list[tuple[str, float]] = []
        for ticker in price_map:
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
        "dataset": "2025_2q_low_concentration",
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
    parser.add_argument("--variants", nargs="*", default=None)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s = spec()
    earnings_map = prepare_earnings_map(s)
    variants = [
        {"name": "baseline_tp8_sl5", "tp": 8.0, "sl": 5.0},
        {"name": "tp6_sl4", "tp": 6.0, "sl": 4.0},
        {"name": "tp5_sl4", "tp": 5.0, "sl": 4.0},
        {"name": "tp6_sl3", "tp": 6.0, "sl": 3.0},
    ]
    if args.variants:
        want = set(args.variants)
        variants = [v for v in variants if v["name"] in want]
    rows = []
    for variant in variants:
        result = run_dataset_custom(s, earnings_map, variant["tp"], variant["sl"])
        out_dir = OUT_DIR / variant["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "variant": variant["name"],
            "take_profit_pct": variant["tp"],
            "stop_loss_pct": variant["sl"],
            **{k: v for k, v in result.items() if k not in {"trade_log", "equity_curve"}},
        }
        (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        result["trade_log"].to_csv(out_dir / "trade_log.csv", index=False, encoding="utf-8-sig")
        result["equity_curve"].to_csv(out_dir / "equity_curve.csv", index=False, encoding="utf-8-sig")
        rows.append(payload)
    pd.DataFrame(rows).sort_values("total_return_pct", ascending=False).to_csv(OUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows)[["variant", "total_return_pct", "max_drawdown_pct", "num_buys"]].sort_values("total_return_pct", ascending=False).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

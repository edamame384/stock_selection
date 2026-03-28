from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.shikiho_text_parser import backtest_phase_adaptive_practical_v35_alloc50_candidate as base_mod  # noqa: E402
from projects.shikiho_text_parser.compare_phase_adaptive_practical_v3 import TableSpec  # noqa: E402


OUT_DIR = ROOT / "projects" / "shikiho_text_parser" / "output" / "compare_2025_2q_post_major_stock_trend_tuning_candidate"
DETAIL_CSV = ROOT / "projects" / "quarterly_ranker" / "output" / "2025_2q_low_concentration_sector_candidate" / "pre_shikiho_feature_ranking_filtered.csv"
SELECTED_CSV = ROOT / "projects" / "quarterly_ranker" / "output" / "2025_2q_low_concentration_sector_candidate" / "operational" / "pre_selected_candidates_operational.csv"
POST_MAJOR_PHASES = {"high_vol", "capitulation_end", "settling", "normal"}
POST_MAJOR_TP = 0.06
POST_MAJOR_SL = 0.03

VARIANTS = [
    {"name": "baseline_break0p2_topall_cool0", "break_mult": 1.002, "top_n": 999, "cooldown": 0},
    {"name": "break0p2_top3_cool0", "break_mult": 1.002, "top_n": 3, "cooldown": 0},
    {"name": "break0p2_top2_cool0", "break_mult": 1.002, "top_n": 2, "cooldown": 0},
    {"name": "break0p5_top3_cool0", "break_mult": 1.005, "top_n": 3, "cooldown": 0},
    {"name": "break0p5_top2_cool0", "break_mult": 1.005, "top_n": 2, "cooldown": 0},
    {"name": "break0p2_top3_cool3", "break_mult": 1.002, "top_n": 3, "cooldown": 3},
    {"name": "break0p5_top3_cool3", "break_mult": 1.005, "top_n": 3, "cooldown": 3},
]


def spec() -> TableSpec:
    return TableSpec(
        "q3",
        DETAIL_CSV,
        SELECTED_CSV,
        "2025-04-01",
        "2025-06-30",
        3_000_000.0,
    )


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


def prev_high_break_signal(df: pd.DataFrame, basis_date: pd.Timestamp, trade_date: pd.Timestamp, break_mult: float) -> tuple[bool, float | None]:
    hist = df.loc[:basis_date]
    if len(hist) < 3 or trade_date not in df.index:
        return False, None
    prev_high = float(hist.iloc[-1]["High"])
    trigger = prev_high * break_mult
    day_high = float(df.loc[trade_date, "High"]) if not pd.isna(df.loc[trade_date, "High"]) else float(df.loc[trade_date, "Close"])
    return day_high >= trigger, trigger


def prepare_earnings_map(s: TableSpec) -> dict[str, list[pd.Timestamp]]:
    detail_df = pd.read_csv(s.detail_csv)
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


def run_variant(variant: dict, phase_state: pd.DataFrame, earnings_map: dict[str, list[pd.Timestamp]]) -> dict:
    s = spec()
    detail_df = base_mod.normalize_static(pd.read_csv(s.detail_csv)).drop_duplicates(subset=["ticker"], keep="first")
    start_date = pd.Timestamp(s.start_date)
    end_date = pd.Timestamp(s.end_date)
    price_map_all = base_mod.load_price_map(detail_df["ticker"].astype(str).tolist(), end_date)
    trade_dates = sorted(set().union(*[set(df[(df.index >= start_date) & (df.index <= end_date)].index.tolist()) for df in price_map_all.values()]))
    standard_tables = base_mod.build_interval_standard_tables(s.name, detail_df, price_map_all, trade_dates, 10)
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

    cash = float(s.initial_capital)
    positions: dict[str, dict] = {}
    prev_signal = {ticker: False for ticker in price_map}
    trade_rows: list[dict] = []
    equity_rows: list[dict] = []
    buy_count = 0
    last_post_major_entry: dict[str, pd.Timestamp] = {}

    for date in dates:
        phase_name = base_mod.projected_phase_name(date, phase_map, 1, "difficult_v11")
        is_weak_uptrend = base_mod.weak_uptrend_flag(date, phase_map, 1, "difficult_v11")
        anchor_key = anchor_for_date[date]
        standard_lookup = standard_tables[anchor_key].set_index("ticker").to_dict("index") if anchor_key in standard_tables and not standard_tables[anchor_key].empty else {}
        active_lookup = crash_tables[anchor_key].set_index("ticker").to_dict("index") if phase_name == "crash" and anchor_key in crash_tables and not crash_tables[anchor_key].empty else standard_lookup
        rule_name = "q2_defensive" if phase_name == "crash" else mapping.get(phase_name, "condition2")

        prior_phase_row = phase_state.loc[phase_state.index < date].tail(1)
        post_major = False if prior_phase_row.empty else bool(prior_phase_row.iloc[0]["post_major_crash_mode"])
        post_major_phase = "" if prior_phase_row.empty else str(prior_phase_row.iloc[0]["phase_name"])
        post_major_active = post_major and post_major_phase in POST_MAJOR_PHASES

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
                signal_today[ticker] = False
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
            ret = close_price / entry_price - 1.0
            exit_reason = None
            exit_price = None

            if earnings_block_today.get(ticker, False):
                exit_reason = "pre_earnings_exit"
                exit_price = open_price
            elif phase_name == "crash":
                prev_idx = df.index[df.index < date]
                if len(prev_idx):
                    prev_close = float(df.loc[prev_idx[-1], "Close"])
                    early_trigger = prev_close * 0.98
                    if low_price <= early_trigger:
                        exit_reason = "crash_early_exit"
                        exit_price = min(open_price, early_trigger)

            if positions[ticker].get("entry_mode") == "post_major_trend":
                if exit_reason is None and ret >= POST_MAJOR_TP:
                    exit_reason = "take_profit"
                    exit_price = close_price
                elif exit_reason is None and ret <= -POST_MAJOR_SL:
                    exit_reason = "stop_loss"
                    exit_price = close_price
                else:
                    basis_idx = df.index[df.index < date]
                    hold_sig = False
                    if len(basis_idx) and post_major_active:
                        hold_sig, _ = prev_high_break_signal(df, basis_idx[-1], date, variant["break_mult"])
                    if exit_reason is None and (not post_major_active or not hold_sig):
                        exit_reason = "sell_signal"
                        exit_price = close_price
            else:
                tp = base_mod.TRADING_RULE["take_profit_pct"] / 100.0 if positions[ticker]["rule_name"] == "q2_defensive" else 0.08
                sl = base_mod.TRADING_RULE["stop_loss_pct"] / 100.0 if positions[ticker]["rule_name"] == "q2_defensive" else 0.05
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

        candidates: list[tuple[str, float, float, str]] = []
        for ticker in price_map:
            if ticker in positions:
                continue
            if earnings_block_today.get(ticker, False) or earnings_post_block_today.get(ticker, False):
                continue
            df = price_map[ticker]
            if date not in df.index:
                continue

            if post_major_active:
                last_entry = last_post_major_entry.get(ticker)
                if last_entry is not None and (date - last_entry).days <= variant["cooldown"]:
                    continue
                basis_idx = df.index[df.index < date]
                if not len(basis_idx):
                    continue
                sig, trigger = prev_high_break_signal(df, basis_idx[-1], date, variant["break_mult"])
                if not sig or trigger is None:
                    continue
                candidates.append((ticker, score_today.get(ticker, 0.0), float(trigger), "post_major_trend"))
            else:
                if signal_today.get(ticker, False) and not prev_signal.get(ticker, False):
                    prev_close = float(df.loc[basis_date_today[ticker], "Close"])
                    entry_limit = base_mod.TRADING_RULE["entry_limit_pct"] if rule_name == "q2_defensive" else 1.5
                    trigger = prev_close * (1.0 + entry_limit / 100.0)
                    candidates.append((ticker, score_today.get(ticker, 0.0), float(trigger), "base"))

        candidates.sort(key=lambda x: x[1], reverse=True)
        post_major_candidates = [c for c in candidates if c[3] == "post_major_trend"][: variant["top_n"]]
        base_candidates = [c for c in candidates if c[3] == "base"]
        final_candidates = post_major_candidates + base_candidates

        for ticker, _score, trigger_price, entry_mode in final_candidates:
            if ticker in positions:
                continue
            df = price_map[ticker]
            day = df.loc[date]
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
            if shares < 100 or shares * fill_price > cash:
                continue
            cash -= shares * fill_price
            positions[ticker] = {
                "shares": shares,
                "entry_price": fill_price,
                "rule_name": rule_name,
                "entry_phase": phase_name,
                "entry_weak_uptrend": is_weak_uptrend,
                "entry_mode": entry_mode,
            }
            if entry_mode == "post_major_trend":
                last_post_major_entry[ticker] = date
                reason = "post_major_prev_high_break_entry"
            else:
                reason = f"{rule_name}_entry"
            trade_rows.append({"date": date.strftime("%Y-%m-%d"), "ticker": ticker, "action": "BUY", "price": float(fill_price), "shares": int(shares), "reason": reason, "nikkei_phase": phase_name})
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

    equity_df = pd.DataFrame(equity_rows)
    equity_df["pnl_day"] = equity_df["equity"].diff().fillna(equity_df["equity"] - s.initial_capital)
    phase_pnl = equity_df.groupby("nikkei_phase")["pnl_day"].sum().to_dict()
    drawdown = equity_df["equity"].astype(float) / equity_df["equity"].astype(float).cummax() - 1.0
    return {
        "variant": variant["name"],
        "dataset": "2025_2q",
        "final_capital": float(cash),
        "total_return_pct": (float(cash) / float(s.initial_capital) - 1.0) * 100.0,
        "num_buys": int(buy_count),
        "max_drawdown_pct": float(drawdown.min()) * 100.0,
        "phase_pnl": phase_pnl,
        "trade_log": pd.DataFrame(trade_rows),
        "equity_curve": equity_df,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase_df = build_post_major_state(pd.read_csv(base_mod.PHASE_CSV, skiprows=[1]).rename(columns={"phase": "phase_name"}))
    phase_df["Date"] = pd.to_datetime(phase_df["Date"])
    for col in ["ret5", "dd20", "vol10"]:
        phase_df[col] = pd.to_numeric(phase_df[col], errors="coerce")
    phase_state = phase_df.set_index("Date").sort_index()
    earnings_map = prepare_earnings_map(spec())
    rows = []
    for variant in VARIANTS:
        result = run_variant(variant, phase_state, earnings_map)
        out_dir = OUT_DIR / variant["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(json.dumps({k: v for k, v in result.items() if k not in {"trade_log", "equity_curve"}}, ensure_ascii=False, indent=2), encoding="utf-8")
        result["trade_log"].to_csv(out_dir / "trade_log.csv", index=False, encoding="utf-8-sig")
        result["equity_curve"].to_csv(out_dir / "equity_curve.csv", index=False, encoding="utf-8-sig")
        rows.append({k: v for k, v in result.items() if k not in {"trade_log", "equity_curve"}})
    pd.DataFrame(rows).sort_values("total_return_pct", ascending=False).to_csv(OUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows)[["variant", "total_return_pct", "max_drawdown_pct", "num_buys"]].sort_values("total_return_pct", ascending=False).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yfinance as yf

from src.stock_signal import resolve_default_discord_webhook_url, send_discord_webhook

METHOD_LABELS = {
    "condition2": "上昇時メソッド",
    "breakout_1.5": "安定局面メソッド",
    "q3_post_high_vol": "反発局面メソッド",
    "no_trade": "no_trade",
    "post_major_multi_etf_entry": "crash時ETFメソッド",
    "post_major_prev_high_break_entry": "暴落後全体上昇時メソッド",
    "post_major_multi_etf": "crash時ETFメソッド",
    "post_major_stock_prev_high_break": "暴落後全体上昇時メソッド",
    "no_trade_runner_hard_detached": "no_trade短期追随メソッド",
    "no_trade_runner_hard_detached_entry": "no_trade短期追随メソッド",
    "crash_late_runner_hard_detached_entry": "crash後半短期追随メソッド",
    "crash_late_runner_theme_or_hard_entry": "crash後半テーマ/需給追随メソッド",
    "v38_post_crash_concentrated_etf_entry": "v3.8暴落後ETF集中メソッド",
    "v38_post_crash_dispersed_prev_high_entry": "v3.8暴落後分散個別株メソッド",
}

REGIME_LABELS = {
    "up": "上昇局面",
    "sideways": "安定局面",
    "down": "下降局面",
    "stable": "安定局面",
    "uptrend": "上昇局面",
    "downtrend": "下降局面",
    "crash": "暴落局面",
    "high_vol": "高ボラ局面",
    "capitulation_end": "投げ売り終盤局面",
    "settling": "落ち着き始め局面",
    "normal": "通常局面",
    "reversal_up": "上方反転局面",
    "reversal_down": "下方反転局面",
    "overheated_range": "過熱持ち合い局面",
    "weak_uptrend": "弱い上昇局面",
    "raw_post_crash_high_vol": "暴落直後高ボラ局面",
    "rebound_confirmed_post_crash_high_vol": "暴落後反発確認済み高ボラ局面",
    "generic_high_vol": "一般高ボラ局面",
    "post_major_crash": "大暴落後回復モード",
    "post_crash_stable": "大暴落後・安定局面",
    "post_crash_uptrend": "大暴落後・上昇局面",
    "post_crash_downtrend": "大暴落後・下降局面",
    "post_crash_high_vol": "大暴落後・高ボラ局面",
    "post_crash_capitulation_end": "大暴落後・投げ売り終盤局面",
    "post_crash_settling": "大暴落後・落ち着き始め局面",
    "post_crash_normal": "大暴落後・通常風局面",
    "post_crash_reversal_up": "大暴落後・上方反転局面",
    "post_crash_reversal_down": "大暴落後・下方反転局面",
    "post_crash_overheated_range": "大暴落後・過熱持ち合い局面",
    "post_crash_weak_uptrend": "大暴落後・弱い上昇局面",
    "post_crash_raw_post_crash_high_vol": "大暴落後・直後高ボラ局面",
    "post_crash_surge": "大暴落後・急騰局面",
}

BUY_RE = re.compile(
    r"^\[BUY\]\s+(?P<symbol>[0-9A-Z]+\.T)\s+\|\s+.*?close=(?P<close>[\d.]+)\s+tp_prob=(?P<tp>[\d.]+)%.*?lmt=(?P<lmt_ratio>[-\d.]+)%\s+lmt_price=(?P<lmt>[\d.]+)(?:\s+tp=(?P<tp_ratio>[\d.]+)%\s+tp_price=(?P<tp_price>[\d.]+))?\s+sl=(?P<sl_ratio>[\d.]+)%\s+sl_price=(?P<sl>[\d.]+)"
)
HOLD_RE = re.compile(
    r"^\[(?P<kind>HOLD)\]\s+(?P<symbol>[0-9A-Z]+\.T)\s+\|\s+.*?close=(?P<close>[\d.]+)\s+tp_prob=(?P<tp>[\d.]+)%.*?lmt=(?P<lmt_ratio>[-\d.]+)%\s+lmt_price=(?P<lmt>[\d.]+)(?:\s+tp=(?P<tp_ratio>[\d.]+)%\s+tp_price=(?P<tp_price>[\d.]+))?\s+sl=(?P<sl_ratio>[\d.]+)%"
)
PICK_RE = re.compile(
    r"^\[PICK\]\[(?P<group>[^\]]+)\]\s+(?P<symbol>[0-9A-Z]+\.T)\s+tp_prob=(?P<tp>[\d.]+)%\s+sector=(?P<sector>.+?)(?:\s+method=(?P<method>[A-Za-z0-9_.-]+))?(?:\s+company=(?P<company>.+))?$"
)
META_RE = re.compile(
    r"^\[META\]\s+regime=(?P<regime>[a-zA-Z_]+)\s+method=(?P<method>[A-Za-z0-9_.-]+)\s+signal_date=(?P<signal_date>\d{4}-\d{2}-\d{2})\s+trade_date=(?P<trade_date>\d{4}-\d{2}-\d{2})(?:\s+ssa_confirm_prior=(?P<ssa_confirm_prior>True|False))?(?:\s+ssa_available_prior=(?P<ssa_available_prior>True|False))?(?:\s+crash_late_active=(?P<crash_late_active>True|False))?(?:\s+crash_pos=(?P<crash_pos>\d*))?(?:\s+post_major_crash_mode=(?P<post_major_crash_mode>True|False))?(?:\s+post_major_phase=(?P<post_major_phase>[A-Za-z0-9_]+))?(?:\s+sector_mode=(?P<sector_mode>[A-Za-z0-9_]+))?(?:\s+effective_regime=(?P<effective_regime>[A-Za-z0-9_]+))?$"
)


def fetch_symbol_name(symbol: str) -> str:
    try:
        info = yf.Ticker(symbol).get_info()
        if isinstance(info, dict):
            name = info.get("shortName") or info.get("longName") or info.get("displayName")
            if name:
                return str(name).strip()
    except Exception:
        pass
    return symbol.removesuffix(".T")


def display_method_name(method_name: str | None) -> str:
    if not method_name:
        return ""
    raw = method_name.removesuffix("_entry")
    return METHOD_LABELS.get(raw, METHOD_LABELS.get(method_name, method_name))


def strategy_bucket(method_name: str | None) -> str:
    if not method_name:
        return "主戦略"
    raw = method_name.removesuffix("_entry")
    if raw == "no_trade_runner_hard_detached":
        return "サブ戦略"
    if raw == "crash_late_runner_hard_detached":
        return "サブ戦略"
    if raw == "crash_late_runner_theme_or_hard":
        return "サブ戦略"
    if raw.startswith("v38_post_crash_"):
        return "サブ戦略"
    return "主戦略"


def display_regime_name(regime_name: str | None) -> str:
    if not regime_name:
        return ""
    return REGIME_LABELS.get(regime_name, regime_name)


def meta_regime_name(meta: dict) -> str:
    return meta.get("effective_regime") or meta.get("regime") or ""


def build_run_url() -> str:
    server = os.getenv("GITHUB_SERVER_URL", "https://github.com")
    repo = os.getenv("GITHUB_REPOSITORY", "")
    run_id = os.getenv("GITHUB_RUN_ID", "")
    if repo and run_id:
        return f"{server}/{repo}/actions/runs/{run_id}"
    return ""


def notify_failure(webhook_url: str) -> int:
    workflow = os.getenv("GITHUB_WORKFLOW", "stock-signal-runner")
    ref = os.getenv("GITHUB_REF_NAME", "")
    actor = os.getenv("GITHUB_ACTOR", "")
    run_url = build_run_url()
    lines = [
        f"[FAIL] {workflow}",
        f"ref={ref} actor={actor}",
    ]
    if run_url:
        lines.append(run_url)
    send_discord_webhook(webhook_url, "\n".join(lines))
    print("[DISCORD] failure notification sent.")
    return 0


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"symbols": {}, "updated_at": ""}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"symbols": {}, "updated_at": ""}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_signal_lines(log_path: Path) -> tuple[list[dict], list[dict], dict[str, dict], dict]:
    buy_rows: list[dict] = []
    pick_rows: list[dict] = []
    detail_rows: dict[str, dict] = {}
    meta: dict = {}
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        meta_match = META_RE.match(line)
        if meta_match:
            meta = meta_match.groupdict()
            continue
        buy_match = BUY_RE.match(line)
        if buy_match:
            row = {
                "symbol": buy_match.group("symbol"),
                "tp_prob": float(buy_match.group("tp")),
                "close_price": float(buy_match.group("close")),
                "entry_price": float(buy_match.group("lmt")),
                "entry_ratio_pct": float(buy_match.group("lmt_ratio")),
                "take_profit_ratio_pct": float(buy_match.group("tp_ratio")) if buy_match.group("tp_ratio") else None,
                "take_profit_price": float(buy_match.group("tp_price")) if buy_match.group("tp_price") else None,
                "stop_loss_ratio_pct": float(buy_match.group("sl_ratio")),
                "stop_loss_price": float(buy_match.group("sl")),
                "kind": "BUY",
            }
            buy_rows.append(row)
            detail_rows[row["symbol"]] = row
            continue
        hold_match = HOLD_RE.match(line)
        if hold_match:
            detail_rows[hold_match.group("symbol")] = {
                "symbol": hold_match.group("symbol"),
                "tp_prob": float(hold_match.group("tp")),
                "close_price": float(hold_match.group("close")),
                "entry_price": float(hold_match.group("lmt")),
                "entry_ratio_pct": float(hold_match.group("lmt_ratio")),
                "take_profit_ratio_pct": float(hold_match.group("tp_ratio")) if hold_match.group("tp_ratio") else None,
                "take_profit_price": float(hold_match.group("tp_price")) if hold_match.group("tp_price") else None,
                "stop_loss_ratio_pct": float(hold_match.group("sl_ratio")),
                "stop_loss_price": None,
                "kind": "HOLD",
            }
            continue
        pick_match = PICK_RE.match(line)
        if pick_match:
            pick_rows.append(
                {
                    "group": pick_match.group("group"),
                    "symbol": pick_match.group("symbol"),
                    "tp_prob": float(pick_match.group("tp")),
                    "sector": pick_match.group("sector").strip(),
                    "method": (pick_match.group("method") or "").strip(),
                    "company": (pick_match.group("company") or "").strip(),
                }
            )
    return buy_rows, pick_rows, detail_rows, meta


def format_buy_line(symbol: str, company_name: str, sector: str, detail: dict, take_profit_ratio: float, method_name: str | None = None) -> str:
    entry_price = detail.get("entry_price")
    close_price = detail.get("close_price")
    take_profit_price = detail.get("take_profit_price")
    stop_loss_price = detail.get("stop_loss_price")
    tp_prob = detail.get("tp_prob")
    if isinstance(entry_price, (int, float)):
        entry_text = f"{entry_price:,.2f}円"
    else:
        entry_text = "N/A"
    if not isinstance(take_profit_price, (int, float)) and isinstance(entry_price, (int, float)):
        take_profit_price = entry_price * (1.0 + take_profit_ratio)
    tp_text_price = f"{take_profit_price:,.2f}円" if isinstance(take_profit_price, (int, float)) else "N/A"
    if stop_loss_price is None and isinstance(close_price, (int, float)):
        sl_ratio = float(detail.get("stop_loss_ratio_pct", 0.0)) / 100.0
        stop_loss_price = close_price * (1.0 - sl_ratio)
    sl_text_price = f"{stop_loss_price:,.2f}円" if isinstance(stop_loss_price, (int, float)) else "N/A"
    tp_text = f"{tp_prob:.2f}%" if isinstance(tp_prob, (int, float)) else "N/A"
    method_text = f"、手法{display_method_name(method_name)}" if method_name else ""
    bucket_text = strategy_bucket(method_name)
    kind_text = "継続" if detail.get("kind") == "HOLD" else "買い条件"
    return (
        f"[{bucket_text}] {symbol.removesuffix('.T')}[{company_name}] セクター{sector}："
        f"{kind_text}トリガー{entry_text}、利確{tp_text_price}、損切{sl_text_price}、上昇シグナル{tp_text}{method_text}"
    )


def format_sell_line(symbol: str, state_row: dict) -> str:
    company_name = state_row.get("company_name", symbol.removesuffix(".T"))
    sector = state_row.get("sector", "UNKNOWN")
    bucket_text = strategy_bucket(state_row.get("method"))
    return f"[{bucket_text}] {symbol.removesuffix('.T')}[{company_name}] セクター{sector}：早期利確（購入シグナル消失）"


def notify_signal_from_log(webhook_url: str, log_path: Path, max_lines: int, state_path: Path, take_profit_ratio: float) -> int:
    if not log_path.exists():
        return 0
    buy_rows, pick_rows, detail_rows, meta = parse_signal_lines(log_path)
    workflow = os.getenv("GITHUB_WORKFLOW", "stock-signal-runner")
    run_url = build_run_url()
    previous_state = load_state(state_path)
    previous_symbols = set((previous_state.get("symbols") or {}).keys())
    chosen_symbols = [row["symbol"] for row in pick_rows] if pick_rows else [row["symbol"] for row in buy_rows]
    pick_map = {row["symbol"]: row for row in pick_rows}
    current_state_symbols: dict[str, dict] = {}
    for symbol in chosen_symbols:
        pick = pick_map.get(symbol, {})
        detail = detail_rows.get(symbol, {})
        company_name = pick.get("company") or fetch_symbol_name(symbol)
        sector = str(pick.get("sector", "UNKNOWN"))
        current_state_symbols[symbol] = {
            "company_name": company_name,
            "sector": sector,
            "method": pick.get("method") or meta.get("method", ""),
            "kind": "HOLD" if symbol in previous_symbols else "BUY",
            "tp_prob": pick.get("tp_prob", detail.get("tp_prob")),
            "entry_price": detail.get("entry_price"),
            "entry_ratio_pct": detail.get("entry_ratio_pct"),
            "stop_loss_ratio_pct": detail.get("stop_loss_ratio_pct"),
            "stop_loss_price": detail.get("stop_loss_price"),
            "close_price": detail.get("close_price"),
        }

    disappeared_symbols = sorted(previous_symbols - set(current_state_symbols.keys()))
    main_symbols = [
        s for s in chosen_symbols
        if strategy_bucket(current_state_symbols[s].get("method") or meta.get("method", "")) == "主戦略"
    ]
    sub_symbols = [
        s for s in chosen_symbols
        if strategy_bucket(current_state_symbols[s].get("method") or meta.get("method", "")) == "サブ戦略"
    ]

    lines = []
    header_suffix = ""
    if meta.get("method"):
        header_suffix += f" method={display_method_name(meta['method'])}"
    display_regime = meta_regime_name(meta)
    if display_regime:
        header_suffix += f" regime={display_regime_name(display_regime)}"
    if chosen_symbols:
        buckets = sorted({strategy_bucket(current_state_symbols[s].get("method") or meta.get("method", "")) for s in chosen_symbols})
        lines.extend([f"[SIGNAL] {workflow}{header_suffix}", f"count={len(chosen_symbols)} strategy={','.join(buckets)}"])
        if meta.get("signal_date") or meta.get("trade_date"):
            lines.append(
                f"signal_date={meta.get('signal_date', 'N/A')} trade_date={meta.get('trade_date', 'N/A')}"
            )
        if display_regime:
            lines.append(f"翌営業日の日経トレンド予測：{display_regime_name(display_regime)}")
        if meta.get("post_major_crash_mode") is not None:
            lines.append(
                "crash_mode: "
                f"{'ON' if str(meta.get('post_major_crash_mode')) == 'True' else 'OFF'}"
                f" / base_phase={display_regime_name(meta.get('post_major_phase')) if meta.get('post_major_phase') not in (None, 'none') else 'N/A'}"
                f" / sector_mode={meta.get('sector_mode', 'N/A')}"
            )
        if meta.get("ssa_confirm_prior") is not None:
            lines.append(f"SSA回復確認: {'ON' if str(meta.get('ssa_confirm_prior')) == 'True' else 'OFF'}")
        lines.append(f"主戦略シグナル: {'あり' if main_symbols else 'なし'} ({len(main_symbols)}件)")
        lines.append(f"サブ戦略シグナル: {'あり' if sub_symbols else 'なし'} ({len(sub_symbols)}件)")
        for symbol in chosen_symbols[:max_lines]:
            state_row = current_state_symbols[symbol]
            lines.append(
                format_buy_line(
                    symbol=symbol,
                    company_name=state_row["company_name"],
                    sector=state_row["sector"],
                    detail=state_row,
                    take_profit_ratio=take_profit_ratio,
                    method_name=state_row.get("method") or meta.get("method", ""),
                )
            )
        if len(chosen_symbols) > max_lines:
            lines.append(f"... and {len(chosen_symbols) - max_lines} more")
    else:
        lines.extend([f"[NO_SIGNAL] {workflow}{header_suffix}", "シグナル無し"])
        if meta.get("signal_date") or meta.get("trade_date"):
            lines.append(
                f"signal_date={meta.get('signal_date', 'N/A')} trade_date={meta.get('trade_date', 'N/A')}"
            )
        if display_regime:
            lines.append(f"翌営業日の日経トレンド予測：{display_regime_name(display_regime)}")
        if meta.get("post_major_crash_mode") is not None:
            lines.append(
                "crash_mode: "
                f"{'ON' if str(meta.get('post_major_crash_mode')) == 'True' else 'OFF'}"
                f" / base_phase={display_regime_name(meta.get('post_major_phase')) if meta.get('post_major_phase') not in (None, 'none') else 'N/A'}"
                f" / sector_mode={meta.get('sector_mode', 'N/A')}"
            )
        lines.append("主戦略シグナル: なし (0件)")
        lines.append("サブ戦略シグナル: なし (0件)")

    if disappeared_symbols:
        lines.append("")
        lines.append(f"[BUY_SIGNAL_LOST] count={len(disappeared_symbols)}")
        for symbol in disappeared_symbols[:max_lines]:
            lines.append(format_sell_line(symbol, (previous_state.get("symbols") or {}).get(symbol, {})))
        if len(disappeared_symbols) > max_lines:
            lines.append(f"... and {len(disappeared_symbols) - max_lines} more")

    if run_url:
        lines.append(run_url)
    send_discord_webhook(webhook_url, "\n".join(lines))
    kind = "signal" if chosen_symbols else "no_signal"
    print(f"[DISCORD] {kind} notification sent. count={len(chosen_symbols)} lost={len(disappeared_symbols)}")
    save_state(
        state_path,
        {
            "updated_at": os.getenv("GITHUB_RUN_ID", ""),
            "symbols": current_state_symbols,
        },
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Send workflow notifications to Discord.")
    parser.add_argument("--mode", choices=["failure", "signal-log"], required=True)
    parser.add_argument("--webhook-url", default=resolve_default_discord_webhook_url())
    parser.add_argument("--log-path", type=Path, default=Path("data/last_run_github.log"))
    parser.add_argument("--max-lines", type=int, default=12)
    parser.add_argument("--state-path", type=Path, default=Path("data/last_signal_state_v38.json"))
    parser.add_argument("--take-profit-ratio", type=float, default=0.05)
    args = parser.parse_args()

    if not args.webhook_url:
        print("Discord webhook is not configured; skip notification.")
        return 0

    if args.mode == "failure":
        return notify_failure(args.webhook_url)
    return notify_signal_from_log(args.webhook_url, args.log_path, args.max_lines, args.state_path, args.take_profit_ratio)


if __name__ == "__main__":
    raise SystemExit(main())

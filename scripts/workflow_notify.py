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
}

REGIME_LABELS = {
    "up": "上昇局面",
    "sideways": "安定局面",
    "down": "下降局面",
    "crash": "暴落局面",
    "high_vol": "高ボラ局面",
    "capitulation_end": "投げ売り終盤局面",
    "settling": "落ち着き始め局面",
    "normal": "通常局面",
    "post_major_crash": "大暴落後回復モード",
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
    r"^\[META\]\s+regime=(?P<regime>[a-zA-Z_]+)\s+method=(?P<method>[A-Za-z0-9_.-]+)\s+signal_date=(?P<signal_date>\d{4}-\d{2}-\d{2})\s+trade_date=(?P<trade_date>\d{4}-\d{2}-\d{2})$"
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


def display_regime_name(regime_name: str | None) -> str:
    if not regime_name:
        return ""
    return REGIME_LABELS.get(regime_name, regime_name)


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
    return (
        f"{symbol.removesuffix('.T')}[{company_name}] セクター{sector}："
        f"逆指値{entry_text}、利確{tp_text_price}、損切{sl_text_price}、上昇シグナル{tp_text}{method_text}"
    )


def format_sell_line(symbol: str, state_row: dict) -> str:
    company_name = state_row.get("company_name", symbol.removesuffix(".T"))
    sector = state_row.get("sector", "UNKNOWN")
    return f"{symbol.removesuffix('.T')}[{company_name}] セクター{sector}：早期利確（購入シグナル消失）"


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
            "tp_prob": pick.get("tp_prob", detail.get("tp_prob")),
            "entry_price": detail.get("entry_price"),
            "entry_ratio_pct": detail.get("entry_ratio_pct"),
            "stop_loss_ratio_pct": detail.get("stop_loss_ratio_pct"),
            "stop_loss_price": detail.get("stop_loss_price"),
            "close_price": detail.get("close_price"),
        }

    disappeared_symbols = sorted(previous_symbols - set(current_state_symbols.keys()))

    lines = []
    header_suffix = ""
    if meta.get("method"):
        header_suffix += f" method={display_method_name(meta['method'])}"
    if meta.get("regime"):
        header_suffix += f" regime={display_regime_name(meta['regime'])}"
    if chosen_symbols:
        lines.extend([f"[SIGNAL] {workflow}{header_suffix}", f"count={len(chosen_symbols)}"])
        if meta.get("signal_date") or meta.get("trade_date"):
            lines.append(
                f"signal_date={meta.get('signal_date', 'N/A')} trade_date={meta.get('trade_date', 'N/A')}"
            )
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
    parser.add_argument("--state-path", type=Path, default=Path("data/last_signal_state_4q2.json"))
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

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yfinance as yf

from src.stock_signal import resolve_default_discord_webhook_url, send_discord_webhook

BUY_RE = re.compile(
    r"^\[BUY\]\s+(?P<symbol>[0-9A-Z]+\.T)\s+\|\s+.*?tp_prob=(?P<tp>[\d.]+)%.*?lmt_price=(?P<lmt>[\d.]+)"
)
PICK_RE = re.compile(
    r"^\[PICK\]\[(?P<group>[^\]]+)\]\s+(?P<symbol>[0-9A-Z]+\.T)\s+tp_prob=(?P<tp>[\d.]+)%\s+sector=(?P<sector>.+)$"
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


def parse_signal_lines(log_path: Path) -> tuple[list[dict], list[dict]]:
    buy_rows: list[dict] = []
    pick_rows: list[dict] = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        buy_match = BUY_RE.match(line)
        if buy_match:
            buy_rows.append(
                {
                    "symbol": buy_match.group("symbol"),
                    "tp_prob": float(buy_match.group("tp")),
                    "lmt_price": float(buy_match.group("lmt")),
                }
            )
            continue
        pick_match = PICK_RE.match(line)
        if pick_match:
            pick_rows.append(
                {
                    "group": pick_match.group("group"),
                    "symbol": pick_match.group("symbol"),
                    "tp_prob": float(pick_match.group("tp")),
                    "sector": pick_match.group("sector").strip(),
                }
            )
    return buy_rows, pick_rows


def notify_signal_from_log(webhook_url: str, log_path: Path, max_lines: int) -> int:
    if not log_path.exists():
        return 0
    buy_rows, pick_rows = parse_signal_lines(log_path)
    workflow = os.getenv("GITHUB_WORKFLOW", "stock-signal-runner")
    run_url = build_run_url()
    if not buy_rows and not pick_rows:
        lines = [f"[NO_SIGNAL] {workflow}", "シグナル無し"]
        if run_url:
            lines.append(run_url)
        send_discord_webhook(webhook_url, "\n".join(lines))
        return 0

    buy_map = {row["symbol"]: row for row in buy_rows}
    chosen_symbols = [row["symbol"] for row in pick_rows] if pick_rows else [row["symbol"] for row in buy_rows]
    pick_map = {row["symbol"]: row for row in pick_rows}

    lines = [f"[SIGNAL] {workflow}", f"count={len(chosen_symbols)}"]
    for symbol in chosen_symbols[:max_lines]:
        pick = pick_map.get(symbol, {})
        buy = buy_map.get(symbol, {})
        company_name = fetch_symbol_name(symbol)
        sector = str(pick.get("sector", "UNKNOWN"))
        lmt_price = buy.get("lmt_price")
        tp_prob = pick.get("tp_prob", buy.get("tp_prob"))
        lmt_text = f"{lmt_price:,.2f}円" if isinstance(lmt_price, (int, float)) else "N/A"
        tp_text = f"{tp_prob:.2f}%" if isinstance(tp_prob, (int, float)) else "N/A"
        lines.append(f"{symbol.removesuffix('.T')}[{company_name}] セクター{sector}：逆指値{lmt_text}、上昇シグナル{tp_text}")
    if len(chosen_symbols) > max_lines:
        lines.append(f"... and {len(chosen_symbols) - max_lines} more")
    if run_url:
        lines.append(run_url)
    send_discord_webhook(webhook_url, "\n".join(lines))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Send workflow notifications to Discord.")
    parser.add_argument("--mode", choices=["failure", "signal-log"], required=True)
    parser.add_argument("--webhook-url", default=resolve_default_discord_webhook_url())
    parser.add_argument("--log-path", type=Path, default=Path("data/last_run_github.log"))
    parser.add_argument("--max-lines", type=int, default=12)
    args = parser.parse_args()

    if not args.webhook_url:
        print("Discord webhook is not configured; skip notification.")
        return 0

    if args.mode == "failure":
        return notify_failure(args.webhook_url)
    return notify_signal_from_log(args.webhook_url, args.log_path, args.max_lines)


if __name__ == "__main__":
    raise SystemExit(main())

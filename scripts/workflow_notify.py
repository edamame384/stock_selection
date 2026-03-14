from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stock_signal import resolve_default_discord_webhook_url, send_discord_webhook


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


def notify_signal_from_log(webhook_url: str, log_path: Path, max_lines: int) -> int:
    if not log_path.exists():
        return 0
    buy_lines: list[str] = []
    pick_lines: list[str] = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "[PICK]" in line:
            pick_lines.append(line.strip())
        elif "[BUY]" in line:
            buy_lines.append(line.strip())
    chosen = pick_lines or buy_lines
    if not chosen:
        return 0
    workflow = os.getenv("GITHUB_WORKFLOW", "stock-signal-runner")
    run_url = build_run_url()
    lines = [f"[SIGNAL] {workflow}", f"count={len(chosen)}"]
    lines.extend(chosen[:max_lines])
    if len(chosen) > max_lines:
        lines.append(f"... and {len(chosen) - max_lines} more")
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

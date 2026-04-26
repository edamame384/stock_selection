from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.shikiho_text_parser.backtest_phase_adaptive_practical_v35_postcrash_sector_alloc_candidate import run_q2_postcrash_sector_alloc_candidate

OUT_DIR = ROOT / "projects" / "shikiho_text_parser" / "output" / "phase_adaptive_practical_v35_postcrash_sector_alloc_q2_candidate"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = run_q2_postcrash_sector_alloc_candidate()
    payload = {k: v for k, v in result.items() if k not in {"trade_log", "equity_curve"}}
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    result["trade_log"].to_csv(OUT_DIR / "trade_log.csv", index=False, encoding="utf-8-sig")
    result["equity_curve"].to_csv(OUT_DIR / "equity_curve.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

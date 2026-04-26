from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.shikiho_text_parser.backtest_phase_adaptive_practical_v31_pre1_post5_gold_switch import (  # noqa: E402
    EARNINGS_CACHE_CSV,
    IRBANK_EARNINGS_CACHE_CSV,
    PARSED_SUMMARY_CSV,
    RAW_4Q2_DIR,
    TableSpec,
    batch_specs,
    build_earnings_map,
    load_phase_map,
    load_local_disclosure_dates,
    run_dataset,
)
from projects.shikiho_text_parser.search_phase_method_optimization import PHASE_CSV  # noqa: E402


OUT_DIR = ROOT / "projects" / "shikiho_text_parser" / "output" / "post_crash_handover_compare"
ETF_TICKER = "1489.T"
THRESHOLDS = [1, 3, 5, 7, 10]


def build_cached_earnings_map(specs: list[TableSpec]) -> dict[str, list[pd.Timestamp]]:
    all_tickers: list[str] = []
    for spec in specs:
        detail = pd.read_csv(spec.detail_csv)
        selected = pd.read_csv(spec.selected_csv)
        all_tickers.extend(detail["ticker"].dropna().astype(str).tolist())
        all_tickers.extend(selected["ticker"].dropna().astype(str).tolist())
    all_tickers = sorted(set(all_tickers))

    local_df = load_local_disclosure_dates(all_tickers)

    frames = [local_df]
    if IRBANK_EARNINGS_CACHE_CSV.exists():
        irbank_df = pd.read_csv(IRBANK_EARNINGS_CACHE_CSV)
        irbank_df["earnings_date"] = pd.to_datetime(irbank_df["earnings_date"], errors="coerce")
        irbank_df = irbank_df.dropna(subset=["earnings_date"])
        frames.append(irbank_df[["ticker", "earnings_date"]])
    if EARNINGS_CACHE_CSV.exists():
        yf_df = pd.read_csv(EARNINGS_CACHE_CSV)
        yf_df["earnings_date"] = pd.to_datetime(yf_df["earnings_date"], errors="coerce")
        yf_df = yf_df.dropna(subset=["earnings_date"])
        frames.append(yf_df[["ticker", "earnings_date"]])

    if PARSED_SUMMARY_CSV.exists():
        try:
            parsed = pd.read_csv(PARSED_SUMMARY_CSV, usecols=["ticker_code", "disclosure_date"]).dropna()
            parsed["ticker"] = parsed["ticker_code"].astype(str).str.strip() + ".T"
            parsed["earnings_date"] = pd.to_datetime(parsed["disclosure_date"], errors="coerce")
            parsed = parsed.dropna(subset=["earnings_date"])
            frames.append(parsed[["ticker", "earnings_date"]])
        except Exception:
            pass

    if RAW_4Q2_DIR.exists():
        rows: list[dict[str, object]] = []
        for path in RAW_4Q2_DIR.glob("*.txt"):
            ticker = f"{path.stem}.T"
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for line in text.splitlines():
                if "直近決算発表日" not in line:
                    continue
                dt = pd.to_datetime(line.split("：", 1)[-1].replace("NEW!", "").strip(), errors="coerce")
                if pd.notna(dt):
                    rows.append({"ticker": ticker, "earnings_date": dt})
                break
        if rows:
            frames.append(pd.DataFrame(rows))

    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["ticker", "earnings_date"])
    if not merged.empty:
        merged = merged.dropna(subset=["ticker", "earnings_date"])
        merged["ticker"] = merged["ticker"].astype(str)
        merged["earnings_date"] = pd.to_datetime(merged["earnings_date"], errors="coerce")
        merged = merged.dropna(subset=["earnings_date"]).drop_duplicates(subset=["ticker", "earnings_date"]).sort_values(["ticker", "earnings_date"])
    return build_earnings_map(merged)


def run_variant(
    specs: list[TableSpec],
    phase_map: pd.Series,
    earnings_map: dict[str, list[pd.Timestamp]],
    threshold: int | None,
) -> list[dict]:
    rows: list[dict] = []
    for spec in specs:
        gold_phases = set() if threshold is None else {"high_vol"}
        result = run_dataset(
            spec=spec,
            phase_map=phase_map,
            earnings_map=earnings_map,
            earnings_pre_days=1,
            earnings_post_days=5,
            gold_ticker=ETF_TICKER,
            gold_phases=gold_phases,
            gold_allocation_ratio=1.0,
            gold_signal_name="rebound_open",
            gold_require_recent_phase="crash" if threshold is not None else "",
            gold_require_recent_phase_days=threshold or 0,
        )
        rows.append(
            {
                "variant": "baseline_q3_post_high_vol" if threshold is None else f"etf_until_{threshold}d_then_q3_post_high_vol",
                "threshold_days": threshold if threshold is not None else -1,
                "dataset": spec.name,
                "total_return_pct": result["total_return_pct"],
                "final_capital": result["final_capital"],
                "num_buys": result["num_buys"],
                "max_drawdown_pct": result["max_drawdown_pct"],
                "phase_pnl": json.dumps(result["phase_pnl"], ensure_ascii=False),
            }
        )
    return rows


def main() -> int:
    phase_map = load_phase_map(PHASE_CSV)
    specs = [spec for spec in batch_specs() if spec.name in {"q2_2024", "4q2"}]
    earnings_map = build_cached_earnings_map(specs)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    rows.extend(run_variant(specs, phase_map, earnings_map, None))
    for threshold in THRESHOLDS:
        rows.extend(run_variant(specs, phase_map, earnings_map, threshold))

    detail_df = pd.DataFrame(rows)
    detail_df.to_csv(OUT_DIR / "detail.csv", index=False, encoding="utf-8-sig")

    summary_rows: list[dict] = []
    for variant, g in detail_df.groupby("variant", sort=False):
        total_final = float(g["final_capital"].sum())
        total_initial = 3_000_000.0 * len(g)
        summary_rows.append(
            {
                "variant": variant,
                "threshold_days": int(g["threshold_days"].iloc[0]),
                "total_return_pct": (total_final / total_initial - 1.0) * 100.0,
                "total_final_capital": total_final,
                "q2_2024_return_pct": float(g.loc[g["dataset"] == "q2_2024", "total_return_pct"].iloc[0]),
                "q3_return_pct": float(g.loc[g["dataset"] == "q3", "total_return_pct"].iloc[0]) if (g["dataset"] == "q3").any() else None,
                "q4_return_pct": float(g.loc[g["dataset"] == "q4", "total_return_pct"].iloc[0]) if (g["dataset"] == "q4").any() else None,
                "4q2_return_pct": float(g.loc[g["dataset"] == "4q2", "total_return_pct"].iloc[0]),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(["threshold_days", "variant"]).reset_index(drop=True)
    summary_df.to_csv(OUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")

    baseline_total = float(summary_df.loc[summary_df["threshold_days"] == -1, "total_return_pct"].iloc[0])
    threshold_only = summary_df[summary_df["threshold_days"] >= 0].copy()
    best_row = threshold_only.sort_values("total_return_pct", ascending=False).iloc[0]
    threshold_only["delta_vs_baseline_pt"] = threshold_only["total_return_pct"] - baseline_total
    threshold_only.to_csv(OUT_DIR / "threshold_compare.csv", index=False, encoding="utf-8-sig")

    best_payload = {
        "baseline_variant": "baseline_q3_post_high_vol",
        "baseline_total_return_pct": baseline_total,
        "best_variant": str(best_row["variant"]),
        "best_threshold_days": int(best_row["threshold_days"]),
        "best_total_return_pct": float(best_row["total_return_pct"]),
        "best_delta_vs_baseline_pt": float(best_row["total_return_pct"] - baseline_total),
        "interpretation": "ETF rebound is superior until the best threshold; after that, q3_post_high_vol is better than continuing ETF use within post-crash high_vol.",
    }
    (OUT_DIR / "best_threshold.json").write_text(json.dumps(best_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

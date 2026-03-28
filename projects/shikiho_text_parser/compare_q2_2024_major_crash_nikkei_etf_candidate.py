from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.shikiho_text_parser import backtest_phase_adaptive_practical_v31_pre1_post5_gold_switch as etf_mod  # noqa: E402
from projects.shikiho_text_parser.backtest_phase_adaptive_practical_v31_pre1_post5_gold_switch import TableSpec  # noqa: E402
from projects.shikiho_text_parser.search_phase_method_optimization import PHASE_CSV  # noqa: E402


OUT_DIR = ROOT / "projects" / "shikiho_text_parser" / "output" / "q2_2024_major_crash_nikkei_etf_candidate"


def q2_spec() -> TableSpec:
    return TableSpec(
        "q2_2024",
        ROOT / "projects" / "quarterly_ranker" / "output" / "q2_2024_pre_analysis_20240630_aligned" / "q2_2024_pre_shikiho_feature_ranking.csv",
        ROOT / "projects" / "quarterly_ranker" / "output" / "q2_2024_pre_analysis_20240630_aligned" / "operational" / "q2_2024_pre_selected_candidates_operational.csv",
        "2024-07-01",
        "2024-09-30",
        3_000_000.0,
    )


def load_phase_frame() -> pd.DataFrame:
    df = pd.read_csv(PHASE_CSV, skiprows=[1])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.rename(columns={"phase": "phase_name"})
    for col in ["ret5", "dd20", "vol10"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["Date", "phase_name", "ret5", "dd20", "vol10"]].copy()


def classify_major_crash(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["major_crash"] = (out["phase_name"].astype(str) == "crash") & ((out["ret5"] <= -0.08) | (out["dd20"] <= -0.14))
    return out


def recent_prior_labels(df: pd.DataFrame, idx: int, lookback_days: int) -> list[str]:
    start = max(0, idx - lookback_days)
    return [str(x) for x in df.iloc[start:idx]["phase_name"].tolist()]


def transformed_phase_map(df: pd.DataFrame, mode: str, normal_lookback_days: int = 10) -> pd.Series:
    x = df.copy().reset_index(drop=True)
    x["phase_name_mod"] = x["phase_name"].astype(str)
    x.loc[x["major_crash"], "phase_name_mod"] = "downtrend"

    for idx in range(len(x)):
        current = str(x.loc[idx, "phase_name"])
        if current != "normal":
            continue
        prior_mod = [str(v) for v in x.loc[max(0, idx - normal_lookback_days): idx - 1, "phase_name_mod"].tolist()]
        prior_raw = [str(v) for v in x.loc[max(0, idx - normal_lookback_days): idx - 1, "phase_name"].tolist()]
        if "downtrend" in prior_mod and ("high_vol" in prior_raw or "capitulation_end" in prior_raw):
            x.loc[idx, "phase_name_mod"] = "downtrend"

    if mode == "stock_only_strict":
        return x.set_index("Date")["phase_name_mod"].sort_index()
    if mode == "nikkei1321_post_major_highvol":
        return x.set_index("Date")["phase_name_mod"].sort_index()
    raise ValueError(mode)


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
    return etf_mod.build_earnings_map(merged)


def run_mode(spec: TableSpec, phase_map: pd.Series, earnings_map: dict[str, list[pd.Timestamp]], mode: str) -> dict:
    if mode == "stock_only_strict":
        return etf_mod.run_dataset(
            spec,
            phase_map,
            earnings_map,
            earnings_pre_days=1,
            earnings_post_days=5,
            gold_ticker="1321.T",
            gold_phases=set(),
            gold_allocation_ratio=1.0,
            gold_signal_name="rebound_open",
            gold_require_recent_phase="downtrend",
            gold_require_recent_phase_days=10,
            phase_shift_days=1,
            gold_phase_shift_days=1,
        )
    if mode == "nikkei1321_post_major_highvol":
        return etf_mod.run_dataset(
            spec,
            phase_map,
            earnings_map,
            earnings_pre_days=1,
            earnings_post_days=5,
            gold_ticker="1321.T",
            gold_phases={"high_vol"},
            gold_allocation_ratio=1.0,
            gold_signal_name="rebound_open",
            gold_require_recent_phase="downtrend",
            gold_require_recent_phase_days=10,
            phase_shift_days=1,
            gold_phase_shift_days=1,
        )
    raise ValueError(mode)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    spec = q2_spec()
    earnings_map = prepare_earnings_map(spec)
    phase_df = classify_major_crash(load_phase_frame())

    rows: list[dict] = []
    for mode in ["stock_only_strict", "nikkei1321_post_major_highvol"]:
        phase_map = transformed_phase_map(phase_df, mode)
        result = run_mode(spec, phase_map, earnings_map, mode)
        out_dir = OUT_DIR / mode
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "mode": mode,
            "major_crash_definition": "phase=crash and (ret5<=-0.08 or dd20<=-0.14)",
            "major_crash_rule": "downtrend(no_trade)",
            "normal_after_major_crash_highvol_rule": "downtrend(no_trade)",
            "etf_ticker": "1321.T",
            "etf_activation": "high_vol with recent downtrend(major_crash transformed) within 10 days" if mode == "nikkei1321_post_major_highvol" else "disabled",
            **{k: v for k, v in result.items() if k not in {"trade_log", "equity_curve"}},
        }
        (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        result["trade_log"].to_csv(out_dir / "trade_log.csv", index=False, encoding="utf-8-sig")
        result["equity_curve"].to_csv(out_dir / "equity_curve.csv", index=False, encoding="utf-8-sig")
        rows.append(payload)

    pd.DataFrame(rows).to_csv(OUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    days = phase_df[phase_df["major_crash"]][["Date", "phase_name", "ret5", "dd20", "vol10"]].copy()
    days["Date"] = days["Date"].dt.strftime("%Y-%m-%d")
    days.to_csv(OUT_DIR / "major_crash_days.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows)[["mode", "total_return_pct", "num_buys", "max_drawdown_pct"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

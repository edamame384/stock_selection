from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_data(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=[0, 1])
    if "Date" in raw.columns.get_level_values(0):
        cols = []
        for top, bottom in raw.columns:
            if top == "Date":
                cols.append("Date")
            else:
                cols.append(top)
        raw.columns = cols
    else:
        raw = pd.read_csv(path)

    df = raw.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    df["close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["ret1"] = df["close"].pct_change()
    df["ret5"] = df["close"].pct_change(5)
    df["ret20"] = df["close"].pct_change(20)
    df["ret60"] = df["close"].pct_change(60)
    df["dd20"] = df["close"] / df["close"].rolling(20).max() - 1.0
    df["range20"] = df["close"].rolling(20).max() / df["close"].rolling(20).min() - 1.0
    df["vol10"] = df["ret1"].rolling(10).std()
    df["vol20"] = df["ret1"].rolling(20).std()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    df["ma20_slope_5"] = df["ma20"].pct_change(5)
    df["ma_gap_20_60"] = df["ma20"] / df["ma60"] - 1.0
    df["prior_surge_10d"] = df["ret5"].rolling(10, min_periods=1).max() >= 0.05
    df["prior_crash_10d"] = df["ret5"].rolling(10, min_periods=1).min() <= -0.05
    return df


def assign_phase(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    out = df.copy()
    vol_lo = float(out["vol10"].quantile(0.35))
    vol_mid = float(out["vol10"].quantile(0.55))
    vol_hi = float(out["vol10"].quantile(0.75))

    phase = pd.Series("normal", index=out.index, dtype="object")

    surge = (
        (out["vol10"] >= vol_hi)
        & ((out["ret5"] >= 0.05) | (out["ret20"] >= 0.10))
        & (out["dd20"] >= -0.03)
    )
    crash = (
        (out["vol10"] >= vol_hi)
        & ((out["ret5"] <= -0.05) | (out["dd20"] <= -0.08))
    )
    high_vol = (
        (out["vol10"] >= vol_hi)
        & ~surge
        & ~crash
        & (out["prior_surge_10d"] | out["prior_crash_10d"] | (out["dd20"] <= -0.03))
    )
    settling = (
        (out["vol10"] <= vol_mid)
        & (out["vol10"] >= vol_lo)
        & (out["ret5"].abs() <= 0.03)
        & ((out["prior_surge_10d"]) | (out["prior_crash_10d"]))
    )
    stable = (
        (out["vol10"] <= vol_lo)
        & (out["range20"] <= 0.08)
        & (out["ret20"].abs() <= 0.05)
        & (out["ma20_slope_5"].abs() <= 0.01)
    )
    uptrend = (
        (out["vol10"] <= vol_mid)
        & (out["ret20"] >= 0.03)
        & (out["ma_gap_20_60"] >= 0.0)
        & (out["ma20_slope_5"] > 0.0)
        & (out["dd20"] > -0.06)
    )
    downtrend = (
        (out["vol10"] <= vol_mid)
        & (out["ret20"] <= -0.03)
        & (out["ma_gap_20_60"] <= 0.0)
        & (out["ma20_slope_5"] < 0.0)
        & (out["dd20"] <= -0.03)
    )
    reversal_up = (
        (out["ret20"] < 0.0)
        & (out["ret5"] > 0.03)
        & (out["vol10"] <= vol_hi)
        & (out["ma20_slope_5"] > -0.005)
    )
    reversal_down = (
        (out["ret20"] > 0.0)
        & (out["ret5"] < -0.03)
        & (out["vol10"] <= vol_hi)
        & (out["ma20_slope_5"] < 0.005)
    )
    overheated_range = (
        (out["ret20"] >= 0.08)
        & (out["range20"] <= 0.10)
        & (out["vol10"] > vol_mid)
        & (out["vol10"] < vol_hi)
    )
    capitulation_end = (
        (out["dd20"] <= -0.10)
        & (out["ret5"] > 0.02)
        & (out["vol10"] >= vol_hi)
    )

    phase.loc[stable] = "stable"
    phase.loc[uptrend] = "uptrend"
    phase.loc[downtrend] = "downtrend"
    phase.loc[reversal_up] = "reversal_up"
    phase.loc[reversal_down] = "reversal_down"
    phase.loc[overheated_range] = "overheated_range"
    phase.loc[settling] = "settling"
    phase.loc[high_vol] = "high_vol"
    phase.loc[surge] = "surge"
    phase.loc[crash] = "crash"
    phase.loc[capitulation_end] = "capitulation_end"

    out["phase"] = phase
    meta = {"vol_lo": vol_lo, "vol_mid": vol_mid, "vol_hi": vol_hi}
    return out, meta


def summarize_periods(df: pd.DataFrame) -> pd.DataFrame:
    periods: list[dict] = []
    cur_phase = None
    start = None
    prev_date = None
    for dt, row in df.iterrows():
        phase = row["phase"]
        if cur_phase is None:
            cur_phase = phase
            start = dt
            prev_date = dt
            continue
        if phase == cur_phase:
            prev_date = dt
            continue
        seg = df.loc[start:prev_date]
        periods.append(
            {
                "phase": cur_phase,
                "start": start.date().isoformat(),
                "end": prev_date.date().isoformat(),
                "days": len(seg),
                "start_close": round(float(seg["close"].iloc[0]), 2),
                "end_close": round(float(seg["close"].iloc[-1]), 2),
                "return_pct": round((float(seg["close"].iloc[-1]) / float(seg["close"].iloc[0]) - 1.0) * 100.0, 2),
            }
        )
        cur_phase = phase
        start = dt
        prev_date = dt
    if cur_phase is not None:
        seg = df.loc[start:prev_date]
        periods.append(
            {
                "phase": cur_phase,
                "start": start.date().isoformat(),
                "end": prev_date.date().isoformat(),
                "days": len(seg),
                "start_close": round(float(seg["close"].iloc[0]), 2),
                "end_close": round(float(seg["close"].iloc[-1]), 2),
                "return_pct": round((float(seg["close"].iloc[-1]) / float(seg["close"].iloc[0]) - 1.0) * 100.0, 2),
            }
        )
    return pd.DataFrame(periods)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build candidate Nikkei phase map from arbitrary daily CSV.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(args.input_csv)
    labeled, meta = assign_phase(df)
    periods = summarize_periods(labeled)
    labeled.reset_index().to_csv(args.out_dir / "nikkei_market_phase_daily_labels.csv", index=False, encoding="utf-8-sig")
    periods.to_csv(args.out_dir / "nikkei_market_phase_periods.csv", index=False, encoding="utf-8-sig")
    (args.out_dir / "nikkei_market_phase_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": int(len(labeled)), "start": labeled.index.min().date().isoformat(), "end": labeled.index.max().date().isoformat()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

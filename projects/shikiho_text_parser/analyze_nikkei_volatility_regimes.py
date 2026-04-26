from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = ROOT / "data" / "external_market" / "nikkei225_daily.csv"
OUT_DIR = ROOT / "projects" / "shikiho_text_parser" / "output" / "nikkei_regime_phases"


def load_nikkei() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV, parse_dates=["Date"]).sort_values("Date").set_index("Date")
    df["close"] = df["Close"].astype(float)
    df["ret1"] = df["close"].pct_change()
    df["ret5"] = df["close"].pct_change(5)
    df["ret20"] = df["close"].pct_change(20)
    df["dd20"] = df["close"] / df["close"].rolling(20).max() - 1.0
    df["vol10"] = df["ret1"].rolling(10).std()
    return df


def assign_regimes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    vol_hi = float(df["vol10"].quantile(0.75))
    vol_lo = float(df["vol10"].quantile(0.55))

    prior_runup = df["ret20"].rolling(15, min_periods=1).max() >= 0.05
    stress_highvol = ((df["dd20"] <= -0.05) | (df["ret5"] <= -0.04)) & (df["vol10"] >= vol_hi)
    settle_ready = (df["vol10"] < vol_lo) & (df["ret5"].abs() <= 0.03)

    phase = pd.Series("normal", index=df.index, dtype="object")
    episode_id = pd.Series(pd.NA, index=df.index, dtype="object")

    event_counter = 0
    i = 0
    dates = df.index.tolist()
    while i < len(dates):
        dt = dates[i]
        if bool(stress_highvol.loc[dt]):
            start = i
            while i + 1 < len(dates) and bool(stress_highvol.loc[dates[i + 1]]):
                i += 1
            end = i
            block_dates = dates[start : end + 1]
            block_has_runup = bool(prior_runup.loc[block_dates].any())

            event_counter += 1
            eid = f"E{event_counter:03d}"
            for j, bdt in enumerate(block_dates):
                if block_has_runup and j < 3:
                    phase.loc[bdt] = "crash_initial"
                else:
                    phase.loc[bdt] = "high_vol_continuation"
                episode_id.loc[bdt] = eid

            settle_start = end + 1
            settle_end = settle_start - 1
            consec = 0
            k = settle_start
            while k < len(dates):
                kdt = dates[k]
                if bool(stress_highvol.loc[kdt]):
                    break
                if bool(settle_ready.loc[kdt]):
                    consec += 1
                else:
                    if consec > 0:
                        consec = 0
                        settle_start = k + 1
                if consec >= 3:
                    settle_end = k
                    m = settle_start
                    while m + 1 < len(dates) and not bool(stress_highvol.loc[dates[m + 1]]) and bool(settle_ready.loc[dates[m + 1]]):
                        m += 1
                    settle_end = m
                    for sdt in dates[settle_start : settle_end + 1]:
                        phase.loc[sdt] = "settling"
                        episode_id.loc[sdt] = eid
                    i = settle_end
                    break
                k += 1
            else:
                i = len(dates) - 1
        i += 1

    out = df.copy()
    out["prior_runup_15d"] = prior_runup
    out["stress_highvol"] = stress_highvol
    out["settle_ready"] = settle_ready
    out["phase"] = phase
    out["episode_id"] = episode_id
    meta = {"vol_hi": vol_hi, "vol_lo": vol_lo}
    return out, meta


def summarize_periods(labeled: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for episode_id, g in labeled.dropna(subset=["episode_id"]).groupby("episode_id", sort=True):
        for phase_name, p in g.groupby("phase", sort=False):
            rows.append(
                {
                    "episode_id": episode_id,
                    "phase": phase_name,
                    "start": p.index.min().date().isoformat(),
                    "end": p.index.max().date().isoformat(),
                    "days": len(p),
                    "start_close": round(float(p["close"].iloc[0]), 2),
                    "end_close": round(float(p["close"].iloc[-1]), 2),
                    "period_return_pct": round((float(p["close"].iloc[-1]) / float(p["close"].iloc[0]) - 1.0) * 100.0, 2),
                    "ret20_start_pct": round(float(p["ret20"].iloc[0]) * 100.0, 2) if pd.notna(p["ret20"].iloc[0]) else None,
                    "ret5_min_pct": round(float(p["ret5"].min()) * 100.0, 2),
                    "min_dd20_pct": round(float(p["dd20"].min()) * 100.0, 2),
                    "avg_vol10_pct": round(float(p["vol10"].mean()) * 100.0, 2),
                    "max_vol10_pct": round(float(p["vol10"].max()) * 100.0, 2),
                    "prior_runup_any": bool(p["prior_runup_15d"].any()),
                }
            )
    return pd.DataFrame(rows)


def summarize_examples(labeled: pd.DataFrame, start: str, end: str, label: str) -> pd.DataFrame:
    window = labeled.loc[pd.Timestamp(start) : pd.Timestamp(end)].copy()
    out = window[["close", "ret5", "ret20", "dd20", "vol10", "phase", "episode_id"]].reset_index()
    out.insert(0, "window_label", label)
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_nikkei()
    labeled, meta = assign_regimes(df)
    periods = summarize_periods(labeled)

    labeled.reset_index().to_csv(OUT_DIR / "nikkei_regime_daily_labels.csv", index=False, encoding="utf-8-sig")
    periods.to_csv(OUT_DIR / "nikkei_regime_periods.csv", index=False, encoding="utf-8-sig")

    examples = pd.concat(
        [
            summarize_examples(labeled, "2025-03-01", "2025-03-31", "2025-03"),
            summarize_examples(labeled, "2026-03-01", "2026-03-17", "2026-03"),
        ],
        axis=0,
        ignore_index=True,
    )
    examples.to_csv(OUT_DIR / "nikkei_regime_examples_202503_202603.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(
        [
            {"metric": "vol10_high_threshold_pct", "value": round(meta["vol_hi"] * 100.0, 3)},
            {"metric": "vol10_settle_threshold_pct", "value": round(meta["vol_lo"] * 100.0, 3)},
            {"metric": "episodes", "value": int(periods["episode_id"].nunique()) if not periods.empty else 0},
            {"metric": "crash_initial_periods", "value": int((periods["phase"] == "crash_initial").sum()) if not periods.empty else 0},
            {"metric": "high_vol_continuation_periods", "value": int((periods["phase"] == "high_vol_continuation").sum()) if not periods.empty else 0},
            {"metric": "settling_periods", "value": int((periods["phase"] == "settling").sum()) if not periods.empty else 0},
        ]
    )
    summary.to_csv(OUT_DIR / "nikkei_regime_summary.csv", index=False, encoding="utf-8-sig")

    print(summary.to_string(index=False))
    print("---")
    if not periods.empty:
        print(periods.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

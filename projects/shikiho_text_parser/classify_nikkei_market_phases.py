from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = ROOT / "data" / "external_market" / "nikkei225_daily.csv"
OUT_DIR = ROOT / "projects" / "shikiho_text_parser" / "output" / "nikkei_market_phase_map"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV, parse_dates=["Date"]).sort_values("Date").set_index("Date")
    df["close"] = df["Close"].astype(float)
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
                "ret5_min_pct": round(float(seg["ret5"].min()) * 100.0, 2) if seg["ret5"].notna().any() else None,
                "ret5_max_pct": round(float(seg["ret5"].max()) * 100.0, 2) if seg["ret5"].notna().any() else None,
                "dd20_min_pct": round(float(seg["dd20"].min()) * 100.0, 2) if seg["dd20"].notna().any() else None,
                "vol10_mean_pct": round(float(seg["vol10"].mean()) * 100.0, 2) if seg["vol10"].notna().any() else None,
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
                "ret5_min_pct": round(float(seg["ret5"].min()) * 100.0, 2) if seg["ret5"].notna().any() else None,
                "ret5_max_pct": round(float(seg["ret5"].max()) * 100.0, 2) if seg["ret5"].notna().any() else None,
                "dd20_min_pct": round(float(seg["dd20"].min()) * 100.0, 2) if seg["dd20"].notna().any() else None,
                "vol10_mean_pct": round(float(seg["vol10"].mean()) * 100.0, 2) if seg["vol10"].notna().any() else None,
            }
        )
    return pd.DataFrame(periods)


def build_method_map() -> pd.DataFrame:
    rows = [
        {"phase": "stable", "japanese_name": "安定期間", "recommended_method": "method_breakout_1p5", "recommended_alias": "横ばい順張り", "note": "レンジ内では厳しめ順張りでノイズを減らす"},
        {"phase": "uptrend", "japanese_name": "上昇期間", "recommended_method": "method_condition2", "recommended_alias": "上向き標準", "note": "広めに拾ってトレンド継続を取る"},
        {"phase": "surge", "japanese_name": "急騰期間", "recommended_method": "no_trade_or_small", "recommended_alias": "新規抑制", "note": "過熱が強く追いかけ買いは不利になりやすい"},
        {"phase": "downtrend", "japanese_name": "下落期間", "recommended_method": "no_trade", "recommended_alias": "新規停止", "note": "順張り買い手法とは相性が悪い"},
        {"phase": "crash", "japanese_name": "急落期間", "recommended_method": "no_trade", "recommended_alias": "新規停止", "note": "リスク回避を優先"},
        {"phase": "high_vol", "japanese_name": "高ボラ期間", "recommended_method": "no_trade", "recommended_alias": "新規停止", "note": "方向感が崩れやすく通常手法は不安定"},
        {"phase": "settling", "japanese_name": "収束期間", "recommended_method": "watch_only", "recommended_alias": "監視", "note": "次のトレンド発生待ち"},
        {"phase": "reversal_up", "japanese_name": "反転初動(上)", "recommended_method": "watch_or_q3_post_high_vol", "recommended_alias": "監視/高ボラ通過後", "note": "高ボラ後の選定手法を組み合わせる余地がある"},
        {"phase": "reversal_down", "japanese_name": "反転初動(下)", "recommended_method": "no_trade", "recommended_alias": "新規停止", "note": "下向き転換の初期で買い優位性が低い"},
        {"phase": "overheated_range", "japanese_name": "過熱もみ合い", "recommended_method": "method_breakout_1p5", "recommended_alias": "横ばい順張り", "note": "condition2 より絞った方がよい"},
        {"phase": "capitulation_end", "japanese_name": "投げ売り終盤", "recommended_method": "watch_or_q3_post_high_vol", "recommended_alias": "監視/高ボラ通過後", "note": "反発狙いは別手法で扱う"},
        {"phase": "normal", "japanese_name": "通常", "recommended_method": "method_condition2", "recommended_alias": "上向き標準", "note": "明確な異常が無い通常局面"},
    ]
    return pd.DataFrame(rows)


def infer_current_phase_from_history(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    feat_cols = ["ret5", "ret20", "dd20", "vol10", "ma_gap_20_60", "ma20_slope_5", "range20"]
    work = df.dropna(subset=feat_cols + ["phase"]).copy()
    latest = work.iloc[-1]
    hist = work.iloc[:-1].copy()
    X = hist[feat_cols].astype(float)
    mu = X.mean()
    sigma = X.std().replace(0, 1.0)
    z_hist = (X - mu) / sigma
    z_latest = ((latest[feat_cols].astype(float) - mu) / sigma).values
    d = np.sqrt(((z_hist.values - z_latest) ** 2).sum(axis=1))
    hist = hist.assign(distance=d).sort_values("distance").head(10)
    phase_votes = hist["phase"].value_counts(normalize=True).to_dict()
    inferred = max(phase_votes, key=phase_votes.get)
    payload = {
        "latest_date": latest.name.date().isoformat(),
        "latest_close": float(latest["close"]),
        "latest_rule_phase": str(latest["phase"]),
        "nearest_neighbor_inferred_phase": inferred,
        "phase_vote_share": phase_votes,
    }
    return hist.reset_index(), payload


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    labeled, meta = assign_phase(df)
    periods = summarize_periods(labeled)
    method_map = build_method_map()
    neighbors, current_payload = infer_current_phase_from_history(labeled)

    labeled.reset_index().to_csv(OUT_DIR / "nikkei_market_phase_daily_labels.csv", index=False, encoding="utf-8-sig")
    periods.to_csv(OUT_DIR / "nikkei_market_phase_periods.csv", index=False, encoding="utf-8-sig")
    method_map.to_csv(OUT_DIR / "nikkei_market_phase_method_map.csv", index=False, encoding="utf-8-sig")
    neighbors.to_csv(OUT_DIR / "nikkei_current_phase_neighbors.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(
        [{"metric": "vol10_low_threshold_pct", "value": round(meta["vol_lo"] * 100.0, 3)},
         {"metric": "vol10_mid_threshold_pct", "value": round(meta["vol_mid"] * 100.0, 3)},
         {"metric": "vol10_high_threshold_pct", "value": round(meta["vol_hi"] * 100.0, 3)}]
        + [
            {"metric": f"phase_days_{phase}", "value": int((labeled["phase"] == phase).sum())}
            for phase in method_map["phase"].tolist()
            if phase in set(labeled["phase"].unique())
        ]
    )
    summary.to_csv(OUT_DIR / "nikkei_market_phase_summary.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "nikkei_current_phase_inference.json").write_text(
        json.dumps(current_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(summary.to_string(index=False))
    print("---")
    print(json.dumps(current_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

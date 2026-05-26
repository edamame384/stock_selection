"""
False Signal Classifier for transition_to_uptrend zones.

Train: 2009-2012 + 2013-2016  (+ optional extended 2006-2008 via DB)
Test:  2018-2021 + 2022-2025

Outputs: data/false_signal_model/
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import psycopg2
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# ── Paths ─────────────────────────────────────────────────────────────────────

TREND_DIR = ROOT / "data" / "ssa_trend_analysis"
EXT_DIR = ROOT / "data" / "external_market"
OUTPUT_DIR = ROOT / "data" / "false_signal_model"

TRAIN_PERIODS = ["2009_2012", "2013_2016"]
TEST_PERIODS = ["2018_2021", "2022_2025"]
ALL_PERIODS = TRAIN_PERIODS + TEST_PERIODS

# ── DB connection ─────────────────────────────────────────────────────────────
DB_URL = "postgresql://postgres:ogm384@localhost:5432/stock_selection"

# ── SSA constants (same as nikkei_ssa_trend_label.py) ────────────────────────
_SSA_W = 40
_SSA_L = 20
_SSA_R = 1
_WARMUP_DAYS = 120  # calendar days before period start

# ── Label constants ───────────────────────────────────────────────────────────
_LABEL_EN = {
    "上昇トレンド":       "uptrend",
    "上昇トレンドへの転換期": "transition_to_uptrend",
    "下降トレンドへの転換期": "transition_to_downtrend",
    "下降トレンド":       "downtrend",
}

SSA_FEATURE_COLS = [
    # CSV から直接取得
    "ssa_40", "price_to_ssa_gap",
    "ssa_slope_3", "ssa_slope_5", "ssa_slope_20", "ssa_curvature",
    # 計算追加
    "ret_1", "ret_5", "ret_20",
    "vol_20", "volume_z20", "range_pos_20",
    # エンジニアリング
    "slope_5_to_20_ratio", "gap_z_40", "days_since_downtrend",
]

CROSS_FEATURE_COLS = [
    "vix_close", "vix_z_20", "vix_ret_5",
    "usdjpy_close", "usdjpy_ret_5", "usdjpy_z_20",
    "dji_ret_1", "dji_ret_5",
    "sp500_ret_5",
]

# B variant: レベル値を除き、変化率・z-スコアのみ使用  + 日次変化率を追加
CROSS_FEATURE_COLS_B = [
    "vix_z_20", "vix_ret_1", "vix_ret_5",
    "usdjpy_ret_1", "usdjpy_ret_5", "usdjpy_z_20",
    "dji_ret_1", "dji_ret_5",
    "sp500_ret_5",
]

# Backward-compat alias for existing callers
FEATURE_COLS = SSA_FEATURE_COLS

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
RANDOM_STATE = 42


# ── Variant-specific model / feature factories ────────────────────────────────

def get_models_for_variant(variant: str) -> dict:
    """Return ordered dict of {label: sklearn-compatible model} for a given variant.
    Variants: "AC", "D", "B", "DB"
    """
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=3, min_samples_leaf=3,
        random_state=RANDOM_STATE, class_weight="balanced",
    )
    # A: LR with StandardScaler pipeline
    lr_a = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced")),
    ])
    # plain LR (baseline / D / B / DB)
    lr_plain = LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced",
    )
    lgbm = LGBMClassifier(
        n_estimators=150, max_depth=3, min_child_samples=3,
        random_state=RANDOM_STATE, class_weight="balanced",
        verbose=-1, n_jobs=1,
    )
    xgb = XGBClassifier(
        n_estimators=150, max_depth=3, min_child_weight=3,
        random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0,
        use_label_encoder=False, n_jobs=1,
    )

    if variant == "AC":
        return {"rf": rf, "lr": lr_a}
    elif variant == "D":
        return {"rf": rf, "lr": lr_plain, "lgbm": lgbm, "xgb": xgb}
    elif variant == "B":
        return {"rf": rf, "lr": lr_plain}
    elif variant == "DB":
        return {"rf": rf, "lr": lr_plain, "lgbm": lgbm, "xgb": xgb}
    else:
        raise ValueError(f"Unknown variant: {variant}. Choose from AC, D, B, DB")


# ── Inline SSA computation (for extended training data) ──────────────────────

def _rolling_ssa_last(window: np.ndarray, l: int = _SSA_L, r: int = _SSA_R) -> float:
    n = len(window)
    k = n - l + 1
    hankel = np.column_stack([window[i: i + k] for i in range(l)])
    u, s, vt = np.linalg.svd(hankel, full_matrices=False)
    recon = np.zeros_like(hankel)
    for i in range(r):
        recon += s[i] * np.outer(u[:, i], vt[i, :])
    last_vals = []
    for row in range(k):
        col = (n - 1) - row
        if 0 <= col < l:
            last_vals.append(recon[row, col])
    return float(np.mean(last_vals)) if last_vals else float("nan")


def _compute_ssa(closes: np.ndarray, w: int = _SSA_W, l: int = _SSA_L, r: int = _SSA_R) -> np.ndarray:
    ssa = np.full(len(closes), np.nan)
    for i in range(w - 1, len(closes)):
        ssa[i] = _rolling_ssa_last(closes[i - w + 1: i + 1], l=l, r=r)
    return ssa


def _assign_label_en(slope5: float, slope20: float) -> str:
    if np.isnan(slope5) or np.isnan(slope20):
        return ""
    if slope5 > 0 and slope20 > 0:
        return "uptrend"
    if slope5 < 0 and slope20 < 0:
        return "downtrend"
    if slope5 >= 0:
        return "transition_to_uptrend"
    return "transition_to_downtrend"


def load_nikkei_from_db(start_date: str = "2005-01-01") -> pd.DataFrame:
    """Load N225 OHLCV from PostgreSQL from start_date onward."""
    conn = psycopg2.connect(DB_URL)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT date, open, high, low, close, volume "
            "FROM market_indices "
            "WHERE symbol = '^N225' AND date >= %s "
            "ORDER BY date",
            (start_date,)
        )
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
    finally:
        conn.close()
    df = pd.DataFrame(rows, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def compute_ssa_labels_for_period(full_df: pd.DataFrame,
                                  period_start: str, period_end: str,
                                  tag: str) -> pd.DataFrame:
    """Run SSA + label assignment for a period, returning analysis-ready DataFrame.
    full_df must have columns: date, open, high, low, close, volume (all rows).
    """
    ps = pd.Timestamp(period_start)
    pe = pd.Timestamp(period_end)
    warmup_start = ps - pd.Timedelta(days=_WARMUP_DAYS)

    work = full_df[(full_df["date"] >= warmup_start) & (full_df["date"] <= pe)].copy()
    if work.empty:
        raise RuntimeError(f"No data for {period_start}~{period_end}")

    closes = work["close"].astype(float).to_numpy()
    work = work.reset_index(drop=True)
    work["ssa_40"] = _compute_ssa(closes)
    work["price_to_ssa_gap"] = work["close"] / work["ssa_40"] - 1.0
    work["ssa_slope_3"]  = work["ssa_40"].pct_change(3)
    work["ssa_slope_5"]  = work["ssa_40"].pct_change(5)
    work["ssa_slope_20"] = work["ssa_40"].pct_change(20)
    work["ssa_curvature"] = work["ssa_slope_5"] - work["ssa_slope_5"].shift(5)
    work["trend_label_en"] = [
        _assign_label_en(s5, s20)
        for s5, s20 in zip(work["ssa_slope_5"], work["ssa_slope_20"])
    ]
    result = work[(work["date"] >= ps) & (work["date"] <= pe)].copy()
    return result


def build_extended_training_signals() -> pd.DataFrame:
    """Load pre-2009 Nikkei data from DB, compute SSA, extract signals for 1990-2008.
    Splits into 4-year sub-periods to keep SSA computation manageable and
    match the granularity of the existing period CSVs.
    Returns signal rows in the same format as find_long_zone_signals().
    """
    # warmup 用に少し前から取得（SSA に 120日分のバッファが必要）
    DB_FETCH_START = "1989-01-01"
    EXT_PERIOD_START = "1990-01-01"   # 実際のラベリング開始
    EXT_PERIOD_END   = "2008-12-31"   # DB 既存データの直前まで

    # 4年ごとのサブ期間に分割
    SUB_PERIODS = [
        ("1990-01-01", "1993-12-31", "1990_1993"),
        ("1994-01-01", "1997-12-31", "1994_1997"),
        ("1998-01-01", "2001-12-31", "1998_2001"),
        ("2002-01-01", "2005-12-31", "2002_2005"),
        ("2006-01-01", "2008-12-31", "2006_2008"),
    ]

    print(f"[EXT] Loading N225 from DB ({DB_FETCH_START} ~)...")
    full_df = load_nikkei_from_db(start_date=DB_FETCH_START)
    if full_df.empty:
        print("[EXT] No data returned — skipping")
        return pd.DataFrame()

    earliest = full_df["date"].min()
    print(f"[EXT] DB date range: {earliest.date()} ~ {full_df['date'].max().date()}  rows={len(full_df)}")

    all_sigs = []
    for ps, pe, tag in SUB_PERIODS:
        ps_ts = pd.Timestamp(ps)
        # データが足りなければスキップ
        if ps_ts < earliest + pd.Timedelta(days=_WARMUP_DAYS + 5):
            avail_start = earliest + pd.Timedelta(days=_WARMUP_DAYS + 5)
            if avail_start >= pd.Timestamp(pe):
                print(f"[EXT] {tag}: not enough warmup data — skip")
                continue
            ps = str(avail_start.date())
            tag = f"{avail_start.year}_{pd.Timestamp(pe).year}"

        try:
            period_df = compute_ssa_labels_for_period(full_df, ps, pe, tag)
        except RuntimeError as e:
            print(f"[EXT] {tag}: {e} — skip")
            continue

        if period_df.empty:
            continue

        period_df = enrich_features(period_df)
        sigs = find_long_zone_signals(period_df, tag)
        n_false = int(sigs["is_false"].sum()) if len(sigs) else 0
        print(f"[EXT] {tag}: signals={len(sigs)}  is_false={n_false}")
        all_sigs.append(sigs)

    if not all_sigs:
        return pd.DataFrame()

    combined = pd.concat(all_sigs, ignore_index=True)
    print(f"[EXT] Total extended signals: {len(combined)}  is_false={combined['is_false'].sum()}")
    return combined


# ── Feature engineering ──────────────────────────────────────────────────────

def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute additional features from base SSA trend CSV columns."""
    df = df.sort_values("date").reset_index(drop=True).copy()

    df["ret_1"]  = df["close"].pct_change(1)
    df["ret_5"]  = df["close"].pct_change(5)
    df["ret_20"] = df["close"].pct_change(20)

    df["vol_20"] = df["ret_1"].rolling(20).std()

    vol_mean = df["volume"].rolling(20).mean()
    vol_std  = df["volume"].rolling(20).std().replace(0, np.nan)
    df["volume_z20"] = ((df["volume"] - vol_mean) / vol_std).fillna(0)

    high20 = df["close"].rolling(20).max()
    low20  = df["close"].rolling(20).min()
    range_w = (high20 - low20).replace(0, np.nan)
    df["range_pos_20"] = ((df["close"] - low20) / range_w).fillna(0.5)

    denom = df["ssa_slope_20"].replace(0, np.nan)
    df["slope_5_to_20_ratio"] = (df["ssa_slope_5"] / denom).replace([np.inf, -np.inf], np.nan).fillna(0)

    gap_mean = df["price_to_ssa_gap"].rolling(40).mean()
    gap_std  = df["price_to_ssa_gap"].rolling(40).std().replace(0, np.nan)
    df["gap_z_40"] = ((df["price_to_ssa_gap"] - gap_mean) / gap_std).fillna(0)

    is_dn = (df["trend_label_en"] == "downtrend").to_numpy()
    days_since = np.full(len(df), 999, dtype=int)
    last = -1
    for i in range(len(df)):
        if is_dn[i]:
            last = i
        if last >= 0:
            days_since[i] = i - last
    df["days_since_downtrend"] = days_since

    return df


# ── External cross indicators ────────────────────────────────────────────────

def _load_ext_csv(name: str) -> pd.DataFrame:
    """Load cached external indicator CSV. Returns DataFrame indexed by date with Close col."""
    path = EXT_DIR / f"{name}_daily.csv"
    if not path.exists():
        raise RuntimeError(f"Missing external CSV {path}. Run scripts/fetch_external_indicators.py first.")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "date"
    df = df.sort_index()
    # ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume'] と多列ある場合は Close 使用
    if "Close" in df.columns:
        df = df[["Close"]].rename(columns={"Close": "close"})
    else:
        df.columns = [c.lower() for c in df.columns]
    return df


def build_ext_features(name: str, ext_df: pd.DataFrame) -> pd.DataFrame:
    """Compute features per external series; returns DataFrame indexed by date."""
    df = ext_df.copy()
    out = pd.DataFrame(index=df.index)
    out[f"{name}_close"] = df["close"]
    out[f"{name}_ret_1"] = df["close"].pct_change(1)
    out[f"{name}_ret_5"] = df["close"].pct_change(5)
    mean20 = df["close"].rolling(20).mean()
    std20  = df["close"].rolling(20).std().replace(0, np.nan)
    out[f"{name}_z_20"] = (df["close"] - mean20) / std20
    return out


def load_all_cross_features() -> pd.DataFrame:
    """Returns a single DataFrame indexed by date with all cross feature columns."""
    parts = []
    for name in ["vix", "dji", "usdjpy", "sp500"]:
        ext = _load_ext_csv(name)
        feats = build_ext_features(name, ext)
        parts.append(feats)
    merged = pd.concat(parts, axis=1).sort_index()
    return merged


def attach_cross_features(signal_df: pd.DataFrame, cross_feat_df: pd.DataFrame) -> pd.DataFrame:
    """For each signal_date in signal_df, attach cross feature values using as-of merge.
    Uses direction='backward' so we get the latest external close at-or-before signal_date.
    """
    sig = signal_df.sort_values("signal_date").reset_index(drop=True).copy()
    cross = cross_feat_df.sort_index().reset_index()
    cross["date"] = pd.to_datetime(cross["date"])
    merged = pd.merge_asof(
        sig, cross,
        left_on="signal_date", right_on="date",
        direction="backward",
        tolerance=pd.Timedelta(days=7),
    )
    merged = merged.drop(columns=["date"])
    # 必要な特徴量を選別（vix_ret_1 等は build_ext_features で作っているが、CROSS_FEATURE_COLS にないものは無視）
    return merged


# ── Signal extraction ────────────────────────────────────────────────────────

def find_long_zone_signals(df: pd.DataFrame, period_tag: str) -> pd.DataFrame:
    """Identify each (transition_to_uptrend OR uptrend) zone, compute trade outcome.
    Returns DataFrame with meta columns + feature columns + is_false label.
    Filter: only zones whose first day is transition_to_uptrend (classifier target).
    Also returns non-transition entries flagged separately for downstream merging.
    """
    df = df.sort_values("date").reset_index(drop=True)
    df["is_long_label"] = df["trend_label_en"].isin(["transition_to_uptrend", "uptrend"])
    df["long_zone_run"] = (df["is_long_label"] != df["is_long_label"].shift()).cumsum()

    rows = []
    for run_id, grp in df[df["is_long_label"]].groupby("long_zone_run"):
        first_idx = int(grp.index[0])
        first_label = df.iloc[first_idx]["trend_label_en"]
        if first_idx + 1 >= len(df):
            continue
        entry_iloc = first_idx + 1
        if df.iloc[entry_iloc]["long_zone_run"] != run_id:
            continue  # 1-day zone
        entry_date = pd.Timestamp(df.iloc[entry_iloc]["date"])
        entry_price = float(df.iloc[entry_iloc]["open"])
        exit_row = grp.iloc[-1]
        exit_date = pd.Timestamp(exit_row["date"])
        exit_price = float(exit_row["close"])
        if entry_date > exit_date:
            continue
        return_pct = (exit_price / entry_price - 1) * 100
        signal_row = df.iloc[first_idx]
        feat = {col: signal_row[col] for col in FEATURE_COLS if col in df.columns}
        rows.append(dict(
            period           = period_tag,
            signal_date      = pd.Timestamp(signal_row["date"]),
            first_label      = first_label,
            starts_with_transition = first_label == "transition_to_uptrend",
            entry_date       = entry_date,
            exit_date        = exit_date,
            entry_price      = round(entry_price, 2),
            exit_price       = round(exit_price, 2),
            hold_days        = int(len(grp)),
            return_pct       = round(return_pct, 4),
            is_false         = int(return_pct < 0),
            **feat,
        ))
    return pd.DataFrame(rows)


def build_signal_dataset(extend_training: bool = True) -> pd.DataFrame:
    """Load all 4 trend CSVs, enrich features, extract signals, concatenate.
    If extend_training=True, also load pre-2009 data from DB (part C).
    """
    all_sig = []
    for tag in ALL_PERIODS:
        path = TREND_DIR / f"nikkei_ssa_trend_{tag}.csv"
        df = pd.read_csv(path, parse_dates=["date"])
        df = enrich_features(df)
        sig = find_long_zone_signals(df, tag)
        all_sig.append(sig)

    # C: 学習データ拡大 — pre-2009 シグナルを DB から追加
    if extend_training:
        ext_sigs = build_extended_training_signals()
        if len(ext_sigs) > 0:
            all_sig.insert(0, ext_sigs)  # 先頭に挿入（時系列順を維持）

    full = pd.concat(all_sig, ignore_index=True)
    # 特徴量に NaN がある行を除外（rolling window 不足など）
    n_before = len(full)
    full = full.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    n_after = len(full)
    if n_before != n_after:
        print(f"[DATA] dropped {n_before - n_after} signals due to NaN features (rolling window)")
    return full


# ── Model training / evaluation (generic, model-agnostic) ────────────────────

def _get_fi_array(model, n: int) -> np.ndarray:
    """Extract feature importance array from any sklearn-compatible model."""
    m = model.named_steps[list(model.named_steps)[-1]] if isinstance(model, Pipeline) else model
    if hasattr(m, "feature_importances_"):
        return m.feature_importances_
    if hasattr(m, "coef_"):
        return np.abs(m.coef_[0])
    return np.zeros(n)


def train_all_models(train_df: pd.DataFrame, feature_cols: list[str],
                     models: dict) -> tuple[dict, dict]:
    """Train all models with CV; returns (fitted_models, cv_metrics)."""
    X = train_df[feature_cols].to_numpy()
    y = train_df["is_false"].to_numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    fitted = {}
    cv_metrics = {}
    for name, model in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_auc = cross_val_score(model, X, y, cv=skf, scoring="roc_auc")
            cv_acc = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
            model.fit(X, y)
        fitted[name] = model
        cv_metrics[name] = dict(
            cv_auc_mean=float(cv_auc.mean()), cv_auc_std=float(cv_auc.std()),
            cv_acc_mean=float(cv_acc.mean()), cv_acc_std=float(cv_acc.std()),
        )
    return fitted, cv_metrics


def evaluate_all_models(test_df: pd.DataFrame, feature_cols: list[str],
                        fitted: dict) -> tuple[dict, dict]:
    """Evaluate all models; returns (test_metrics, proba_dict)."""
    X = test_df[feature_cols].to_numpy()
    y = test_df["is_false"].to_numpy()
    test_metrics = {}
    proba = {}
    for name, model in fitted.items():
        p = model.predict_proba(X)[:, 1]
        proba[name] = p
        pred_05 = (p >= 0.5).astype(int)
        cm = confusion_matrix(y, pred_05, labels=[0, 1])
        test_metrics[name] = dict(
            test_auc =float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
            test_acc =float(accuracy_score(y, pred_05)),
            test_prec=float(precision_score(y, pred_05, zero_division=0)),
            test_rec =float(recall_score(y, pred_05, zero_division=0)),
            cm_tn=int(cm[0, 0]), cm_fp=int(cm[0, 1]),
            cm_fn=int(cm[1, 0]), cm_tp=int(cm[1, 1]),
        )
    return test_metrics, proba


def feature_importance_df(fitted: dict, feature_cols: list[str],
                          fset_name: str) -> pd.DataFrame:
    n = len(feature_cols)
    fi = pd.DataFrame({"feature": feature_cols})
    for name, model in fitted.items():
        fi[f"{name}_importance"] = _get_fi_array(model, n)
    fi["feature_set"] = fset_name
    # Sort by first model's importance
    first_col = [c for c in fi.columns if c.endswith("_importance")][0]
    fi = fi.sort_values(first_col, ascending=False).reset_index(drop=True)
    return fi


# ── Filtered backtest ────────────────────────────────────────────────────────

def apply_filter_backtest(test_df: pd.DataFrame, prob_col: str,
                          thresholds: list[float]) -> pd.DataFrame:
    """Apply thresholds and compute filtered cumulative return per (threshold, period).
    Non-transition (starts_with_transition == False) trades are always kept.
    Transition trades are kept only if p < threshold.
    """
    rows = []
    for tag in TEST_PERIODS:
        sub = test_df[test_df["period"] == tag]
        # Baseline (no filter)
        ret_base = float((1 + sub["return_pct"] / 100).prod() - 1) * 100
        n_base = len(sub)
        wr_base = float(sub["is_false"].apply(lambda x: 0 if x else 1).mean()) if n_base else float("nan")
        rows.append(dict(
            threshold="baseline_all",
            period=tag,
            n_trades=int(n_base),
            n_filtered_out=0,
            win_rate=round(wr_base, 3) if n_base else float("nan"),
            total_return_pct=round(ret_base, 2),
        ))
        for t in thresholds:
            mask_keep = (~sub["starts_with_transition"]) | (sub[prob_col] < t)
            kept = sub[mask_keep]
            ret = float((1 + kept["return_pct"] / 100).prod() - 1) * 100 if len(kept) else 0.0
            wr = float(kept["is_false"].apply(lambda x: 0 if x else 1).mean()) if len(kept) else float("nan")
            rows.append(dict(
                threshold=f"{t:.2f}",
                period=tag,
                n_trades=int(len(kept)),
                n_filtered_out=int(len(sub) - len(kept)),
                win_rate=round(wr, 3) if len(kept) else float("nan"),
                total_return_pct=round(ret, 2),
            ))
    return pd.DataFrame(rows)


# ── Charts ───────────────────────────────────────────────────────────────────

def setup_japanese_font():
    for font in ["BIZ UDGothic", "Meiryo", "MS Gothic", "IPAGothic"]:
        try:
            matplotlib.rcParams["font.family"] = font
            return
        except Exception:
            continue


def load_close_series(period_tag: str) -> pd.DataFrame:
    path = TREND_DIR / f"nikkei_ssa_trend_{period_tag}.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    return df[["date", "close"]].sort_values("date").reset_index(drop=True)


def build_equity_curve_from_trades(close_df: pd.DataFrame,
                                   trades_df: pd.DataFrame) -> np.ndarray:
    """Linear-interpolated equity curve given trades_df with entry_date/exit_date/return_pct."""
    date_to_idx = {pd.Timestamp(d).date(): i for i, d in enumerate(close_df["date"])}
    equity = np.ones(len(close_df))
    running = 1.0
    for _, t in trades_df.sort_values("entry_date").iterrows():
        ei = date_to_idx.get(pd.Timestamp(t["entry_date"]).date())
        xi = date_to_idx.get(pd.Timestamp(t["exit_date"]).date())
        if ei is None or xi is None:
            continue
        end_val = running * (1 + t["return_pct"] / 100)
        if xi > ei:
            for j in range(ei, xi + 1):
                frac = (j - ei) / (xi - ei)
                equity[j] = running + (end_val - running) * frac
        else:
            equity[xi] = end_val
        for j in range(xi + 1, len(equity)):
            equity[j] = end_val
        running = end_val
    return equity


def plot_equity_comparison(test_df: pd.DataFrame, prob_col: str,
                           best_threshold: float, out_path: Path,
                           period_tag: str) -> None:
    setup_japanese_font()
    close_df = load_close_series(period_tag)
    xs = np.arange(len(close_df))

    sub = test_df[test_df["period"] == period_tag]
    base_eq = build_equity_curve_from_trades(close_df, sub)
    mask_keep = (~sub["starts_with_transition"]) | (sub[prob_col] < best_threshold)
    filtered_eq = build_equity_curve_from_trades(close_df, sub[mask_keep])
    bnh = (close_df["close"].to_numpy() / float(close_df["close"].iloc[0]))

    fig, ax = plt.subplots(figsize=(14, 6), facecolor="#1e1e2e")
    ax.set_facecolor("#1e1e2e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")

    ax.plot(xs, (base_eq - 1) * 100, color="#8888cc", lw=1.4, label="フィルタなし(s1ロング)", alpha=0.85)
    ax.plot(xs, (filtered_eq - 1) * 100, color="#f39c12", lw=2.2,
            label=f"フィルタあり(τ={best_threshold:.2f}, {prob_col})")
    ax.plot(xs, (bnh - 1) * 100, color="#aaaaaa", lw=1.0, linestyle="--", label="Buy&Hold", alpha=0.7)
    ax.axhline(0, color="#666688", lw=0.7, linestyle="--")

    # X 軸: 4ヶ月ごと
    ticks, labels = [], []
    prev = None
    for i, d in enumerate(close_df["date"]):
        p = (d.year, (d.month - 1) // 4)
        if p != prev:
            ticks.append(i); labels.append(d.strftime("%Y/%m"))
            prev = p
    ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=30, ha="right",
                                              fontsize=8, color="#cccccc")
    ax.tick_params(colors="#cccccc")
    ax.yaxis.set_tick_params(labelcolor="#cccccc")
    ax.set_ylabel("累積損益(%)", color="#cccccc")
    ax.set_title(f"日経225 偽シグナルフィルタ効果  期間: {period_tag.replace('_', '-')}",
                 color="white", fontsize=11)
    ax.legend(loc="upper left", fontsize=9, facecolor="#2a2a3e",
              edgecolor="#555577", labelcolor="white", framealpha=0.85)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    # ── Variant selection ────────────────────────────────────────────────
    variant = "AC"
    for arg in sys.argv[1:]:
        if arg.startswith("--variant="):
            variant = arg.split("=", 1)[1].upper()
    valid_variants = {"AC", "D", "B", "DB"}
    if variant not in valid_variants:
        print(f"Unknown variant '{variant}'. Choose from: {valid_variants}")
        return 1

    # Variant-specific config
    # 全バリアントで学習データ拡大を有効化（DB に 1965 年以降のデータが入った後）
    use_extend_training = True
    cross_cols = CROSS_FEATURE_COLS_B if variant in ("B", "DB") else CROSS_FEATURE_COLS
    all_cross_cols_needed = CROSS_FEATURE_COLS_B if variant in ("B", "DB") else CROSS_FEATURE_COLS
    # vix_ret_1 and usdjpy_ret_1 are computed by build_ext_features but not in default CROSS_FEATURE_COLS
    # ensure they are available for B/DB

    FEATURE_SETS = {
        "ssa":       SSA_FEATURE_COLS,
        "ssa_cross": SSA_FEATURE_COLS + cross_cols,
    }

    print(f"\n{'#'*70}")
    print(f"# VARIANT: {variant}")
    print(f"#   Models:           {list(get_models_for_variant(variant).keys())}")
    print(f"#   Cross features:   {'B (returns-only)' if variant in ('B','DB') else 'original (with levels)'}")
    print(f"#   Extend training:  {use_extend_training}")
    print(f"{'#'*70}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────
    print("[DATA] Building signal dataset...")
    full_df = build_signal_dataset(extend_training=use_extend_training)
    print(f"[DATA] total signals: {len(full_df)}")

    classifier_df = full_df[full_df["starts_with_transition"]].reset_index(drop=True)
    print(f"[DATA] transition-start signals: {len(classifier_df)}")
    print(classifier_df.groupby(["period", "is_false"]).size().to_string())

    # Attach cross indicators (all columns needed)
    print("\n[DATA] attaching cross indicators (VIX, DJI, USDJPY, S&P500)...")
    cross_feat = load_all_cross_features()
    classifier_df = attach_cross_features(classifier_df, cross_feat)

    # Drop rows missing any cross feature we need
    nan_mask = classifier_df[all_cross_cols_needed].isna().any(axis=1)
    if nan_mask.sum() > 0:
        print(f"[DATA] {nan_mask.sum()} signals have NaN cross features -> dropping")
        classifier_df = classifier_df[~nan_mask].reset_index(drop=True)
    print(f"[DATA] signals after cross-merge: {len(classifier_df)}")

    train_df = classifier_df[~classifier_df["period"].isin(TEST_PERIODS)].reset_index(drop=True)
    test_df  = classifier_df[classifier_df["period"].isin(TEST_PERIODS)].reset_index(drop=True)
    print(f"\n[DATA] train={len(train_df)}  test={len(test_df)}")
    print(f"[DATA] train is_false ratio: {train_df['is_false'].mean():.2%}")
    print(f"[DATA] test  is_false ratio: {test_df['is_false'].mean():.2%}")

    train_df.to_csv(OUTPUT_DIR / f"training_data_{variant}.csv", index=False, encoding="utf-8-sig")

    # ── Train and evaluate ────────────────────────────────────────────────
    all_results: dict = {}
    test_df_out = test_df.copy()
    all_fi: list[pd.DataFrame] = []

    for fset_name, feat_cols in FEATURE_SETS.items():
        print(f"\n{'='*70}")
        print(f"[MODEL] variant={variant}  feature_set={fset_name}  ({len(feat_cols)} features)")
        print("="*70)

        models_dict = get_models_for_variant(variant)
        fitted, cv_metrics = train_all_models(train_df, feat_cols, models_dict)

        for mname, cm in cv_metrics.items():
            print(f"  {mname:<6}  CV AUC: {cm['cv_auc_mean']:.3f} +/- {cm['cv_auc_std']:.3f}")

        test_metrics, proba = evaluate_all_models(test_df, feat_cols, fitted)
        for mname, tm in test_metrics.items():
            print(f"  {mname:<6}  Test AUC={tm['test_auc']:.3f}  acc={tm['test_acc']:.3f}  "
                  f"prec={tm['test_prec']:.3f}  rec={tm['test_rec']:.3f}")
            test_df_out[f"p_{mname}_{fset_name}"] = proba[mname]

        fi = feature_importance_df(fitted, feat_cols, fset_name)
        all_fi.append(fi)
        all_results[fset_name] = {"cv": cv_metrics, "test": test_metrics, "n_feat": len(feat_cols)}

        # Save models
        for mname, model in fitted.items():
            joblib.dump(model, OUTPUT_DIR / f"model_{variant}_{mname}_{fset_name}.pkl")

        print(f"\n  [Top 8 features by RF importance ({fset_name})]")
        top8 = fi.head(8)[["feature"] + [c for c in fi.columns if c.endswith("_importance")]].to_string(index=False)
        print(top8)

    # ── Save outputs ─────────────────────────────────────────────────────
    test_df_out.to_csv(OUTPUT_DIR / f"test_data_{variant}.csv", index=False, encoding="utf-8-sig")
    fi_all = pd.concat(all_fi, ignore_index=True)
    fi_all.to_csv(OUTPUT_DIR / f"feature_importance_{variant}.csv", index=False, encoding="utf-8-sig")
    with open(OUTPUT_DIR / f"model_metrics_{variant}.json", "w", encoding="utf-8") as f:
        json.dump({
            "variant": variant,
            "n_train": int(len(train_df)), "n_test": int(len(test_df)),
            "n_train_false": int(train_df["is_false"].sum()),
            "n_test_false": int(test_df["is_false"].sum()),
            "results": {fs: {
                "cv": {mn: {k: round(v, 4) for k, v in cm.items()} for mn, cm in rs["cv"].items()},
                "test": {mn: {k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in tm.items()} for mn, tm in rs["test"].items()},
            } for fs, rs in all_results.items()},
        }, f, indent=2, ensure_ascii=False)

    # ── AUC summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"[AUC COMPARISON]  variant={variant}")
    print(f"  {'feature_set':<12} {'model':<8} {'CV_AUC':>15}  {'Test_AUC':>8}")
    print("="*70)
    for fset_name, rs in all_results.items():
        for mname in rs["cv"]:
            cm = rs["cv"][mname]
            tm = rs["test"][mname]
            print(f"  {fset_name:<12} {mname:<8} {cm['cv_auc_mean']:.3f}+/-{cm['cv_auc_std']:.3f}   {tm['test_auc']:.3f}")

    # ── Threshold sweep ───────────────────────────────────────────────────
    print("\n[FILTER] threshold sweep...")
    thr_parts = []
    for fset_name in FEATURE_SETS:
        model_names = list(get_models_for_variant(variant).keys())
        for mname in model_names:
            prob_col = f"p_{mname}_{fset_name}"
            if prob_col not in test_df_out.columns:
                continue
            sub = apply_filter_backtest(test_df_out, prob_col, THRESHOLDS)
            sub["model"] = mname
            sub["feature_set"] = fset_name
            thr_parts.append(sub)

    thr_all = pd.concat(thr_parts, ignore_index=True)
    thr_all = thr_all[["feature_set", "model", "threshold", "period",
                        "n_trades", "n_filtered_out", "win_rate", "total_return_pct"]]
    thr_all.to_csv(OUTPUT_DIR / f"threshold_comparison_{variant}.csv", index=False, encoding="utf-8-sig")

    # B&H per period
    print("\n[B&H baseline]")
    for tag in TEST_PERIODS:
        cd = load_close_series(tag)
        bnh = (float(cd["close"].iloc[-1]) / float(cd["close"].iloc[0]) - 1) * 100
        print(f"  {tag}: {bnh:.2f}%")

    baseline_combined = sum(
        thr_all[(thr_all["threshold"] == "baseline_all") & (thr_all["period"] == tag)]["total_return_pct"].iloc[0]
        for tag in TEST_PERIODS
        if len(thr_all[(thr_all["threshold"] == "baseline_all") & (thr_all["period"] == tag)]) > 0
    )
    print(f"  Baseline combined (no filter): {baseline_combined:.2f}%")

    # ── Best threshold per variant ────────────────────────────────────────
    print(f"\n[BEST THRESHOLD per (feature_set, model)] variant={variant}")
    best_overall = ("", "", "", -999.0)
    for fset_name in FEATURE_SETS:
        model_names = list(get_models_for_variant(variant).keys())
        for mname in model_names:
            sub = thr_all[(thr_all["model"] == mname)
                          & (thr_all["feature_set"] == fset_name)
                          & (thr_all["threshold"] != "baseline_all")]
            if sub.empty:
                continue
            agg = sub.groupby("threshold")["total_return_pct"].sum().reset_index()
            best = agg.loc[agg["total_return_pct"].idxmax()]
            best_ret = float(best["total_return_pct"])
            print(f"  {fset_name:<12} {mname:<8} best tau={best['threshold']}  combined={best_ret:.2f}%")
            if best_ret > best_overall[3]:
                best_overall = (fset_name, mname, str(best["threshold"]), best_ret)
            # Chart for ssa_cross only
            if fset_name == "ssa_cross":
                prob_col = f"p_{mname}_{fset_name}"
                best_t = float(best["threshold"])
                for tag in TEST_PERIODS:
                    out_path = OUTPUT_DIR / f"equity_{variant}_{mname}_{fset_name}_{tag}.png"
                    plot_equity_comparison(test_df_out, prob_col, best_t, out_path, tag)
                    print(f"    [CHART] {out_path}")

    print(f"\n[OVERALL BEST] {best_overall[0]} / {best_overall[1]} / tau={best_overall[2]} -> {best_overall[3]:.2f}%")
    print(f"  (baseline no-filter combined: {baseline_combined:.2f}%)")
    print(f"\n[DONE] variant={variant}  outputs in {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

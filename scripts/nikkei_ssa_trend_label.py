"""
Nikkei 225 Rolling SSA Trend Label Analysis

Periods:
  2009-01-01 to 2012-12-31
  2018-01-01 to 2021-12-31

Data source: PostgreSQL market_indices table (READ ONLY)
Outputs: CSV + PNG charts to data/ssa_trend_analysis/
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import psycopg2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

# ── SSA params (same as update_nikkei_phase.py) ─────────────────────────────
W = 40
L = 20
R = 1

# ── Slope thresholds ─────────────────────────────────────────────────────────
UP_THRESHOLD = 0.005    # slope_5サブプロット参考線
DOWN_THRESHOLD = -0.005
SLOPE_FILTER = 0.002    # 戦略③: エントリー時の最小|slope5|

# ── Target periods ────────────────────────────────────────────────────────────
PERIODS = [
    ("2009-01-01", "2012-12-31", "2009_2012"),
    ("2013-01-01", "2016-12-31", "2013_2016"),
    ("2018-01-01", "2021-12-31", "2018_2021"),
    ("2022-01-01", "2025-12-31", "2022_2025"),
]
WARMUP_CALENDAR_DAYS = 120

# ── Labels（4クラス、停滞期なし）────────────────────────────────────────────
LBL_UP        = "上昇トレンド"
LBL_TRANS_UP  = "上昇トレンドへの転換期"
LBL_TRANS_DOWN = "下降トレンドへの転換期"
LBL_DOWN      = "下降トレンド"

LABEL_EN = {
    LBL_UP:        "uptrend",
    LBL_TRANS_UP:  "transition_to_uptrend",
    LBL_TRANS_DOWN: "transition_to_downtrend",
    LBL_DOWN:      "downtrend",
}
LABEL_SCORE = {LBL_UP: 4, LBL_TRANS_UP: 3, LBL_TRANS_DOWN: 2, LBL_DOWN: 1}
LABEL_COLOR = {
    LBL_UP:        "#2ecc71",  # 緑
    LBL_TRANS_UP:  "#3498db",  # 青
    LBL_TRANS_DOWN: "#e67e22", # オレンジ
    LBL_DOWN:      "#e74c3c",  # 赤
}
ALL_LABELS = [LBL_UP, LBL_TRANS_UP, LBL_TRANS_DOWN, LBL_DOWN]

DB_URL = "postgresql://postgres:ogm384@localhost:5432/stock_selection"
OUTPUT_DIR = ROOT / "data" / "ssa_trend_analysis"


# ── Helpers ───────────────────────────────────────────────────────────────────

def setup_japanese_font() -> None:
    for font in ["BIZ UDGothic", "Meiryo", "MS Gothic", "IPAGothic"]:
        try:
            matplotlib.rcParams["font.family"] = font
            # Quick test
            import matplotlib.font_manager as fm
            if fm.findfont(font, fallback_to_default=False):
                return
        except Exception:
            continue


def load_nikkei_from_db() -> pd.DataFrame:
    conn = psycopg2.connect(DB_URL)
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT date, open, high, low, close, volume
            FROM market_indices
            WHERE symbol = '^N225'
              AND date >= '2008-01-01'
            ORDER BY date
        """)
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
    finally:
        conn.close()
    df = pd.DataFrame(rows, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df.index.name = "date"
    return df


def rolling_ssa_last(window: np.ndarray, l: int = L, r: int = R) -> float:
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


def compute_ssa(closes: np.ndarray, w: int, l: int, r: int) -> np.ndarray:
    ssa = np.full(len(closes), np.nan)
    for i in range(w - 1, len(closes)):
        ssa[i] = rolling_ssa_last(closes[i - w + 1: i + 1], l=l, r=r)
    return ssa


def assign_label(slope5: float, slope20: float) -> str:
    """4クラス判定（停滞期なし）。slope5・slope20の符号のみで分類。"""
    if np.isnan(slope5) or np.isnan(slope20):
        return ""
    if slope5 > 0 and slope20 > 0:
        return LBL_UP
    if slope5 < 0 and slope20 < 0:
        return LBL_DOWN
    if slope5 >= 0:  # slope5>0 & slope20<=0、またはslope5==0
        return LBL_TRANS_UP
    return LBL_TRANS_DOWN  # slope5<0 & slope20>=0


def analyze_period(full_df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    period_start = pd.Timestamp(start)
    period_end = pd.Timestamp(end)
    warmup_start = period_start - pd.Timedelta(days=WARMUP_CALENDAR_DAYS)

    df_work = full_df[(full_df.index >= warmup_start) & (full_df.index <= period_end)].copy()
    if df_work.empty:
        raise RuntimeError(f"No data found for {start} to {end}")

    closes = df_work["close"].astype(float).to_numpy()
    df_work["ssa_40"] = compute_ssa(closes, W, L, R)
    df_work["price_to_ssa_gap"] = df_work["close"] / df_work["ssa_40"] - 1.0
    df_work["ssa_slope_3"] = df_work["ssa_40"].pct_change(3)
    df_work["ssa_slope_5"] = df_work["ssa_40"].pct_change(5)
    df_work["ssa_slope_20"] = df_work["ssa_40"].pct_change(20)
    df_work["ssa_curvature"] = df_work["ssa_slope_5"] - df_work["ssa_slope_5"].shift(5)

    df_work["trend_label_ja"] = [
        assign_label(s5, s20)
        for s5, s20 in zip(df_work["ssa_slope_5"], df_work["ssa_slope_20"])
    ]
    df_work["trend_label_en"] = df_work["trend_label_ja"].map(LABEL_EN).fillna("")
    df_work["trend_score"] = df_work["trend_label_ja"].map(LABEL_SCORE)

    result = df_work[(df_work.index >= period_start) & (df_work.index <= period_end)].copy()
    return result


# ── Chart ─────────────────────────────────────────────────────────────────────

def _draw_candlesticks(ax: plt.Axes, df: pd.DataFrame) -> None:
    body_w = 0.4
    for i, (_, row) in enumerate(df.iterrows()):
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        color = "#26a69a" if c >= o else "#ef5350"
        body_lo = min(o, c)
        body_hi = max(o, c)
        ax.add_patch(mpatches.Rectangle(
            (i - body_w, body_lo), body_w * 2, body_hi - body_lo,
            color=color, zorder=2, linewidth=0,
        ))
        ax.plot([i, i], [l, h], color=color, lw=0.7, zorder=2)


def _draw_ssa_colored(ax: plt.Axes, df: pd.DataFrame) -> None:
    xs = np.arange(len(df))
    ys = df["ssa_40"].to_numpy(dtype=float)
    labels = df["trend_label_ja"].to_list()

    # Build segments: each segment is a pair of consecutive points with the same color
    segs_by_label: dict[str, list] = {lbl: [] for lbl in LABEL_COLOR}

    for i in range(len(xs) - 1):
        lbl = labels[i]
        if lbl in segs_by_label and not (np.isnan(ys[i]) or np.isnan(ys[i + 1])):
            segs_by_label[lbl].append([(xs[i], ys[i]), (xs[i + 1], ys[i + 1])])

    for lbl, segs in segs_by_label.items():
        if segs:
            lc = LineCollection(segs, colors=LABEL_COLOR[lbl], linewidths=2.5, zorder=3)
            ax.add_collection(lc)


def _build_xtick_labels(df: pd.DataFrame, interval_months: int = 4) -> tuple[list[int], list[str]]:
    ticks, labels = [], []
    prev_period = None
    for i, dt in enumerate(df.index):
        period = (dt.year, (dt.month - 1) // interval_months)
        if period != prev_period:
            ticks.append(i)
            labels.append(dt.strftime("%Y/%m"))
            prev_period = period
    return ticks, labels


def _draw_trades(ax: plt.Axes, df: pd.DataFrame, trades_df: pd.DataFrame) -> None:
    """トレードの保有期間シェード・エントリ/エグジットマーカー・リターンラベルを描画"""
    date_to_idx = {d.date(): i for i, d in enumerate(df.index)}
    ymin, ymax = ax.get_ylim()
    price_range = ymax - ymin
    offset = price_range * 0.018   # マーカーを価格から少しずらす量

    for _, t in trades_df.iterrows():
        ei = date_to_idx.get(t["entry_date"])
        xi = date_to_idx.get(t["exit_date"])
        if ei is None or xi is None:
            continue

        is_long = t["strategy"] == "long"
        is_win  = t["return_pct"] > 0
        shade   = "#2ecc71" if is_long else "#e74c3c"
        mk_clr  = "#00ff88" if (is_long and is_win) else (
                  "#ffdd44" if (is_long and not is_win) else (
                  "#ff6666" if (not is_long and is_win) else "#ffaa44"))

        # 保有期間を薄くシェード
        ax.axvspan(ei - 0.5, xi + 0.5, alpha=0.07, color=shade, zorder=0)

        ep = float(df.iloc[ei]["low"])
        xp = float(df.iloc[xi]["high"])

        if is_long:
            # ロング: エントリー=下向き矢印(↓買い)、エグジット=上向き矢印(↑売り)
            ax.annotate("", xy=(ei, ep - offset * 0.3), xytext=(ei, ep - offset * 2.0),
                        arrowprops=dict(arrowstyle="->", color="#44ff88", lw=1.4), zorder=6)
            ax.annotate("", xy=(xi, xp + offset * 0.3), xytext=(xi, xp + offset * 2.0),
                        arrowprops=dict(arrowstyle="->", color="#ffdd44", lw=1.4), zorder=6)
        else:
            # ショート: エントリー=上向き矢印(↑売り建て)、エグジット=下向き矢印(↓買い戻し)
            ax.annotate("", xy=(ei, xp + offset * 0.3), xytext=(ei, xp + offset * 2.0),
                        arrowprops=dict(arrowstyle="->", color="#ff6666", lw=1.4), zorder=6)
            ax.annotate("", xy=(xi, ep - offset * 0.3), xytext=(xi, ep - offset * 2.0),
                        arrowprops=dict(arrowstyle="->", color="#ffaa44", lw=1.4), zorder=6)

        # リターンラベル（エグジット付近に表示）
        ret = t["return_pct"]
        label_y = xp + offset * 3.5 if is_long else ep - offset * 3.5
        va = "bottom" if is_long else "top"
        ax.text(xi, label_y, f"{ret:+.1f}%", color=mk_clr,
                fontsize=6.5, ha="center", va=va, zorder=7,
                fontweight="bold")


def _build_equity_curve(df: pd.DataFrame, trades_df: pd.DataFrame) -> np.ndarray:
    """各日付ごとの累積損益倍率を返す（保有中は線形補間、待機中は前値維持）"""
    date_to_idx = {d.date(): i for i, d in enumerate(df.index)}
    equity = np.ones(len(df))

    running = 1.0
    # 時系列順にソート
    for _, t in trades_df.sort_values("entry_date").iterrows():
        ei = date_to_idx.get(t["entry_date"])
        xi = date_to_idx.get(t["exit_date"])
        if ei is None or xi is None:
            continue
        end_val = running * (1 + t["return_pct"] / 100)
        if xi > ei:
            # 保有中は価格ベースで線形補間
            for j in range(ei, xi + 1):
                frac = (j - ei) / (xi - ei)
                equity[j] = running + (end_val - running) * frac
        else:
            equity[xi] = end_val
        # エグジット以降は新しい running 値を維持
        for j in range(xi + 1, len(equity)):
            equity[j] = end_val
        running = end_val

    return equity


def draw_chart(df: pd.DataFrame, trades_df: pd.DataFrame | None,
               period_name: str, out_path: Path,
               equity_label: str = "戦略") -> None:
    setup_japanese_font()

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(20, 14),
        gridspec_kw={"height_ratios": [6, 2, 2]},
        facecolor="#1e1e2e",
    )
    fig.subplots_adjust(hspace=0.06)

    for ax in (ax1, ax2, ax3):
        ax.set_facecolor("#1e1e2e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

    # ── Candlesticks + SSA ────────────────────────────────────────────────────
    _draw_candlesticks(ax1, df)
    _draw_ssa_colored(ax1, df)

    xs = np.arange(len(df))
    ax1.set_xlim(-1, len(df))
    all_prices = pd.concat([df["high"], df["low"], df["ssa_40"].dropna()])
    price_margin = (all_prices.max() - all_prices.min()) * 0.06
    ax1.set_ylim(all_prices.min() - price_margin, all_prices.max() + price_margin)
    ax1.tick_params(colors="#cccccc", labelsize=9)
    ax1.yaxis.set_tick_params(labelcolor="#cccccc")
    ax1.set_xticklabels([])

    # トレンドラベル凡例
    legend_handles = [
        Line2D([0], [0], color=LABEL_COLOR[lbl], lw=2.5, label=lbl)
        for lbl in ALL_LABELS
    ]
    # トレード凡例を追加
    if trades_df is not None and not trades_df.empty:
        legend_handles += [
            Line2D([0], [0], color="#44ff88", lw=0, marker="^", ms=7, label="ロング買い"),
            Line2D([0], [0], color="#ffdd44", lw=0, marker="v", ms=7, label="ロング売り"),
            Line2D([0], [0], color="#ff6666", lw=0, marker="v", ms=7, label="ショート売り"),
            Line2D([0], [0], color="#ffaa44", lw=0, marker="^", ms=7, label="ショート買戻し"),
        ]
    ax1.legend(handles=legend_handles, loc="upper left", fontsize=8,
               facecolor="#2a2a3e", edgecolor="#555577", labelcolor="white",
               framealpha=0.85, ncol=2)

    start_str = df.index[0].strftime("%Y/%m/%d")
    end_str   = df.index[-1].strftime("%Y/%m/%d")
    ax1.set_title(
        f"日経225  Rolling SSA(W={W}, L={L}, R={R})  トレンドラベル + トレード履歴  {start_str} -{end_str}",
        color="white", fontsize=12, pad=10,
    )
    ax1.set_ylabel("終値", color="#cccccc", fontsize=10)

    # トレード描画（価格軸確定後に呼ぶ）
    if trades_df is not None and not trades_df.empty:
        _draw_trades(ax1, df, trades_df)

    # ── slope_5 subplot ───────────────────────────────────────────────────────
    slope5 = df["ssa_slope_5"].to_numpy(dtype=float)
    ax2.plot(xs, slope5, color="#8888cc", lw=1.2, zorder=2)
    ax2.axhline(0, color="#666688", lw=0.8, linestyle="--", zorder=1)
    ax2.fill_between(xs, slope5, 0, where=(slope5 > 0), color="#2ecc71", alpha=0.25, zorder=1)
    ax2.fill_between(xs, slope5, 0, where=(slope5 < 0), color="#e74c3c", alpha=0.25, zorder=1)
    ax2.plot(xs, np.full_like(xs, UP_THRESHOLD,   dtype=float), color="#2ecc71", lw=0.7, linestyle=":", alpha=0.6)
    ax2.plot(xs, np.full_like(xs, DOWN_THRESHOLD, dtype=float), color="#e74c3c", lw=0.7, linestyle=":", alpha=0.6)
    ax2.set_xlim(-1, len(df))
    ax2.set_xticklabels([])
    ax2.tick_params(colors="#cccccc", labelsize=8)
    ax2.yaxis.set_tick_params(labelcolor="#cccccc")
    ax2.set_ylabel("SSA slope(5d)", color="#cccccc", fontsize=9)

    # ── 累積損益 subplot ──────────────────────────────────────────────────────
    if trades_df is not None and not trades_df.empty:
        equity = _build_equity_curve(df, trades_df)
        equity_pct = (equity - 1) * 100
        bnh_pct = (df["close"].to_numpy(dtype=float) / float(df["close"].iloc[0]) - 1) * 100

        ax3.plot(xs, equity_pct, color="#f39c12", lw=2.0, zorder=3, label=equity_label)
        ax3.plot(xs, bnh_pct,    color="#8888cc", lw=1.0, zorder=2, linestyle="--", label="Buy&Hold", alpha=0.7)
        ax3.axhline(0, color="#666688", lw=0.7, linestyle="--", zorder=1)
        ax3.fill_between(xs, equity_pct, 0,
                         where=(equity_pct >= 0), color="#f39c12", alpha=0.15, zorder=1)
        ax3.fill_between(xs, equity_pct, 0,
                         where=(equity_pct < 0),  color="#e74c3c", alpha=0.20, zorder=1)
        ax3.set_xlim(-1, len(df))
        ax3.tick_params(colors="#cccccc", labelsize=8)
        ax3.yaxis.set_tick_params(labelcolor="#cccccc")
        ax3.set_ylabel("累積損益 (%)", color="#cccccc", fontsize=9)
        ax3.legend(loc="upper left", fontsize=8, facecolor="#2a2a3e",
                   edgecolor="#555577", labelcolor="white", framealpha=0.85)

    ticks, tick_labels = _build_xtick_labels(df, interval_months=4)
    ax3.set_xticks(ticks)
    ax3.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=8, color="#cccccc")

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[CHART] saved {out_path}")


# ── Backtest ──────────────────────────────────────────────────────────────────

def build_trades(df: pd.DataFrame) -> pd.DataFrame:
    """上昇トレンド(ロング)・下降トレンド(ショート)の各ブロックを1トレードとして抽出
    エントリー: シグナル発生翌日の始値
    エグジット: ラベルブロック最終日の終値
    """
    df2 = df.copy()
    df2["run_id"] = (df2["trend_label_en"] != df2["trend_label_en"].shift()).cumsum()

    rows = []
    for target_en, direction, strategy in [
        ("uptrend",   1,  "long"),
        ("downtrend", -1, "short"),
    ]:
        for _, grp in df2[df2["trend_label_en"] == target_en].groupby("run_id"):
            signal_iloc = df.index.get_loc(grp.index[0])
            if signal_iloc + 1 >= len(df):
                continue
            next_row = df.iloc[signal_iloc + 1]
            entry_dt = next_row.name
            if entry_dt > grp.index[-1]:
                continue  # 1日ブロックでエントリー日がブロック外になる場合はスキップ
            ep  = float(next_row["open"])
            xp  = float(grp["close"].iloc[-1])
            ret = (xp / ep - 1) * 100 * direction
            rows.append(dict(
                strategy    = strategy,
                label       = target_en,
                entry_date  = entry_dt.date(),
                exit_date   = grp.index[-1].date(),
                hold_days   = len(df[(df.index >= entry_dt) & (df.index <= grp.index[-1])]),
                entry_price = round(ep, 0),
                exit_price  = round(xp, 0),
                return_pct  = round(ret, 2),
                win         = ret > 0,
            ))
    return pd.DataFrame(rows).sort_values("entry_date").reset_index(drop=True)


def _zone_trades(df: pd.DataFrame, pos_labels: list[str], direction: int, strategy: str) -> list[dict]:
    """連続する pos_labels のブロックを1トレードとして抽出するヘルパー
    エントリー: シグナル発生翌日の始値
    エグジット: ゾーン最終日の終値
    """
    df2 = df.copy()
    df2["in_zone"] = df2["trend_label_en"].isin(pos_labels)
    df2["zone_run"] = (df2["in_zone"] != df2["in_zone"].shift()).cumsum()
    rows = []
    for _, grp in df2[df2["in_zone"]].groupby("zone_run"):
        signal_iloc = df.index.get_loc(grp.index[0])
        if signal_iloc + 1 >= len(df):
            continue
        next_row = df.iloc[signal_iloc + 1]
        entry_dt = next_row.name
        if entry_dt > grp.index[-1]:
            continue
        ep  = float(next_row["open"])
        xp  = float(grp["close"].iloc[-1])
        ret = (xp / ep - 1) * 100 * direction
        rows.append(dict(
            strategy     = strategy,
            entry_date   = entry_dt.date(),
            exit_date    = grp.index[-1].date(),
            hold_days    = len(df[(df.index >= entry_dt) & (df.index <= grp.index[-1])]),
            entry_price  = round(ep, 0),
            exit_price   = round(xp, 0),
            return_pct   = round(ret, 2),
            win          = ret > 0,
            entry_slope5 = round(float(grp["ssa_slope_5"].iloc[0]), 5),
        ))
    return rows


def build_strategy1_trades(df: pd.DataFrame) -> pd.DataFrame:
    """①転換期エントリー/エグジット + L+S
    ロング: 上昇転換期 or 上昇トレンド のブロック全体を保有
    ショート: 下降転換期 or 下降トレンド のブロック全体を保有
    """
    rows = (
        _zone_trades(df, ["transition_to_uptrend", "uptrend"],   1, "long")
      + _zone_trades(df, ["transition_to_downtrend", "downtrend"], -1, "short")
    )
    return pd.DataFrame(rows).sort_values("entry_date").reset_index(drop=True)


def build_strategy2_trades(df: pd.DataFrame) -> pd.DataFrame:
    """②ロング専用（ショートなし）
    上昇トレンドブロックのみ保有（エントリー/エグジット基準は旧来と同じ）
    """
    rows = _zone_trades(df, ["uptrend"], 1, "long")
    return pd.DataFrame(rows).sort_values("entry_date").reset_index(drop=True)


def build_strategy3_trades(df: pd.DataFrame, sf: float = SLOPE_FILTER) -> pd.DataFrame:
    """③スロープフィルター付き L+S
    旧来のエントリー基準（uptrend/downtrend）に加え、
    エントリー時 |slope5| >= sf のトレードのみ実施
    エントリー: シグナル発生翌日の始値
    """
    df2 = df.copy()
    df2["run_id"] = (df2["trend_label_en"] != df2["trend_label_en"].shift()).cumsum()
    rows = []
    for target_en, direction, strategy in [
        ("uptrend",   1,  "long"),
        ("downtrend", -1, "short"),
    ]:
        for _, grp in df2[df2["trend_label_en"] == target_en].groupby("run_id"):
            s5 = float(grp["ssa_slope_5"].iloc[0])
            if abs(s5) < sf:
                continue
            signal_iloc = df.index.get_loc(grp.index[0])
            if signal_iloc + 1 >= len(df):
                continue
            next_row = df.iloc[signal_iloc + 1]
            entry_dt = next_row.name
            if entry_dt > grp.index[-1]:
                continue
            ep  = float(next_row["open"])
            xp  = float(grp["close"].iloc[-1])
            ret = (xp / ep - 1) * 100 * direction
            rows.append(dict(
                strategy     = strategy,
                entry_date   = entry_dt.date(),
                exit_date    = grp.index[-1].date(),
                hold_days    = len(df[(df.index >= entry_dt) & (df.index <= grp.index[-1])]),
                entry_price  = round(ep, 0),
                exit_price   = round(xp, 0),
                return_pct   = round(ret, 2),
                win          = ret > 0,
                entry_slope5 = round(s5, 5),
            ))
    return pd.DataFrame(rows).sort_values("entry_date").reset_index(drop=True)


# ── Grid Search (hysteresis long-only) ────────────────────────────────────────

W_GRID = [40, 50, 60, 80, 100]
TAU_ENTER_GRID = [0.0, 0.001, 0.002, 0.005]
TAU_EXIT_GRID = [0.0, 0.001, 0.002]
BULL_PERIODS = {"2013_2016", "2022_2025"}
RANGE_PERIODS = {"2009_2012", "2018_2021"}


def label_with_hysteresis(slope_long: np.ndarray, tau_enter: float, tau_exit: float) -> np.ndarray:
    """ssa_slope_20 にヒステリシスを適用し、ロングゾーン bool 配列を返す。
    state OUT,  slope > tau_enter   → state LONG
    state LONG, slope < -tau_exit   → state OUT
    それ以外（NaN含む）はステート維持。
    """
    n = len(slope_long)
    state = np.zeros(n, dtype=bool)
    cur = False
    for i in range(n):
        s = slope_long[i]
        if not np.isnan(s):
            if cur:
                if s < -tau_exit:
                    cur = False
            else:
                if s > tau_enter:
                    cur = True
        state[i] = cur
    return state


def backtest_long_zone(df: pd.DataFrame) -> pd.DataFrame:
    """in_long_zone(bool)に基づくロング専用バックテスト。
    エントリー: ゾーン入り日(state False→True)の翌日始値
    エグジット: ゾーン出る日(state True→False)の翌日始値
    期間最終日までゾーン継続中の場合は最終日終値でクローズ。
    """
    zone = df["in_long_zone"].to_numpy(dtype=bool)
    prev = np.r_[False, zone[:-1]]
    enter_idx = np.where(zone & ~prev)[0]
    exit_idx = np.where(~zone & prev)[0]
    dates = df.index
    opens = df["open"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    n = len(df)

    rows = []
    for ei in enter_idx:
        if ei + 1 >= n:
            continue
        # 次のエグジットシグナルを探す
        next_exits = exit_idx[exit_idx > ei]
        if len(next_exits) > 0:
            xi = int(next_exits[0])
            if xi + 1 < n:
                exit_dt = dates[xi + 1]
                xp = float(opens[xi + 1])
            else:
                exit_dt = dates[xi]
                xp = float(closes[xi])
        else:
            # 期末までゾーン継続 → 最終日終値クローズ
            xi = n - 1
            exit_dt = dates[xi]
            xp = float(closes[xi])
        entry_dt = dates[ei + 1]
        ep = float(opens[ei + 1])
        if entry_dt > exit_dt:
            continue
        ret = (xp / ep - 1) * 100
        hold_days = int(((dates >= entry_dt) & (dates <= exit_dt)).sum())
        rows.append(dict(
            strategy   = "long",
            entry_date = entry_dt.date(),
            exit_date  = exit_dt.date(),
            hold_days  = hold_days,
            entry_price= round(ep, 0),
            exit_price = round(xp, 0),
            return_pct = round(ret, 2),
            win        = ret > 0,
        ))
    return pd.DataFrame(rows)


def compute_metrics(trades_df: pd.DataFrame, df: pd.DataFrame) -> dict:
    bnh_pct = (float(df["close"].iloc[-1]) / float(df["close"].iloc[0]) - 1) * 100
    if trades_df is None or trades_df.empty:
        return dict(
            n_trades=0, win_rate=float("nan"), total_return_pct=0.0,
            bnh_pct=round(bnh_pct, 2), gap_vs_bnh=round(-bnh_pct, 2),
            max_dd=0.0, avg_hold_days=float("nan"),
        )
    compound = float((1 + trades_df["return_pct"] / 100).prod() - 1) * 100
    equity = _build_equity_curve(df, trades_df)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(dd.min()) * 100
    return dict(
        n_trades         = int(len(trades_df)),
        win_rate         = round(float(trades_df["win"].mean()), 3),
        total_return_pct = round(compound, 2),
        bnh_pct          = round(bnh_pct, 2),
        gap_vs_bnh       = round(compound - bnh_pct, 2),
        max_dd           = round(max_dd, 2),
        avg_hold_days    = round(float(trades_df["hold_days"].mean()), 1),
    )


def run_grid_search(full_df: pd.DataFrame, periods: list[tuple[str, str, str]],
                    w_grid: list[int], tau_enter_grid: list[float],
                    tau_exit_grid: list[float], r: int = R,
                    warmup_calendar_days: int = WARMUP_CALENDAR_DAYS) -> tuple[pd.DataFrame, dict]:
    """グリッドサーチ本体。
    Returns:
        results_df: 各 (period, W, tau_e, tau_x) ごとの行
        zones_by_key: {(period_tag, W, tau_e, tau_x): (df_period, trades_df)} 後段のチャート用
    """
    results = []
    zones_by_key = {}

    for start, end, tag in periods:
        period_start = pd.Timestamp(start)
        period_end   = pd.Timestamp(end)
        max_w = max(w_grid)
        # ウォームアップは calendar days と W*1.5 取引日相当の大きい方
        warmup_days = max(warmup_calendar_days, int(max_w * 2))
        warmup_start = period_start - pd.Timedelta(days=warmup_days)
        df_work = full_df[(full_df.index >= warmup_start) & (full_df.index <= period_end)].copy()
        if df_work.empty:
            continue
        closes_arr = df_work["close"].astype(float).to_numpy()

        # W ごとに SSA をキャッシュ
        ssa_by_w = {}
        for w in w_grid:
            l = w // 2
            ssa_by_w[w] = compute_ssa(closes_arr, w, l, r)

        for w in w_grid:
            l = w // 2
            ssa = ssa_by_w[w]
            slope_long = pd.Series(ssa, index=df_work.index).pct_change(20).to_numpy()
            for tau_e in tau_enter_grid:
                for tau_x in tau_exit_grid:
                    in_zone_full = label_with_hysteresis(slope_long, tau_e, tau_x)
                    df_full = df_work.copy()
                    df_full["ssa_40"] = ssa  # チャート流用のため固定列名
                    df_full["ssa_slope_5"] = pd.Series(ssa, index=df_work.index).pct_change(5).to_numpy()
                    df_full["ssa_slope_20"] = slope_long
                    df_full["in_long_zone"] = in_zone_full
                    df_period = df_full[(df_full.index >= period_start) & (df_full.index <= period_end)].copy()
                    if df_period.empty:
                        continue
                    trades = backtest_long_zone(df_period)
                    metrics = compute_metrics(trades, df_period)
                    row = dict(period=tag, W=w, L=l, tau_enter=tau_e, tau_exit=tau_x, **metrics)
                    results.append(row)
                    zones_by_key[(tag, w, tau_e, tau_x)] = (df_period, trades)

    return pd.DataFrame(results), zones_by_key


def rank_configs(grid_df: pd.DataFrame) -> pd.DataFrame:
    """構成ごとに 4 期間の gap を集計してランキング DataFrame を返す。"""
    base = (grid_df
            .groupby(["W", "L", "tau_enter", "tau_exit"])
            .agg(
                worst_gap         = ("gap_vs_bnh", "min"),
                avg_gap           = ("gap_vs_bnh", "mean"),
                avg_total_return  = ("total_return_pct", "mean"),
                avg_n_trades      = ("n_trades", "mean"),
                avg_win_rate      = ("win_rate", "mean"),
                avg_max_dd        = ("max_dd", "mean"),
                avg_hold_days     = ("avg_hold_days", "mean"),
            )
            .reset_index())
    bull = (grid_df[grid_df["period"].isin(BULL_PERIODS)]
            .groupby(["W", "L", "tau_enter", "tau_exit"])
            .agg(bull_gap=("gap_vs_bnh", "mean"))
            .reset_index())
    rng = (grid_df[grid_df["period"].isin(RANGE_PERIODS)]
           .groupby(["W", "L", "tau_enter", "tau_exit"])
           .agg(range_gap=("gap_vs_bnh", "mean"))
           .reset_index())
    summary = base.merge(bull, on=["W","L","tau_enter","tau_exit"], how="left")
    summary = summary.merge(rng, on=["W","L","tau_enter","tau_exit"], how="left")
    # 数値整形
    for col in ["worst_gap","avg_gap","bull_gap","range_gap","avg_total_return",
                "avg_max_dd","avg_hold_days"]:
        summary[col] = summary[col].round(2)
    summary["avg_win_rate"] = summary["avg_win_rate"].round(3)
    summary["avg_n_trades"] = summary["avg_n_trades"].round(1)
    return summary.sort_values("worst_gap", ascending=False).reset_index(drop=True)


def print_summary(tag: str, label: str, tdf: pd.DataFrame, bnh: float) -> None:
    if tdf.empty:
        print(f"  [{label}] 取引なし")
        return
    compound = (1 + tdf["return_pct"] / 100).prod() - 1
    for st in ["long", "short"]:
        sub = tdf[tdf["strategy"] == st]
        if sub.empty:
            continue
        c = (1 + sub["return_pct"] / 100).prod() - 1
        print(f"  [{st}] 回数:{len(sub)}  勝率:{sub['win'].mean()*100:.1f}%  "
              f"平均:{sub['return_pct'].mean():.2f}%  累計:{c*100:.1f}%")
    print(f"  [合計] 取引:{len(tdf)}回  累計(複利):{compound*100:.1f}%  BnH:{bnh:.1f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[DATA] Loading Nikkei 225 from PostgreSQL...")
    full_df = load_nikkei_from_db()
    print(f"[DATA] {len(full_df)} rows  {full_df.index.min().date()} -{full_df.index.max().date()}")

    for start, end, tag in PERIODS:
        print(f"\n[SSA] Analyzing {start} to {end}...")
        df = analyze_period(full_df, start, end)

        dist = df["trend_label_ja"].value_counts()
        for lbl in ALL_LABELS:
            print(f"  {lbl}: {dist.get(lbl, 0)} days")

        csv_path = OUTPUT_DIR / f"nikkei_ssa_trend_{tag}.csv"
        df.reset_index().to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"[CSV] saved {csv_path} ({len(df)} rows)")

        bnh = (float(df["close"].iloc[-1]) / float(df["close"].iloc[0]) - 1) * 100

        slabel = "①転換期エントリー(L+S)"
        tdf = build_strategy1_trades(df)
        csv_p = OUTPUT_DIR / f"backtest_s1_{tag}.csv"
        tdf.to_csv(csv_p, index=False, encoding="utf-8-sig")
        print(f"\n--- {slabel} ---")
        print_summary(tag, slabel, tdf, bnh)
        chart_p = OUTPUT_DIR / f"nikkei_ssa_chart_s1_{tag}.png"
        long_tdf = tdf[tdf["strategy"] == "long"].reset_index(drop=True)
        draw_chart(df, long_tdf, tag, chart_p, equity_label="①転換期エントリー(ロングのみ表示)")

    # ── Grid Search ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[GRID] Running hysteresis long-only grid search")
    print(f"  W={W_GRID}  tau_enter={TAU_ENTER_GRID}  tau_exit={TAU_EXIT_GRID}")
    print(f"  configs={len(W_GRID) * len(TAU_ENTER_GRID) * len(TAU_EXIT_GRID)}"
          f"  periods={len(PERIODS)}")
    print("=" * 70)

    grid_dir = OUTPUT_DIR / "grid_search"
    grid_dir.mkdir(parents=True, exist_ok=True)

    grid_df, zones_by_key = run_grid_search(
        full_df, PERIODS, W_GRID, TAU_ENTER_GRID, TAU_EXIT_GRID
    )
    grid_csv = grid_dir / "grid_results_all.csv"
    grid_df.to_csv(grid_csv, index=False, encoding="utf-8-sig")
    print(f"[GRID] saved {grid_csv}  rows={len(grid_df)}")

    summary_df = rank_configs(grid_df)
    summary_csv = grid_dir / "grid_summary_by_config.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"[GRID] saved {summary_csv}  configs={len(summary_df)}")

    top10 = summary_df.head(10)
    top10_csv = grid_dir / "grid_top10_worst_gap.csv"
    top10.to_csv(top10_csv, index=False, encoding="utf-8-sig")
    print(f"[GRID] saved {top10_csv}")

    print("\n[GRID] Top 10 by worst_gap (= 4期間中の最悪 gap_vs_bnh が最も大きい構成)")
    print(top10[["W","tau_enter","tau_exit","worst_gap","avg_gap","bull_gap",
                 "range_gap","avg_total_return","avg_win_rate","avg_n_trades"]]
          .to_string(index=False))

    # ── Render charts for total winner across 4 periods ──────────────────────
    winner = summary_df.iloc[0]
    w_w, w_te, w_tx = int(winner["W"]), float(winner["tau_enter"]), float(winner["tau_exit"])
    print(f"\n[GRID] Winner: W={w_w}  tau_enter={w_te}  tau_exit={w_tx}  "
          f"worst_gap={winner['worst_gap']}  avg_gap={winner['avg_gap']}")

    for start, end, tag in PERIODS:
        key = (tag, w_w, w_te, w_tx)
        if key not in zones_by_key:
            continue
        df_p, trades = zones_by_key[key]
        # in_long_zone を 2 色のラベルに変換して既存の draw_chart を流用
        df_p = df_p.copy()
        df_p["trend_label_ja"] = np.where(df_p["in_long_zone"], LBL_UP, LBL_DOWN)
        equity_label = f"Grid勝者(W={w_w}, τe={w_te}, τx={w_tx})"
        chart_p = grid_dir / f"nikkei_ssa_chart_best_{tag}.png"
        draw_chart(df_p, trades, tag, chart_p, equity_label=equity_label)

        # 勝者のトレード履歴 CSV
        trades_csv = grid_dir / f"backtest_best_{tag}.csv"
        trades.to_csv(trades_csv, index=False, encoding="utf-8-sig")
        print(f"[GRID] saved {trades_csv}")

    print("\n[DONE] All outputs written to", OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

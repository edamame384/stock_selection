"""
週次セクター別 年初来高値更新銘柄数の分析・プロット

ローカルの価格CSV (data/prices/) とセクターマスター (data/sector_master_template.csv) を使い、
各週で年初来高値 (YTD high) を更新した銘柄数をセクター別に集計して積み上げ棒グラフを出力する。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# src.stock_signal から借用した最小限のユーティリティ
# (yfinance 未インストール環境でも動作するように直接実装)
# ---------------------------------------------------------------------------

def normalize_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if symbol.startswith("TYO:"):
        symbol = symbol.split(":", 1)[1]
    if symbol.endswith(".T"):
        return symbol
    return f"{symbol}.T"


def load_sector_master(path: Path) -> dict[str, str]:
    df = pd.read_csv(path)
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        sym_raw = str(row["symbol"]).strip()
        sec = str(row["sector"]).strip()
        if not sym_raw or not sec:
            continue
        out[normalize_symbol(sym_raw)] = sec
    return out


# ---------------------------------------------------------------------------
# 日本語フォント検出
# ---------------------------------------------------------------------------

def _detect_japanese_font() -> str | None:
    """利用可能な日本語フォントを探す。見つからなければ None を返す。"""
    candidates = [
        "IPAGothic", "IPAPGothic", "Noto Sans CJK JP", "Noto Sans JP",
        "TakaoGothic", "VL Gothic", "MS Gothic",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None


# ---------------------------------------------------------------------------
# セクターマスター読込
# ---------------------------------------------------------------------------

def load_sector_map(master_path: Path) -> dict[str, str]:
    """セクターマスターを読み込み、セクターが '-' のエントリ (ETF等) を除外する。

    Returns
    -------
    dict[str, str]
        正規化済みシンボル -> セクター名 のマッピング
    """
    raw = load_sector_master(master_path)
    return {sym: sec for sym, sec in raw.items() if sec and sec != "-"}


# ---------------------------------------------------------------------------
# 銘柄ごとの年初来高値更新日を取得
# ---------------------------------------------------------------------------

def compute_ytd_high_updates(symbol: str, price_path: Path, year: int) -> list[tuple[int, str]]:
    """指定銘柄について対象年に年初来高値を更新した週番号リストを返す。

    Parameters
    ----------
    symbol : str
        正規化済みシンボル (例: '1301.T')
    price_path : Path
        価格CSVファイルのパス
    year : int
        対象年

    Returns
    -------
    list[tuple[int, str]]
        (week_number, symbol) のリスト。各週は重複排除済み。
    """
    try:
        df = pd.read_csv(
            price_path,
            usecols=["Date", "High", "Volume"],
            parse_dates=["Date"],
        )
    except Exception:
        return []

    # 対象年のデータのみ抽出
    df = df[df["Date"].dt.year == year].copy()
    if df.empty:
        return []

    # Volume=0 の行を除外 (休日プレースホルダ)
    df = df[df["Volume"] > 0]
    if df.empty:
        return []

    df = df.sort_values("Date").reset_index(drop=True)

    # 当日を含む YTD high の cummax を計算し、1日シフトして「前日までの最高値」を得る
    df["ytd_high_prev"] = df["High"].cummax().shift(1)
    # 初日は比較対象なし → YTD高値更新とみなす
    df["is_new_ytd_high"] = (df["High"] >= df["ytd_high_prev"]) | df["ytd_high_prev"].isna()

    ytd_days = df[df["is_new_ytd_high"]].copy()
    if ytd_days.empty:
        return []

    # 週番号: 年初からの経過日数ベース (ISO週の年境界問題を回避)
    day_of_year = ytd_days["Date"].dt.dayofyear
    ytd_days["week"] = ((day_of_year - 1) // 7 + 1).clip(upper=52)

    # 各週で重複排除 (同一週に複数回更新しても1回のみカウント)
    weeks_seen: set[int] = set()
    result: list[tuple[int, str]] = []
    for week in ytd_days["week"]:
        if week not in weeks_seen:
            weeks_seen.add(week)
            result.append((int(week), symbol))

    return result


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="週次セクター別 年初来高値更新銘柄数をプロットする"
    )
    parser.add_argument("--year", type=int, default=2025, help="対象年 (デフォルト: 2025)")
    parser.add_argument(
        "--price-dir",
        type=Path,
        default=ROOT_DIR / "data" / "prices",
        help="価格CSVディレクトリ",
    )
    parser.add_argument(
        "--sector-master",
        type=Path,
        default=ROOT_DIR / "data" / "sector_master_template.csv",
        help="セクターマスターCSVパス",
    )
    parser.add_argument("--top-n", type=int, default=10, help="上位表示セクター数 (デフォルト: 10)")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="出力PNGパス (デフォルト: data/ytd_high_by_sector_{year}.png)",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="出力CSVパス (デフォルト: data/ytd_high_by_sector_{year}.csv)",
    )
    args = parser.parse_args()

    year: int = args.year
    price_dir: Path = args.price_dir
    top_n: int = args.top_n
    output_png: Path = args.output or ROOT_DIR / "data" / f"ytd_high_by_sector_{year}.png"
    output_csv: Path = args.csv_output or ROOT_DIR / "data" / f"ytd_high_by_sector_{year}.csv"

    # --- セクターマスター読込 ---
    print(f"[1/4] セクターマスターを読み込み中: {args.sector_master}")
    sector_map = load_sector_map(args.sector_master)
    print(f"      対象銘柄数: {len(sector_map)} (ETF等を除外済み)")

    # --- 各銘柄の年初来高値更新日を収集 ---
    print(f"[2/4] {year}年の年初来高値更新日を収集中 ...")
    records: list[dict[str, object]] = []
    skipped = 0
    total = len(sector_map)

    for idx, (symbol, sector) in enumerate(sector_map.items(), start=1):
        # ファイル名: 1301.T -> 1301_T.csv
        fname = symbol.replace(".", "_") + ".csv"
        price_path = price_dir / fname

        if not price_path.exists():
            skipped += 1
            continue

        updates = compute_ytd_high_updates(symbol, price_path, year)
        for week, sym in updates:
            records.append({"symbol": sym, "sector": sector, "week": week})

        if idx % 500 == 0 or idx == total:
            print(f"      処理済み: {idx}/{total}  スキップ: {skipped}")

    if not records:
        print("ERROR: 年初来高値更新の記録が見つかりませんでした。")
        return 1

    print(f"      完了。スキップ銘柄数: {skipped}")

    result_df = pd.DataFrame(records)
    print(f"      記録数: {len(result_df)}  週数: {result_df['week'].nunique()}  セクター数: {result_df['sector'].nunique()}")

    # --- セクター×週 ピボット ---
    print("[3/4] 集計・プロット準備中 ...")
    pivot = result_df.groupby(["week", "sector"]).size().unstack(fill_value=0)

    # 全52週のインデックスを確保
    all_weeks = range(1, 53)
    pivot = pivot.reindex(all_weeks, fill_value=0)

    # Top-N セクター選択
    sector_totals = pivot.sum(axis=0).sort_values(ascending=False)
    top_sectors = sector_totals.head(top_n).index.tolist()
    other_sectors = [c for c in pivot.columns if c not in top_sectors]

    plot_df = pivot[top_sectors].copy()
    if other_sectors:
        plot_df["その他"] = pivot[other_sectors].sum(axis=1)

    # CSV保存
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    plot_df.to_csv(output_csv)
    print(f"      CSVを保存しました: {output_csv}")

    # --- プロット ---
    print("[4/4] グラフを生成中 ...")
    jp_font = _detect_japanese_font()
    if jp_font:
        plt.rcParams["font.family"] = jp_font
        use_japanese = True
    else:
        use_japanese = False
        print("      警告: 日本語フォントが見つかりません。ラベルを英語表記にします。")

    # カラーパレット (Top-10 + その他 = 最大11色)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(plot_df.columns))]
    # 「その他」は灰色
    if "その他" in plot_df.columns:
        colors[-1] = (0.7, 0.7, 0.7, 1.0)

    fig, ax = plt.subplots(figsize=(16, 8))
    plot_df.plot(kind="bar", stacked=True, ax=ax, width=0.85, color=colors)

    if use_japanese:
        ax.set_title(f"セクター別 年初来高値更新銘柄数の推移 ({year}年)", fontsize=14, pad=12)
        ax.set_xlabel("週番号", fontsize=11)
        ax.set_ylabel("年初来高値更新銘柄数", fontsize=11)
    else:
        ax.set_title(f"Weekly YTD-High Update Count by Sector ({year})", fontsize=14, pad=12)
        ax.set_xlabel("Week Number", fontsize=11)
        ax.set_ylabel("Number of Stocks Hitting YTD Highs", fontsize=11)

    # x軸ラベル: 4週ごとに表示
    tick_positions = list(range(0, 52, 4)) + [51]
    tick_labels = [str(w + 1) for w in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0)

    # 凡例をプロット右外側に
    ax.legend(
        title="セクター" if use_japanese else "Sector",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize=9,
    )

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      グラフを保存しました: {output_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

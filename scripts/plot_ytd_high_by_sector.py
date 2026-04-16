"""
週次セクター別 年初来高値更新銘柄数の分析・プロット

ローカルの価格CSV (data/prices/) とセクターマスターを使い、
各週で年初来高値 (YTD high) を更新した銘柄数をセクター別に集計して折れ線グラフを出力する。

セクター分類:
  fine  (デフォルト): 四季報 scored_universe.csv の小分類 (92カテゴリ)
  coarse           : 東証33業種 sector_master_template.csv
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
    """東証33業種マスターを読み込み、セクターが '-' のエントリ (ETF等) を除外する。

    Returns
    -------
    dict[str, str]
        正規化済みシンボル -> セクター名 のマッピング
    """
    raw = load_sector_master(master_path)
    return {sym: sec for sym, sec in raw.items() if sec and sec != "-"}


def load_fine_sector_map(path: Path) -> dict[str, str]:
    """四季報 scored_universe.csv から細分類セクターマップを構築。

    simple_sector 列は '大分類/小分類' 形式 (例: '情報通信/SI・ソフトウエア開発')。
    '/' の右側（小分類）を抽出して返す。

    Returns
    -------
    dict[str, str]
        正規化済みシンボル -> 小分類セクター名 のマッピング
    """
    df = pd.read_csv(path, usecols=["ticker", "simple_sector"])
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        sym = normalize_symbol(str(row["ticker"]))
        raw = str(row["simple_sector"]).strip()
        if not raw or raw in ("nan", "-"):
            continue
        sub = raw.split("/", 1)[1].strip() if "/" in raw else raw
        if sub:
            out[sym] = sub
    return out


def build_sector_map(
    sector_master: Path,
    granularity: str,
    fine_sector_csv: Path,
) -> dict[str, str]:
    """セクターマップを構築する。

    fine モード: 四季報小分類を優先し、未収録銘柄は33業種にフォールバック。
    coarse モード: 33業種のみ。
    """
    coarse_map = load_sector_map(sector_master)
    if granularity == "coarse":
        return coarse_map

    fine_map = load_fine_sector_map(fine_sector_csv)
    combined: dict[str, str] = {}
    fine_hit = 0
    for sym, coarse_sec in coarse_map.items():
        if sym in fine_map:
            combined[sym] = fine_map[sym]
            fine_hit += 1
        else:
            combined[sym] = coarse_sec
    fallback = len(coarse_map) - fine_hit
    print(f"      細分類マッチ: {fine_hit}  フォールバック(33業種): {fallback}")
    return combined


# ---------------------------------------------------------------------------
# 銘柄ごとの年初来高値更新日を取得
# ---------------------------------------------------------------------------

def compute_ytd_high_updates(symbol: str, price_path: Path, year: int) -> list[tuple[int, str]]:
    """指定銘柄について対象年に年初来高値を更新した週番号リストを返す。

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

    df = df[df["Date"].dt.year == year].copy()
    if df.empty:
        return []

    # Volume=0 の行を除外 (休日プレースホルダ)
    df = df[df["Volume"] > 0]
    if df.empty:
        return []

    df = df.sort_values("Date").reset_index(drop=True)

    # 前日までのYTD最高値と比較して更新判定
    df["ytd_high_prev"] = df["High"].cummax().shift(1)
    df["is_new_ytd_high"] = (df["High"] >= df["ytd_high_prev"]) | df["ytd_high_prev"].isna()

    ytd_days = df[df["is_new_ytd_high"]].copy()
    if ytd_days.empty:
        return []

    # 週番号: 年初からの経過日数ベース (ISO週の年境界問題を回避)
    day_of_year = ytd_days["Date"].dt.dayofyear
    ytd_days = ytd_days.copy()
    ytd_days["week"] = ((day_of_year - 1) // 7 + 1).clip(upper=52)

    # 各週で重複排除
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
        help="東証33業種マスターCSVパス",
    )
    parser.add_argument(
        "--granularity",
        choices=["coarse", "fine"],
        default="fine",
        help="セクター粒度: coarse=33業種, fine=四季報小分類 (デフォルト: fine)",
    )
    parser.add_argument(
        "--fine-sector-csv",
        type=Path,
        default=ROOT_DIR / "projects" / "shikiho_text_parser" / "output" / "4q2_selection" / "4q2_scored_universe.csv",
        help="細分類セクターCSV (四季報 scored_universe)",
    )
    parser.add_argument("--top-n", type=int, default=10, help="週次/年間上位セクター数 (デフォルト: 10)")
    parser.add_argument(
        "--selection",
        choices=["annual", "weekly"],
        default="weekly",
        help=(
            "セクター選択方法: "
            "annual=年間累計Top-N, "
            "weekly=各週Top-N に --min-weeks 以上登場したセクター "
            "(デフォルト: weekly)"
        ),
    )
    parser.add_argument(
        "--min-weeks",
        type=int,
        default=2,
        help="selection=weekly のとき、Top-Nランクイン最低週数 (デフォルト: 2)",
    )
    parser.add_argument(
        "--ma-window",
        type=int,
        default=3,
        help="折れ線に適用する移動平均の週数 (0=平滑化なし、デフォルト: 3)",
    )
    parser.add_argument(
        "--y-metric",
        choices=["count", "ratio"],
        default="ratio",
        help="Y軸: count=銘柄数, ratio=セクター内での割合 (%) (デフォルト: ratio)",
    )
    parser.add_argument(
        "--show-other",
        action="store_true",
        default=False,
        help="折れ線グラフに「その他」集計線を表示する (デフォルト: 非表示)",
    )
    parser.add_argument(
        "--bucket-split",
        dest="bucket_split",
        action="store_true",
        default=True,
        help="selection=weekly のときランクイン週数でバケット分割プロット (デフォルト: 有効)",
    )
    parser.add_argument(
        "--no-bucket-split",
        dest="bucket_split",
        action="store_false",
        help="バケット分割を無効化し、単一グラフで全セクター表示",
    )
    parser.add_argument(
        "--buckets",
        type=str,
        default="40,20,10,2",
        help="バケット境界 (降順, カンマ区切り). デフォルト: '40,20,10,2'",
    )
    parser.add_argument(
        "--bucket-top-n",
        type=int,
        default=10,
        help="各バケット内で表示するセクター数の上限 (0=制限なし、デフォルト: 10)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="出力PNGパス (デフォルト: data/ytd_high_by_sector_{year}[_fine].png)",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="出力CSVパス (デフォルト: data/ytd_high_by_sector_{year}[_fine].csv)",
    )
    args = parser.parse_args()

    year: int = args.year
    price_dir: Path = args.price_dir
    top_n: int = args.top_n
    granularity_suffix = "_fine" if args.granularity == "fine" else ""
    selection_suffix = "_weekly" if args.selection == "weekly" else ""
    ma_suffix = f"_ma{args.ma_window}" if args.ma_window and args.ma_window > 1 else ""
    metric_suffix = "_ratio" if args.y_metric == "ratio" else ""
    suffix = f"{granularity_suffix}{selection_suffix}{metric_suffix}{ma_suffix}"
    output_png: Path = args.output or ROOT_DIR / "data" / f"ytd_high_by_sector_{year}{suffix}.png"
    output_csv: Path = args.csv_output or ROOT_DIR / "data" / f"ytd_high_by_sector_{year}{suffix}.csv"

    # --- セクターマップ構築 ---
    print(f"[1/4] セクターマップを構築中 (granularity={args.granularity}) ...")
    sector_map = build_sector_map(args.sector_master, args.granularity, args.fine_sector_csv)
    print(f"      対象銘柄数: {len(sector_map)}  ユニークセクター数: {len(set(sector_map.values()))}")

    # --- セクターサイズ (分母) を計算: 価格データが実在する銘柄のみカウント ---
    sector_size: dict[str, int] = {}
    for sym, sec in sector_map.items():
        fname = sym.replace(".", "_") + ".csv"
        if (price_dir / fname).exists():
            sector_size[sec] = sector_size.get(sec, 0) + 1

    # --- 各銘柄の年初来高値更新日を収集 ---
    print(f"[2/4] {year}年の年初来高値更新日を収集中 ...")
    records: list[dict[str, object]] = []
    skipped = 0
    total = len(sector_map)

    for idx, (symbol, sector) in enumerate(sector_map.items(), start=1):
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
    pivot = pivot.reindex(range(1, 53), fill_value=0)

    # --- y-metric=ratio のときはセクター選択前に割合へ変換 ---
    # (絶対数ベースだと大型セクターが常に Top-N に入り、中小セクターが拾えないため)
    # count_pivot は「その他」の正しい集計 (合計カウント ÷ 合計セクターサイズ) に使う
    count_pivot = pivot.copy()
    if args.y_metric == "ratio":
        pivot = pivot.astype(float)
        for sec in pivot.columns:
            denom = sector_size.get(sec, 0)
            if denom > 0:
                pivot[sec] = pivot[sec] / denom * 100.0
            else:
                pivot[sec] = 0.0

    # --- セクター選択 ---
    top_n_counts: dict[str, int] = {}
    if args.selection == "annual":
        # 年間累計でTop-N
        sector_totals = pivot.sum(axis=0).sort_values(ascending=False)
        selected_sectors = sector_totals.head(top_n).index.tolist()
        ordered = selected_sectors
        print(
            f"      選択方式: annual  累計Top{top_n} セクター数: {len(selected_sectors)}"
        )
    else:  # weekly
        # 第2週以降で各週Top-N にランクインした回数を集計
        weekly_pivot = pivot.loc[2:]
        for week, row in weekly_pivot.iterrows():
            if row.sum() == 0:
                continue
            weekly_topn = row.sort_values(ascending=False).head(top_n).index.tolist()
            for sec in weekly_topn:
                top_n_counts[sec] = top_n_counts.get(sec, 0) + 1

        # min-weeks 以上ランクインしたセクターのみ採用
        selected_sectors = [
            sec for sec, cnt in top_n_counts.items() if cnt >= args.min_weeks
        ]
        # ランクイン週数の多い順で並べる (同数なら年間累計順)
        annual_totals = pivot.sum(axis=0)
        ordered = sorted(
            selected_sectors,
            key=lambda s: (-top_n_counts[s], -annual_totals.get(s, 0)),
        )
        print(
            f"      選択方式: weekly  週次Top{top_n}に{args.min_weeks}週以上登場: "
            f"{len(selected_sectors)} セクター"
        )
        # ログ: 各セクターのランクイン週数
        for sec in ordered:
            print(f"        {sec}: {top_n_counts[sec]}週")

    other_sectors = [c for c in pivot.columns if c not in selected_sectors]
    plot_df = pivot[ordered].copy()
    if other_sectors:
        if args.y_metric == "ratio":
            # ratio では単純合計すると無意味なので、合計カウント ÷ 合計サイズ で算出
            other_denom = sum(sector_size.get(s, 0) for s in other_sectors)
            if other_denom > 0:
                plot_df["その他"] = (
                    count_pivot[other_sectors].sum(axis=1) / other_denom * 100.0
                )
            else:
                plot_df["その他"] = 0.0
        else:
            plot_df["その他"] = pivot[other_sectors].sum(axis=1)

    # CSV保存 (第1週含む全データ)
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

    # 第1週を除外してプロット (年始初日の全銘柄スパイクを回避)
    # 「その他」は折れ線グラフでは非表示がデフォルト (--show-other で有効化)
    cols_to_plot = list(plot_df.columns)
    if not args.show_other and "その他" in cols_to_plot:
        cols_to_plot = [c for c in cols_to_plot if c != "その他"]
    plot_display = plot_df.loc[2:, cols_to_plot]

    # 移動平均で平滑化 (min_periods=1 で先頭週も値を保持)
    if args.ma_window and args.ma_window > 1:
        plot_display = plot_display.rolling(
            window=args.ma_window, min_periods=1
        ).mean()
        ma_label = f"{args.ma_window}週移動平均" if True else ""
    else:
        ma_label = ""

    granularity_label = "細分類" if args.granularity == "fine" else "33業種"
    if args.selection == "weekly":
        selection_label = f"週次Top{top_n}に{args.min_weeks}週以上登場"
    else:
        selection_label = f"年間累計Top{top_n}"
    ma_suffix_jp = f"・{ma_label}" if ma_label else ""
    ma_suffix_en = f", MA={args.ma_window}w" if ma_label else ""
    metric_label_jp = "セクター内割合" if args.y_metric == "ratio" else "銘柄数"
    metric_label_en = "Ratio (%)" if args.y_metric == "ratio" else "Count"
    base_title = (
        f"セクター別 年初来高値更新{metric_label_jp}の推移 "
        f"({year}年・{granularity_label}・{selection_label}{ma_suffix_jp})"
        if use_japanese else
        f"Weekly YTD-High Update {metric_label_en} by Sector "
        f"({year}, {args.granularity}, selection={args.selection}{ma_suffix_en})"
    )

    # --- プロット分岐: バケット分割 or 単一グラフ ---
    use_bucket_split = (
        args.selection == "weekly" and args.bucket_split and len(plot_display.columns) > 0
    )

    if use_bucket_split:
        # ランクイン週数でバケット分割
        try:
            boundaries = sorted(
                (int(x) for x in args.buckets.split(",") if x.strip()),
                reverse=True,
            )
        except ValueError:
            print(f"ERROR: --buckets の形式が不正です: {args.buckets}")
            return 1

        # バケットを構築: [(label, [sectors])]
        # 例: boundaries=[40,20,10,2] → "40週以上", "20〜39週", "10〜19週", "2〜9週"
        buckets: list[tuple[str, list[str]]] = []
        # 「その他」は除外
        rankable_cols = [c for c in plot_display.columns if c != "その他"]
        for i, lower in enumerate(boundaries):
            upper = boundaries[i - 1] - 1 if i > 0 else None
            if use_japanese:
                label = f"{lower}週以上" if upper is None else f"{lower}〜{upper}週"
            else:
                label = f">={lower}w" if upper is None else f"{lower}-{upper}w"
            members = [
                c for c in rankable_cols
                if top_n_counts.get(c, 0) >= lower
                and (upper is None or top_n_counts.get(c, 0) <= upper)
            ]
            # 既に上位バケットに含まれているものを除外 (累積にならないように)
            used = {s for _, lst in buckets for s in lst}
            members = [m for m in members if m not in used]
            # ランクイン週数降順でソートし、上限で切る
            members.sort(key=lambda s: -top_n_counts.get(s, 0))
            if args.bucket_top_n > 0 and len(members) > args.bucket_top_n:
                members = members[: args.bucket_top_n]
            if members:
                buckets.append((label, members))

        n_buckets = len(buckets)
        if n_buckets == 0:
            print("WARN: 該当セクターがありません。単一グラフにフォールバックします。")
            use_bucket_split = False
        else:
            fig, axes = plt.subplots(
                n_buckets, 1,
                figsize=(16, 3.2 * n_buckets + 1.5),
                sharex=True,
            )
            if n_buckets == 1:
                axes = [axes]

            cmap = plt.get_cmap("tab10")
            for ax_i, (label, members) in zip(axes, buckets):
                for j, sec in enumerate(members):
                    n_size = sector_size.get(sec, 0)
                    if use_japanese:
                        leg = f"{sec} ({top_n_counts.get(sec, 0)}週/{n_size}銘柄)"
                    else:
                        leg = f"{sec} ({top_n_counts.get(sec, 0)}w/{n_size})"
                    ax_i.plot(
                        plot_display.index,
                        plot_display[sec],
                        color=cmap(j % 10),
                        marker="o",
                        markersize=4,
                        linewidth=1.6,
                        label=leg,
                    )
                bucket_title = (
                    f"ランクイン {label} ({len(members)}セクター)"
                    if use_japanese else
                    f"Rank-in {label} ({len(members)} sectors)"
                )
                ax_i.set_title(bucket_title, fontsize=11, loc="left", pad=6)
                if args.y_metric == "ratio":
                    y_unit = "割合 (%)" if use_japanese else "Ratio (%)"
                else:
                    y_unit = "銘柄数" if use_japanese else "Count"
                ax_i.set_ylabel(y_unit, fontsize=10)
                ax_i.grid(axis="both", linestyle="--", alpha=0.3)
                legend_ncol = 2 if len(members) >= 10 else 1
                ax_i.legend(
                    bbox_to_anchor=(1.01, 1),
                    loc="upper left",
                    borderaxespad=0,
                    fontsize=8,
                    ncol=legend_ncol,
                )

            # x軸ラベル (最下段のみ)
            tick_positions = list(range(2, 53, 4))
            if 52 not in tick_positions:
                tick_positions.append(52)
            axes[-1].set_xticks(tick_positions)
            axes[-1].set_xticklabels([str(w) for w in tick_positions], rotation=0)
            axes[-1].set_xlabel(
                "週番号" if use_japanese else "Week Number",
                fontsize=11,
            )

            fig.suptitle(base_title, fontsize=13, y=0.995)
            fig.tight_layout(rect=[0, 0, 1, 0.99])
            output_png.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_png, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"      バケット分割グラフを保存しました: {output_png}")
            return 0

    # --- 単一グラフ（従来動作） ---
    cmap = plt.get_cmap("tab20")
    n_cols = len(plot_display.columns)
    linestyles = ["-", "--"]
    markers = ["o", "s", "^", "D", "v"]

    colors: list[tuple] = []
    styles: list[str] = []
    marker_list: list[str] = []
    for i in range(n_cols):
        colors.append(cmap(i % 20))
        styles.append(linestyles[(i // 20) % len(linestyles)])
        marker_list.append(markers[i % len(markers)])

    if "その他" in plot_display.columns:
        idx = list(plot_display.columns).index("その他")
        colors[idx] = (0.6, 0.6, 0.6, 1.0)
        styles[idx] = ":"

    fig, ax = plt.subplots(figsize=(16, 9))
    for i, col in enumerate(plot_display.columns):
        ax.plot(
            plot_display.index,
            plot_display[col],
            color=colors[i],
            linestyle=styles[i],
            marker=marker_list[i],
            markersize=4,
            linewidth=1.4,
            label=col,
        )

    ax.set_title(base_title, fontsize=13, pad=12)
    ax.set_xlabel(
        "週番号" if use_japanese else "Week Number", fontsize=11,
    )
    if args.y_metric == "ratio":
        ylabel = (
            "年初来高値更新銘柄率 (%)" if use_japanese
            else "Share of Stocks Hitting YTD Highs (%)"
        )
    else:
        ylabel = (
            "年初来高値更新銘柄数" if use_japanese
            else "Number of Stocks Hitting YTD Highs"
        )
    ax.set_ylabel(ylabel, fontsize=11)

    tick_positions = list(range(2, 53, 4))
    if 52 not in tick_positions:
        tick_positions.append(52)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(w) for w in tick_positions], rotation=0)

    ax.grid(axis="both", linestyle="--", alpha=0.3)

    legend_ncol = 2 if n_cols >= 12 else 1
    ax.legend(
        title="セクター" if use_japanese else "Sector",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize=9,
        ncol=legend_ncol,
    )

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      グラフを保存しました: {output_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

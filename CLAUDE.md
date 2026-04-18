# Stock Selection - セッションコンテキスト

## プロジェクト概要

日本株の年初来高値更新銘柄をセクター別に週次分析・可視化するツール群。

## 主要スクリプト

### `scripts/plot_ytd_high_by_sector.py`

週次セクター別の年初来高値更新分析・プロットスクリプト。以下の機能を持つ:

- **セクター分類**: 四季報小分類 (92カテゴリ, `--granularity fine`) or 東証33業種 (`coarse`)
- **Y軸メトリクス**: セクター内割合 (%, `--y-metric ratio`, デフォルト) or 絶対銘柄数 (`count`)
- **週次Top-Nランクイン**: 各週でYTD高値更新割合の上位N (デフォルト10) セクターを集計
- **日経超えゲート**: セクターの等加重平均リターンが日経225 ETF (1321.T) を上回った週のみランクイン対象 (`--beat-nikkei`, デフォルト有効)
- **バケット分割プロット**: ランクイン週数で3段に分割 (40+, 10-19, 2-9週)、各バケット上限10セクター
- **3週移動平均**: `--ma-window 3` (デフォルト)
- **凡例**: セクター名 (ランクイン週数/セクター銘柄数)

#### 主要CLI引数

```
--year              対象年 (デフォルト: 2025)
--granularity       coarse | fine (デフォルト: fine)
--y-metric          count | ratio (デフォルト: ratio)
--beat-nikkei       日経超えゲート有効 (デフォルト)
--no-beat-nikkei    日経超えゲート無効
--nikkei-symbol     ベンチマーク (デフォルト: 1321.T)
--top-n             週次上位セクター数 (デフォルト: 10)
--selection         annual | weekly (デフォルト: weekly)
--min-weeks         ランクイン最低週数 (デフォルト: 2)
--ma-window         移動平均窓 (デフォルト: 3)
--bucket-split / --no-bucket-split
--bucket-top-n      バケット内表示上限 (デフォルト: 10)
--buckets           バケット境界 (デフォルト: "40,20,10,2")
--show-other        「その他」線を表示
```

#### 主要関数

- `normalize_symbol()` - シンボル正規化 (.T 付与)
- `load_sector_master()` - 東証33業種マスター読込
- `load_fine_sector_map()` - 四季報 `simple_sector` の小分類読込 (大分類/小分類 → 小分類)
- `build_sector_map()` - fine=四季報優先+33業種フォールバック, coarse=33業種のみ
- `compute_ytd_high_updates()` - 銘柄の年初来高値更新週を判定 (`cummax().shift(1)`)
- `compute_weekly_closes()` - 各週の最終終値を取得
- `weekly_returns_from_closes()` - 週次リターン算出
- `_detect_japanese_font()` - matplotlib用日本語フォント検出

#### 週番号の定義

ISO週ではなく dayofyear ベース: `((day_of_year - 1) // 7 + 1).clip(upper=52)`

### `scripts/fetch_all_prices.py`

全銘柄の価格データをダウンロードするスクリプト (yfinance 使用)。

## データファイル

| パス | 内容 |
|------|------|
| `data/prices/*.csv` | 個別銘柄の日次価格 (Date, Adj Close, Close, High, Low, Open, Volume) |
| `data/prices/1321_T.csv` | 日経225 ETF (ベンチマーク用) |
| `data/sector_master_template.csv` | 東証33業種マスター (4,428銘柄, ETF除外で3,947) |
| `projects/shikiho_text_parser/output/4q2_selection/4q2_scored_universe.csv` | 四季報データ (3,383銘柄, `simple_sector` 列) |

## 環境制約

- **yfinance がインストール不可** (multitasking のビルド失敗) → `src.stock_signal` の import を回避し、必要関数をスクリプト内にインライン実装
- matplotlib は `Agg` バックエンド (ヘッドレス環境)
- 日本語フォント: IPAGothic 利用可能

## ブランチ

- 開発ブランチ: `claude/weekly-sector-highs-analysis-Dg3IO`

## 最新出力例

- `data/ytd_high_by_sector_2025_fine_weekly_beatnk_ratio_ma3.png` - 日経超えゲート有効の割合グラフ
- `data/ytd_high_by_sector_2025_fine_weekly_beatnk_ratio_ma3.csv` - 同CSV

## 開発履歴

1. 基本的な週次YTD高値更新の積み上げ棒グラフ
2. 四季報小分類 (92カテゴリ) への細分化 + 折れ線グラフ化
3. 週次Top-N選択 (年間Top-Nから切替)、min-weeks フィルタ
4. ランクイン週数バケット分割プロット (3-4パネル)
5. 3週移動平均の追加
6. Y軸を絶対数→セクター内割合 (%) に変更 (デフォルト)
7. バケット内上限10セクター + 凡例にセクター銘柄数表示
8. 日経超えゲート追加 (セクター週次リターン > 日経225 ETF リターン)

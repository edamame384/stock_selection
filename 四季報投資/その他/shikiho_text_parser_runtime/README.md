# shikiho_text_parser

四季報コピペテキストを構造化し、4Q-2 銘柄を選定し、標準売買条件 `condition2` でバックテストするための独立プロジェクトです。

## 再現性の前提

GitHub 共有先で新規データ取得なしに同じ処理を行うには、次のデータをこのプロジェクト配下に含めます。

- `data/raw/4Q-2/*.txt`
- `data/prices_4q2/*.csv`
- `data/reference/sector_master_template.csv`

この構成が揃っていれば、共有先は追加ダウンロードなしで同じ処理を再実行できます。

## フォルダ構成

```text
projects/shikiho_text_parser/
  backtest_4q2_signals.py
  parse_shikiho_text.py
  prepare_github_bundle.py
  select_promising_4q2.py
  paths.py
  data/
    raw/
      4Q-2/
    prices_4q2/
    prices_full/
    reference/
      sector_master_template.csv
      bundle_manifest.json
      data_inventory.md
      data_inventory.csv
      futures_log_manifest.json
  output/
```

## 依存ライブラリ

ルートの `requirements.txt` またはこのプロジェクト配下の `requirements.txt` を使います。

```powershell
pip install -r requirements.txt
```

または:

```powershell
pip install -r projects/shikiho_text_parser/requirements.txt
```

## GitHub 共有向けデータ束ね

ローカルにある元データを、このプロジェクト配下にコピーします。

```powershell
python projects/shikiho_text_parser/prepare_github_bundle.py
```

生成物:

- `data/raw/4Q-2`
- `data/prices_4q2`
- `data/prices_full`
- `data/reference/sector_master_template.csv`
- `data/reference/bundle_manifest.json`
- `data/reference/data_inventory.md`
- `data/reference/data_inventory.csv`

## 実行手順

### 1. 四季報テキストの構造化

```powershell
python projects/shikiho_text_parser/parse_shikiho_text.py `
  --input projects/shikiho_text_parser/data/raw/4Q-2 `
  --output-dir projects/shikiho_text_parser/output
```

### 2. 銘柄選定

```powershell
python projects/shikiho_text_parser/select_promising_4q2.py
```

出力:

- `output/4q2_selection/4q2_scored_universe.csv`
- `output/4q2_selection/4q2_selected_candidates.csv`

### 3. バックテスト

```powershell
python projects/shikiho_text_parser/backtest_4q2_signals.py
```

出力:

- `output/4q2_signal_backtest/summary.json`
- `output/4q2_signal_backtest/trade_log.csv`
- `output/4q2_signal_backtest/roundtrip_trades.csv`
- `output/4q2_signal_backtest/equity_curve.csv`

### まとめて実行

```powershell
python projects/shikiho_text_parser/run_pipeline.py
```

すでに `data/raw/4Q-2` と `data/prices_4q2` を同梱済みなら:

```powershell
python projects/shikiho_text_parser/run_pipeline.py --skip-bundle
```

## 標準条件 condition2

- `promising_score_min = 0.66`
- `trend_r2_min = 0.50`
- `sector_per_score_min = 0.55`
- `entry_mode = breakout_up`
- `entry_limit_pct = 1.5`
- `take_profit_pct = 8.0`
- `stop_loss_pct = 5.0`

意味:

- 前営業日終値でシグナル予兆を判定
- 当日の高値が前営業日終値 `+1.5%` を上回ったら逆指値買い
- 利確 `+8%`
- 損切 `-5%`

## 注意

- `output/` の最適化試行結果は容量が大きくなりやすいです。
- GitHub に載せる場合は、必要な成果物だけを残し、不要な試行ディレクトリは外すのが妥当です。
- 現在の束ね結果では `4Q-2` テキスト 4,223 件に対して価格 CSV は 4,220 件です。3 件は元データ側の価格欠損です。
- `prices_full` は大きいため、GitHub では Git LFS 前提で扱うのが妥当です。

## 再現できる処理 / できない処理

### 再現できる処理

- 4Q-2 テキストの構造化
- 4Q-2 銘柄の選定
- `condition2` のバックテスト
- `prices_full` を使った追加の株価分析

### 再現できない処理

- 夜間先物を特徴量に使う戦略
- ダウ先物 / S&P500先物 / 日経先物の実データ時系列を使う戦略

理由:

- ローカルワークスペース内では、先物の実データ CSV は見つかっていません
- 見つかったのは先物処理を実行したログのみです

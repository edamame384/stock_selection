# 運用方針 v3.4

## 概要
- 個別株専用
- ベースは `v3.1` の因果版
- 決算ルール:
  - 決算前 `1` 営業日で売却
  - 決算後 `5` 営業日は新規購入禁止
- 日経局面ラベルは `1` 営業日遅れで利用
- 四季報データは四半期固定、価格系指標は月次で再評価して標準テーブルを再構築
- 現在の正式候補

## 運用データの扱い
- 運用で参照する `selected_csv` は `operational/` 配下のCSVに固定する
- 研究用CSVには将来検証列が残ることがあるため、運用ロジックでは使わない
- 現行の運用用参照先:
  - `projects/quarterly_ranker/output/q2_2024_pre_analysis_20240630_aligned/operational/q2_2024_pre_selected_candidates_operational.csv`
  - `projects/quarterly_ranker/output/q3_pre_analysis_20250630_aligned/threshold_search_post_high_vol/operational/best_selected_candidates_operational.csv`
  - `projects/quarterly_ranker/output/q4_pre_analysis_20250930_full/operational/q4_pre_selected_candidates_operational.csv`
  - `projects/shikiho_text_parser/output/4q2_selection/operational/4q2_selected_candidates_operational.csv`

## 選定困難局面
### 1. `raw_post_crash_high_vol`
- 直近 `5` 営業日に `crash`
- その後に `reversal_up / capitulation_end / normal / uptrend` を挟んでいない `high_vol`
- 扱い: `no_trade`

### 2. `weak_uptrend`
- `uptrend` 連続が `3` 日以内
- 直前 `5` 営業日に `high_vol / crash / surge / settling` を含む
- 扱い: `q3_post_high_vol`

### 3. `rebound_confirmed_post_crash_high_vol` / `generic_high_vol`
- 扱い: `condition2`

## フル実行結果
- `q2_2024`: `+1.67%`
- `3Q`: `+18.31%`
- `4Q`: `+9.65%`
- `4Q-2`: `+4.88%`
- 合計: `+8.63%`

## 月次標準テーブル件数の例
- `q2_2024`
  - `2024-07`: `23`
  - `2024-08`: `26`
  - `2024-09`: `9`
- `4Q-2`
  - `2026-01`: `68`
  - `2026-02`: `80`
  - `2026-03`: `77`

## 補足
- 月次標準テーブル再構築により、従来の固定テーブル版とは収益構成が変わる
- 現在の `v3.4` は固定テーブル版ではなく、月次再評価版を指す
- レジーム連動で月次基準まで切り替える比較版は別候補として分離し、正式候補には反映していない

## 現時点の整理
- `raw_post_crash_high_vol` は利益を取りに行く局面ではなく、損失抑制局面
- `weak_uptrend` は通常の `uptrend` と分けて扱う必要がある
- これを一本化したものが `v3.4`

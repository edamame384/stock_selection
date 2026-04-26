# 運用方針 v3.1

## 目的

- 日経局面に応じて売買手法を切り替える
- 通常局面の収益性を維持しつつ、`q2_2024` のような悪地合いを改善する
- 既存の強い手法を壊さずに、`crash` だけを防御的に補正する

## 現時点の採用手法

- 基本手法: `実務向け局面切替v3`
- 実装:
  - `projects/shikiho_text_parser/backtest_phase_adaptive_practical_v3.py`
  - `projects/shikiho_text_parser/methods/method_phase_adaptive_practical_v3.py`

## 局面ごとの売買手法

- `normal`, `uptrend`
  - `condition2`
- `stable`
  - `breakout_1.5`
- `high_vol`, `settling`, `reversal_up`, `capitulation_end`, `overheated_range`
  - `q3_post_high_vol`
- `surge`, `downtrend`, `reversal_down`
  - `no_trade`
- `crash`
  - 通常の `v2` では `no_trade + 早期損切`
  - `v3.1` ではこれに加えて、`crash` 専用テーブルを使う

## 銘柄テーブル

### 1. 標準テーブル

- 各データセットの通常選定銘柄を使う
- 現在の採用件数:
  - `q2_2024`: `24`
  - `3Q`: `67`
  - `4Q`: `60`
  - `4Q-2`: `75`

補足:
- `3Q` は 13 銘柄版より、67 銘柄版の方が良い
- `v3.1` で比較すると
  - `13銘柄`: `+12.68%`
  - `67銘柄`: `+14.91%`
- したがって、`3Q` の標準テーブルは **67銘柄版を採用**

### 2. 暴落後テーブル

- 手法名: `暴落後拡張`
- 実装:
  - `projects/shikiho_text_parser/methods/method_post_crash_broad.py`
- 条件:
  - `trend_r2 >= 0.60`
  - `annual_return_pct >= 25`
  - `quarter_return_pct >= 10`
  - `positive_month_ratio_pct >= 50`
  - `persistence_20d_pct >= 55`
  - `sector_adjusted_per_score >= 0.35`
  - `ocr_per <= 20`
- 売買前提:
  - `前日終値 +1.0%`
  - `+5% / -4%`

## テーブル切替ルール

- `crash` のときだけ `暴落後拡張` テーブルを使う
- それ以外は標準テーブルを使う
- `crash` テーブルは **月次再評価**

## バックテスト結果

### v2

- `q2_2024`: `+1.20%`
- `3Q`: `+12.68%`
- `4Q`: `+11.80%`
- `4Q-2`: `+11.41%`
- 合計: `+9.27%`

### v3.1

- `q2_2024`: `+6.15%`
- `3Q`: `+14.91%`
- `4Q`: `+11.80%`
- `4Q-2`: `+11.41%`
- 合計: `+11.07%`

### 結論

- `v3.1` は `q2_2024` を改善し、`3Q / 4Q / 4Q-2` を維持する
- 現時点では、`v2` より `v3.1` を採用する方が合理的

## 実務上の解釈

- `crash` は「通常の標準テーブルでは弱い局面」
- ただし、高ボラ全般で防御テーブルに切り替えると `3Q / 4Q / 4Q-2` を壊しやすい
- したがって、防御テーブルの適用は **`crash` 限定** が妥当

## 未確定事項

- `stable` でのテーブル差の追加検証
- `settling` と `reversal_up` の分離検証
- 月次更新に加えた局面転換時の臨時更新

## 参照ファイル

- `projects/shikiho_text_parser/output/phase_adaptive_practical_v3_batch/summary_all.csv`
- `projects/shikiho_text_parser/output/phase_adaptive_practical_v3_batch/v2_vs_v3_compare.csv`
- `projects/shikiho_text_parser/output/q2_defensive_phase_usage/search_results_reduced_combined.csv`

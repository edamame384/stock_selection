# Methods

このフォルダには、売買手法を名前付きで固定した実行ラッパーを置きます。

## 現在の正式候補

- `method_phase_adaptive_practical_v34.py`
  - 日本語別名: `実務向け局面切替v3.4`
  - 現在の正式候補
  - 個別株専用の因果版
  - `raw_post_crash_high_vol` は `no_trade`
  - `weak_uptrend` は `q3_post_high_vol`
  - `rebound_confirmed_post_crash_high_vol` と `generic_high_vol` は `condition2`
  - 決算ルール:
    - 決算前 `1` 営業日で売却
    - 決算後 `5` 営業日は新規購入禁止

## 手法一覧

- `method_condition2.py`
  - 日本語別名: `上向き標準`
  - 上向き相場で使う標準手法
  - `promising_score >= 0.66`
  - `trend_r2 >= 0.50`
  - `sector_per_score >= 0.55`
  - `前日終値 +1.5%` の逆指値買い
  - `+8%` 利確 / `-5%` 損切

- `method_breakout_1p5.py`
  - 日本語別名: `横ばい順張り`
  - 横ばい相場で使う強い順張り手法
  - `promising_score >= 0.72`
  - `trend_r2 >= 0.60`
  - `sector_per_score >= 0.55`
  - `前日終値 +1.5%` の逆指値買い
  - `+8%` 利確 / `-5%` 損切

- `method_regime_switch.py`
  - 日本語別名: `日経切替`
  - 日経 20 日騰落率で `condition2` と `breakout_1.5` を切り替える手法
  - `20日騰落率 > 0` : `condition2`
  - `|20日騰落率| <= 7%` : `breakout_1.5`
  - それ以外 : 新規買い停止

- `method_q3_post_high_vol.py`
  - 日本語別名: `高ボラ通過後`
  - 高ボラ局面通過後の四半期専用手法
  - `trend_r2 >= 0.60`
  - `annual_return_pct >= 25`
  - `quarter_return_pct >= 10`
  - `positive_month_ratio_pct >= 50`
  - `persistence_20d_pct >= 55`
  - `sector_adjusted_per_score >= 0.35`
  - `ocr_per <= 20`
  - `前日終値 +1.5%` の逆指値買い
  - `+8%` 利確 / `-5%` 損切

- `method_phase_adaptive_practical_v2.py`
  - 日本語別名: `実務向け局面切替v2`
  - `上向き標準 / 横ばい順張り / 高ボラ通過後` を局面で切り替える実務向け手法
  - `uptrend, normal` : `上向き標準`
  - `stable` : `横ばい順張り`
  - `high_vol, settling, reversal_up, capitulation_end, overheated_range` : `高ボラ通過後`
  - `surge, downtrend, reversal_down` : 新規停止
  - `crash` : 新規停止 + 既存ポジションは前日終値基準の早期損切

- `method_q2_2024_bad_regime_concentrated.py`
  - 日本語別名: `悪条件集中防御`
  - 悪地合い用の少数集中・防御型手法
  - `trend_r2 >= 0.35`
  - `annual_return_pct >= 0`
  - `quarter_return_pct >= -10`
  - `positive_month_ratio_pct >= 35`
  - `persistence_20d_pct >= 35`
  - `max_drawdown_pct <= 18`
  - `avg_monthly_high_low_change_pct <= 14`
  - `end_to_trailing_high_pct >= 88`
  - `sector_adjusted_per_score >= 0.35`
  - `ocr_per <= 18`
  - `sector_adjusted_per_score` 上位 `5銘柄`
  - `前日終値 +1.0%` の逆指値買い
  - `+5%` 利確 / `-4%` 損切

- `method_post_crash_broad.py`
  - 日本語別名: `暴落後拡張`
  - 暴落後の基準テーブル
  - `trend_r2 >= 0.60`
  - `annual_return_pct >= 25`
  - `quarter_return_pct >= 10`
  - `positive_month_ratio_pct >= 50`
  - `persistence_20d_pct >= 55`
  - `sector_adjusted_per_score >= 0.35`
  - `ocr_per <= 20`
  - 売買前提は防御型
  - `前日終値 +1.0%` の逆指値買い
  - `+5%` 利確 / `-4%` 損切

- `method_phase_adaptive_practical_v3.py`
  - 日本語別名: `実務向け局面切替v3`
  - `実務向け局面切替v2` を基礎にした拡張版
  - `crash` のときだけ `暴落後拡張` テーブルを使う
  - `crash` テーブルは月次で再評価する
  - それ以外の局面は `v2` のまま

- `method_phase_adaptive_practical_v31.py`
  - 日本語別名: `実務向け局面切替v3.1`
  - `v3` の 3Q 標準テーブルを 67 銘柄版に広げた運用版
  - `crash` のときだけ `暴落後拡張` テーブルを使う
  - `crash` テーブルは月次で再評価する
  - それ以外の局面は `v2` のまま

- `method_phase_adaptive_practical_v31_pre_earnings1.py`
  - 日本語別名: `実務向け局面切替v3.1_決算前1営業日`
  - `v3.1` に決算前 1 営業日の新規停止・保有解消ルールを加えた版

- `method_phase_adaptive_practical_v31_post_earnings5.py`
  - 日本語別名: `実務向け局面切替v3.1_決算後5営業日`
  - `v3.1` に決算後 5 営業日以内のみ新規買いを許可する版

- `method_phase_adaptive_practical_v31_pre1_post5.py`
  - 日本語別名: `実務向け局面切替v3.1_決算前1営業日_決算後5営業日見送り`
  - `v3.1` に
    - 決算前 1 営業日の保有解消
    - 決算後 5 営業日の新規買い見送り
    を加えた版

- `method_phase_adaptive_practical_v31_pre1_post5_gold_switch.py`
  - 日本語別名: `実務向け局面切替v3.1_決算前1営業日_決算後5営業日見送り_金ETF切替`
  - `pre1_post5` を土台に、`crash` 局面では個別株の代わりに金ETFを使う比較用メソッド
  - 既定値は `1328.T`、適用局面は `crash`

- `method_etf_dedicated_rebound.py`
  - 日本語別名: `ETF専用反発`
  - ETF単独で運用する専用メソッド
  - 既定シグナルは `rebound_open`
  - 悪地合いの反発取りを想定

## 使い方

例:

```powershell
python projects/shikiho_text_parser/methods/method_condition2.py `
  --selected-csv projects/shikiho_text_parser/output/4q2_selection/4q2_selected_candidates.csv `
  --output-dir projects/shikiho_text_parser/output/method_condition2_run `
  --start-date 2026-01-01 `
  --end-date 2026-03-10
```

```powershell
python projects/shikiho_text_parser/methods/method_breakout_1p5.py `
  --selected-csv projects/shikiho_text_parser/output/4q2_selection/4q2_selected_candidates.csv `
  --output-dir projects/shikiho_text_parser/output/method_breakout_1p5_run `
  --start-date 2026-01-01 `
  --end-date 2026-03-10
```

```powershell
python projects/shikiho_text_parser/methods/method_regime_switch.py `
  --selected-csv projects/quarterly_ranker/output/q4_pre_analysis_20250930_full/q4_pre_selected_candidates_condition2_input.csv `
  --output-dir projects/shikiho_text_parser/output/method_regime_switch_run `
  --start-date 2025-10-01 `
  --end-date 2025-12-31
```

```powershell
python projects/shikiho_text_parser/methods/method_q3_post_high_vol.py `
  --selected-csv projects/quarterly_ranker/output/q3_pre_analysis_20250630_aligned/threshold_search_post_high_vol/best_selected_candidates.csv `
  --output-dir projects/quarterly_ranker/output/q3_pre_analysis_20250630_aligned/method_q3_post_high_vol_run `
  --start-date 2025-07-01 `
  --end-date 2025-09-30
```

```powershell
python projects/shikiho_text_parser/methods/method_phase_adaptive_practical_v2.py `
  --selected-csv projects/shikiho_text_parser/output/4q2_selection/4q2_selected_candidates.csv `
  --output-dir projects/shikiho_text_parser/output/method_phase_adaptive_practical_v2_run `
  --start-date 2026-01-01 `
  --end-date 2026-03-10 `
  --dataset-name 4q2
```

```powershell
python projects/shikiho_text_parser/methods/method_phase_adaptive_practical_v31.py `
  --detail-csv projects/shikiho_text_parser/output/4q2_selection/4q2_scored_universe.csv `
  --selected-csv projects/shikiho_text_parser/output/4q2_selection/4q2_selected_candidates.csv `
  --output-dir projects/shikiho_text_parser/output/method_phase_adaptive_practical_v31_run `
  --start-date 2026-01-01 `
  --end-date 2026-03-10 `
  --dataset-name 4q2
```

- method_phase_adaptive_practical_v31_pre1_post5_etf1489_crash.py
  - 日本語別名: 実務向け局面切替v3.1_決算前1営業日_決算後5営業日見送り_高配当ETF切替
  - pre1_post5 を土台に、crash 局面では 1489.T を使う派生版

- `method_phase_adaptive_practical_v31_pre1_post5_etf1489_crash_rebound.py`
  - 日本語別名: `実務向け局面切替v3.1_決算前1営業日_決算後5営業日見送り_高配当ETF反発切替`
  - `pre1_post5` を土台に、`crash` 局面では `1489.T` を `rebound_open` の反発専用ロジックで使う派生版

- `method_phase_adaptive_practical_v31_pre1_post5_etf1489_postcrash_highvol_rebound.py`
  - 日本語別名: `実務向け局面切替v3.1_決算前1営業日_決算後5営業日見送り_高配当ETF暴落後高ボラ反発切替`
  - `pre1_post5` を土台に、直近 `5` 営業日に `crash` を含む `high_vol` 局面だけ `1489.T` を `rebound_open` で使う派生版

- `method_phase_adaptive_practical_v32.py`
  - 日本語別名: `実務向け局面切替v3.2`
  - `v3.1 + 決算前1営業日売却 + 決算後5営業日見送り` を土台に、`crash` 後の `high_vol` だけ `1489.T` の反発専用ロジックを使う

- `method_phase_adaptive_practical_v33.py`
  - 日本語別名: `実務向け局面切替v3.3`
  - ETFを使わない個別株専用の因果版
  - `high_vol` を選定困難局面として分解し、`raw_post_crash_high_vol` は `no_trade`、`rebound_confirmed_post_crash_high_vol` と `generic_high_vol` は `condition2`、それ以外は `v3.1` 系の因果ルールを使う

- `method_phase_adaptive_practical_v34_weak_uptrend_candidate.py`
  - 日本語別名: `実務向け局面切替v3.4候補_weak_uptrend`
  - `weak_uptrend` を別ラベル化し、`q3_post_high_vol` に切り替える比較用候補版

- `method_phase_adaptive_practical_v34.py`
  - 日本語別名: `実務向け局面切替v3.4`
  - `raw_post_crash_high_vol` を `no_trade`、`weak_uptrend` を `q3_post_high_vol`、`rebound_confirmed_post_crash_high_vol` と `generic_high_vol` を `condition2` に切り替える一本化版
  - 現在の正式候補

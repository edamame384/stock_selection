# 運用方針 v3.8

## 位置づけ
- `v3.8` は、`v3.7` で未確立だった **crash から通常局面へ戻るまでの区間** を取るための運用版。
- ローカル通知・GitHub Actions は `run_v38_signal.py` を使う。
- `phase_name` が `normal / uptrend / high_vol / settling` でも、`post_major_crash_mode` 中は通常局面とは区別する。
- 表示・通知では `post_major_crash_mode = True` の間、通常名ではなく `post_crash_*` 名を使う。
  - `normal` -> `post_crash_normal`
  - `uptrend` -> `post_crash_uptrend`
  - `high_vol` -> `post_crash_high_vol`
  - `settling` -> `post_crash_settling`
  - `downtrend` -> `post_crash_downtrend`
  - `capitulation_end` -> `post_crash_capitulation_end`
- これは通常局面の `normal / uptrend` と、crash後回復モード中の同名ラベルを混同しないための表示名。

## 3メソッド構成
### 1. crash期間中メソッド
- 対象: `phase_name = crash`
- 採用済み:
  - `crash_late_runner_theme_or_hard_entry`
- 条件:
  - `crash_pos >= 4`
  - 20日高値ブレイク
  - 5日騰落率 `>= 12%`
  - 出来高倍率 `>= 2.0`
  - `close_pos >= 0.60`
  - `upper_shadow_ratio <= 0.25`
  - `theme_cluster=True` または `hard_detached=True`
  - 1日上位2銘柄まで
  - 同一銘柄は過去5営業日内の再シグナルを除外
- `q2_defensive_entry` は不採用。

### 2. crash後の高ボラ・横ばい期間メソッド
- 対象候補:
  - `post_major_crash_mode = True`
  - かつ `phase_name in high_vol / settling / surge / reversal_down / raw_post_crash_high_vol`
- `concentrated` は通常の集中相場とは分けて扱う。
- 現状候補:
  - `broad_runner_daily_top2`
  - `broad_runner_daily_top2_hard_detached`
- ただし、実データ上の候補発生は主に `concentrated downtrend / surge` に寄っているため、まだ正式採用しない。

### 3. crash後から上昇する期間メソッド
- 実務候補は `post_major_crash_mode` 中の **分散相場 prev_high_break top1**。

## 対象区間
- `post_major_crash_mode = True`
- `post_major_active = True`
- `sector_mode = dispersed`
- 実務候補では細かい局面ラベルに頼りすぎない
- 研究上の限定候補では次のみに限定:
  - `capitulation_end`
  - `downtrend`

## 買い条件
- 個別株の `prev_high_break`
- 当日候補の **上位1銘柄のみ**
- 研究上の限定候補では `normal / settling` は対象外
- 実務候補では `normal` も対象に含めるが、出口を浅くする
- `settling` は現時点では対象外

## 売り条件
- 利確 `+10%`
- 損切 `-5%`
- `prev_high_break` シグナル消失時は売却
- バックテストではOHLCベースで利確/損切を再評価

## 検証結果
- 出力:
  - `output/optimize_post_major_dispersed_prev_high_candidate/variant_summary.csv`
  - `output/optimize_post_major_dispersed_prev_high_candidate/simulated_trades.csv`
  - `output/optimize_post_major_dispersed_prev_high_candidate/summary.json`

### 研究上の限定候補
- variant: `cap_down_top1_tp10_sl5`
- 件数: `7`
- 勝率: `100%`
- 損益: `+174,039円`
- 平均リターン: `+3.21%`
- 最大同時保有額: `1,966,425円`
- 最大同時保有本数: `2`

### 実務候補
- variant: `all_top1_tp10_sl5`
- 対象:
  - `post_major_crash_mode = True`
  - `post_major_active = True`
  - `sector_mode = dispersed`
  - `phase_name != crash`
- 件数: `16`
- 勝率: `68.75%`
- 損益: `+171,251円`
- 平均リターン: `+0.86%`
- 最大同時保有額: `4,438,058円`
- 最大同時保有本数: `2`

### 実務向け保護条件候補
- variant: `mixed_capdown_tp10_normal_tp6`
- 対象:
  - `post_major_crash_mode = True`
  - `post_major_active = True`
  - `sector_mode = dispersed`
  - `phase_name != crash`
- 買い:
  - top1
- 出口:
  - `capitulation_end / downtrend`: 利確 `+10%`, 損切 `-5%`
  - `normal`: 利確 `+6%`, 損切 `-3%`
- 検証結果:
  - 件数 `16`
  - 勝率 `75.00%`
  - 損益 `+213,517円`
  - 平均リターン `+1.42%`
  - 最大損失 `-14,580円`
  - 最大同時保有額 `4,438,058円`
  - 最大同時保有本数 `2`

### 参考: リターン重視案
- variant: `cap_down_top2_tp10_sl5`
- 件数: `14`
- 勝率: `78.57%`
- 損益: `+201,188円`
- 平均リターン: `+2.22%`
- 最大同時保有額: `2,420,832円`
- 最大同時保有本数: `3`

## 判断
- `capitulation_end / downtrend` 限定は研究上は最良だが、実務では局面ラベルが細かすぎる。
- 実務寄せでは `post_major_active + dispersed + phase != crash` の top1 とする。
- ただし `normal` まで広げると勝率と資金効率は落ちるため、正式採用前に追加の保護条件を検討する。
- 現時点では、`v3.8` の実務候補は **post-major dispersed 全体 top1、TP+10% / SL-5%** とする。
- ただし、`normal` まで許す場合は `normal` だけ出口を浅くする **mixed出口案** が最有力。

## 追加最適化: post_crash_* 分離後の個別株メソッド
- 追加出力:
  - `output/optimize_v38_post_major_individual_candidate/variant_summary.csv`
  - `output/optimize_v38_post_major_individual_candidate/variant_trade_details.csv`
  - `output/optimize_v38_post_major_individual_candidate/dataset_variant_summary.csv`
  - `output/optimize_v38_post_major_individual_candidate/effective_regime_variant_summary.csv`
  - `output/optimize_v38_post_major_individual_candidate/no_lookahead_audit.csv`

### 暫定最有力
- variant: `mixed_normal_price_top3_le042`
- 対象:
  - `post_major_active = True`
  - `sector_mode = dispersed`
  - `phase_name != crash`
  - top1
- 出口:
  - `post_crash_capitulation_end / post_crash_downtrend`: 利確 `+10%`, 損切 `-5%`
  - `post_crash_normal`: 利確 `+6%`, 損切 `-3%`
- 追加保護:
  - `post_crash_normal` は `price_top3_share <= 0.42` のときだけ許可
- 検証結果:
  - 件数 `11`
  - 勝率 `90.91%`
  - 損益 `+245,587円`
  - 平均リターン `+2.76%`
  - 最大損失 `-11,390円`
  - 最大同時保有額 `4,077,138円`
  - 最大同時保有本数 `1`

### 保守候補
- variant: `mixed_normal_early_and_price`
- 追加保護:
  - `post_crash_normal` は `post_major_active_pos <= 10`
  - かつ `price_top3_share <= 0.42`
- 検証結果:
  - 件数 `8`
  - 勝率 `100%`
  - 損益 `+214,901円`
  - 平均リターン `+2.93%`
  - 最大損失 `+716円`
  - 最大同時保有額 `4,077,138円`
  - 最大同時保有本数 `1`

### 注意
- `post_crash_normal` の追加保護はサンプルが小さいため、過剰最適化に注意する。
- 実装時は、まず `mixed_normal_price_top3_le042` を暫定候補、`mixed_normal_early_and_price` を保守候補として扱う。

## 未確立の領域
- `post_major_crash_mode` 中の `normal / uptrend / stable`
- `settling / high_vol` 専用メソッド

## 追加最適化: post_crash_concentrated ETF メソッド
- 追加出力:
  - `output/optimize_v38_post_crash_concentrated_etf_candidate/variant_summary.csv`
  - `output/optimize_v38_post_crash_concentrated_etf_candidate/etf_variant_trades.csv`
  - `output/optimize_v38_post_crash_concentrated_etf_continuous_candidate/variant_summary.csv`
  - `output/optimize_v38_post_crash_concentrated_etf_continuous_candidate/etf_variant_trades.csv`
  - `output/optimize_v38_post_crash_concentrated_etf_continuous_candidate/no_lookahead_audit.csv`

### 検証上の注意
- ETF集中側は四半期境界をまたぐため、データセット単位で強制クローズすると歪む。
- そのため、連続日付ベースの再検証を追加した。
- 500万円運用に合わせて、ETFの1回あたり投下上限を `5,000,000円` に固定した。
- `SSA` は `2020年` に履歴がないため、`SSA必須` にすると2020年の回復局面を丸ごと取り逃す。

### 暫定最有力
- variant: `ssa_when_available_all`
- TP/SL: 利確 `+8%`, 損切 `-4%`
- 対象:
  - `post_major_active = True`
  - `sector_mode = concentrated`
  - `phase_name != crash`
  - SSA履歴が存在する期間は `ssa_recovery_confirm=True` を要求
  - SSA履歴が存在しない期間は、SSAではブロックしない
- 検証結果:
  - 件数 `20`
  - 勝率 `75.00%`
  - 損益 `+1,711,663円`
  - 平均リターン `+1.74%`
  - 最大損失 `-350,000円`
  - 最大同時保有額 `4,998,000円`
  - 最大同時保有本数 `1`

### 保守候補
- variant: `ssa_when_available_non_normal`
- TP/SL: 利確 `+6%` または `+8%`, 損切 `-3%` または `-4%`
- `post_crash_normal` は除外する。
- 検証結果:
  - 件数 `15`
  - 勝率 `73.33%`
  - 損益 `+1,097,269円`
  - 最大同時保有額 `4,998,000円`
  - 最大同時保有本数 `1`

### 判断
- `post_crash_concentrated` は未開拓日数が最も多く、ETFメソッドの候補として有望。
- 現時点では、`SSA` を「利用可能なら確認、履歴なしなら不使用」とする `ssa_when_available_all` が最有力。
- ただし `2020-1Q` の早すぎるETF入りは損失になっているため、正式採用前に追加保護条件をもう一段確認する。

### 実務寄せ保護条件の追加確認
- `post_crash_capitulation_end` は早すぎるETF入りになりやすく、2020-1Qの損失を拾っていた。
- `post_crash_settling` も単発で損失になりやすかった。
- 実務候補としては、`capitulation_end` と `settling` を除外する。

### 実務寄せ暫定本命
- variant: `ssa_when_available_no_capitulation_no_settling`
- TP/SL: 利確 `+8%`, 損切 `-4%`
- 対象:
  - `post_major_active = True`
  - `sector_mode = concentrated`
  - `phase_name != crash`
  - `phase_name != capitulation_end`
  - `phase_name != settling`
  - SSA履歴が存在する期間は `ssa_recovery_confirm=True` を要求
  - SSA履歴が存在しない期間は、SSAではブロックしない
- 検証結果:
  - 件数 `17`
  - 勝率 `82.35%`
  - 損益 `+1,745,578円`
  - 平均リターン `+2.07%`
  - 最大損失 `-163,902円`
  - 最大同時保有額 `4,998,000円`
  - 最大同時保有本数 `1`

### ライブ通知でのETFトリガー
- ETF集中メソッドの通知では、翌営業日の高値を使ってシグナル有無を判定しない。
- `signal_date` までに確定しているETF価格だけで回復強度を評価し、翌営業日用のトリガー価格を通知する。
- トリガー価格は当面、`signal_date` のETF高値に `STOCK_BREAK_MULT` を掛けた値とする。
- ETF候補は、リスクオン回復メソッドとして金ETF `1328.T` を除外する。
- ETFは売買単位を1口として検証する。
- 背景:
  - 旧ロジックはバックテスト用の `prev_high_break` 判定をライブ通知にも使っていた。
  - そのため、通知時点では未確定の翌営業日高値が存在しない限りETFシグナルが点灯しにくかった。
  - crash後ETFでは、crash前高値の回復を待つのではなく、`post_crash_*` 状態・SSA回復確認・ETFの短期回復強度で判断する。

### ライブ通知での個別株トリガー
- `v38_post_crash_dispersed_prev_high_entry` と通常主戦略の通知でも、翌営業日の高値を使ってシグナル有無を判定しない。
- `signal_date` までの価格・指標で候補を選び、翌営業日用のトリガー価格を通知する。
- バックテスト上の約定判定と、ライブ通知上の注文トリガー生成は分離する。
- ETF候補も鮮度チェック対象に含め、判定対象CSVが古い場合は通知へ進まない。

### ETFスリーブ再検証
- 出力:
  - `output/optimize_v38_post_crash_concentrated_etf_sleeve_candidate/variant_summary.csv`
  - `output/optimize_v38_post_crash_concentrated_etf_sleeve_candidate/etf_sleeve_trades.csv`
  - `output/optimize_v38_post_crash_concentrated_etf_sleeve_candidate/no_lookahead_audit.csv`
- 比較:
  - `fixed_1321`
  - `fixed_1321_1306`
  - `core_1321_plus_top1_no_gold`
  - `top2_no_gold`
  - `top1_no_gold`
  - それぞれの保持型 `_hold`
- 現時点の最有力:
  - `top1_no_gold_hold`
  - 件数 `9`
  - 勝率 `77.78%`
  - 損益 `+2,579,341円`
  - 平均リターン `+5.73%`
  - 最大損失 `-200,961円`
  - 最大同時保有額 `4,999,778円`
  - 最大同時保有本数 `1`
- 解釈:
  - ETFスリーブとして複数案を比較したが、固定の日経ETFよりも、金ETFを除いた回復強度上位ETFを保持する案が最も良かった。
  - `1321.T` は候補として有効だが、検証上は `2516.T` などの強い回復ETFを選ぶ方が成績が良かった。

### 更新後の判断
- 最初の暫定本命だった `ssa_when_available_all` は収益は良いが、`capitulation_end` と `settling` のノイズを含んでいた。
- 実務候補としては、`ssa_when_available_no_capitulation_no_settling` を優先する。

## 参照
- `output/analyze_post_crash_three_method_framework_candidate/POST_CRASH_THREE_METHOD_PLAN.md`
- `output/analyze_post_crash_three_method_framework_candidate/post_crash_state_summary.csv`
- `output/analyze_post_crash_three_method_framework_candidate/candidate_method_summary.csv`
- `output/optimize_post_major_dispersed_prev_high_candidate/practical_vs_limited_top1_tp10_sl5.csv`
- `output/optimize_post_major_dispersed_prev_high_candidate/practical_guard_variants_top1.csv`
- `output/optimize_post_major_dispersed_prev_high_candidate/practical_all_post_major_dispersed_top1_tp10_sl5_trades.csv`

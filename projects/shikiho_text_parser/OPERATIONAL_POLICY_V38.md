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

## v3.8本体の役割
- `v3.8` 本体は、`v3.5 core` では取りにくい **crash / post-crash の専用局面** を埋める役割とする。
- `theme_cluster または hard_detached` を用いた短期ブレイク戦略は、後日あらためて因果的バックテストで再設計する。
- したがって、**現在の運用版 `v3.8` にはサブ戦略を含めない**。

## 現行採用方針
- `crash` は新規エントリーしない。
- `surge` / `post_crash_surge` も新規エントリーしない。
- 理由:
  - surgeを事前に読むには夜間先物・CME・SGX・寄付き前気配などが必要。
  - 現在のローカルデータだけでは4/8型の急騰を因果的に検出できない。
  - 候補銘柄の抽出は可能だが、勝率の高い買い方はサンプルが小さく、件数を増やすと成績が悪化した。
- よって `crash` と `surge` は **取り逃し許容区間** とし、ライブ通知・統合バックテストから除外する。
- v3.8の採用対象は、`crash/surge` を除いた `post_crash_concentrated` ETF、`post_crash_dispersed` 個別株、`post_crash_normal/downtrend` リバウンド候補に限定する。

## 3メソッド構成
### 1. crash期間中メソッド
- `q2_defensive_entry` は不採用。
- 現在の `v3.8` は **crash 中の新規エントリーを行わない**。
- crash 用サブ戦略は、先読みを除いた再設計が完了するまで運用から外す。
- crash局面surgeも、夜間先物などの事前データが不足しているため正式メソッド化しない。

### 2. crash後の高ボラ・横ばい期間メソッド
- 対象候補:
  - `post_major_crash_mode = True`
  - かつ `phase_name in high_vol / reversal_down / raw_post_crash_high_vol`
  - `surge` は除外
  - `settling` はETFでは除外候補
- `concentrated` は通常の集中相場とは分けて扱う。
- 現状候補:
  - `broad_runner_daily_top2`
  - `broad_runner_daily_top2_hard_detached`
- ただし、実データ上の候補発生は主に `concentrated downtrend / surge` に寄っていたため、surge絡みの候補は正式採用しない。

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

## 候補: post_crash_normal / post_crash_downtrend リバウンド・デイトレ
- 位置づけ:
  - `post_crash_surge` を直接取りに行くのではなく、**crash後の回復継続**を短期で取る候補メソッド。
  - `post_crash_normal / post_crash_downtrend` 専用の補助候補として扱う。
- 条件:
  - `post_major_crash_mode = True`
  - 前営業日の phase が `normal` または `downtrend`
  - 前夜の `S&P500` リターン `>= +1.0%`
  - 前営業日に
    - `ret1 >= 5%`
    - `vol_ratio20 >= 2.0`
    - 20日高値ブレイク
  - 候補は `top1`
- 売買:
  - 翌営業日に **前日高値ブレイク**で買い
  - **当日引け**で売り
  - 先読みなし
- 単体検証:
  - 出力:
    - `output/backtest_post_crash_surge_breakout_sp500_candidate/post_crash_normal_downtrend_only_summary.json`
    - `output/backtest_post_crash_surge_breakout_sp500_candidate/post_crash_normal_downtrend_only_dataset_summary.csv`
  - 結果:
    - トレード数 `13`
    - 勝率 `61.54%`
    - 平均トレードリターン `+10.66%`
    - 平均区間リターン `+3.27%`
    - 損失区間 `0`
    - 年率換算Sharpe `3.14`
- 統合差し込み確認:
  - 出力:
    - `output/backtest_v38_acquired_intervals_candidate/with_post_crash_rebound_daytrade/summary.json`
    - `output/backtest_v38_acquired_intervals_candidate/with_post_crash_rebound_daytrade/component_summary.csv`
  - 差し込み後:
    - `v38_post_crash_rebound_daytrade`
    - 採用トレード数 `13`
    - 勝率 `61.54%`
    - 損益 `+649,914円`
    - 平均リターン `+10.66%`
    - 最大1日投入額 `498,300円`
    - 損失最大 `-60,400円`
    - `post_crash_downtrend`: `4件`, 損益 `+289,400円`
    - `post_crash_normal`: `9件`, 損益 `+360,514円`
  - 資金配分後:
    - 出力:
      - `output/backtest_v38_acquired_intervals_candidate/with_post_major_inactive_transition_allocated_etf300/summary.json`
      - `output/backtest_v38_acquired_intervals_candidate/with_post_major_inactive_transition_allocated_etf300/component_summary.csv`
    - 総損益 `+5,769,904円`
    - Sharpe `1.556`
    - 最大資金拘束 `4,991,060円`
    - `v38_post_crash_rebound_daytrade` は `13件` すべて採用

### 採用判断
- **正式採用する。**
- 理由:
  - 対象4区間すべてで区間損益がプラス。
  - 資金配分後も全13件が採用され、既存メソッドを押し出していない。
  - 1銘柄あたり投入額は約50万円で、ETFや個別株スイングより資金拘束が軽い。
  - 買いは `signal_date < trade_date` で、S&P500も日本の翌営業日前に確定するため、先読みではない。
- 運用上の注意:
  - これは翌日寄り成のメソッドではなく、**前日高値ブレイクを当日監視するデイトレ候補**。
  - 当日高値がトリガーに届かなければ約定しない。
  - 同日引けで手仕舞いする。
  - `post_crash_surge` 実績日は除外し、`post_crash_normal / post_crash_downtrend` 専用として扱う。

### ライブ通知
- 通常v3.8通知とは分離する。
- 通常workflow:
  - `stock-signal-runner-v38`
  - `run_v38_signal.py --disable-daytrade`
- 朝のデイトレ専用workflow:
  - `stock-signal-runner-v38-daytrade`
  - 日本時間8:20頃に実行
  - `run_v38_daytrade_signal.py`
  - S&P500前日リターンを取得・CSV更新してから `--daytrade-only` で判定する。
- 通知文面:
  - `v3.8暴落後normal/downtrendリバウンド・デイトレメソッド`
  - `デイトレ買いトリガー`
  - `同日引け成で手仕舞い`
- S&P500取得に失敗した場合:
  - 既存CSVにフォールバック。
  - `trade_date` 行がまだない場合は、未来データを使わず、4日以内の直近確定S&P500リターンを使用する。
    - 総損益 `+7,283,863円`
    - 平均区間リターン `+9.71%`
    - 年率換算Sharpe `1.30`
  - 差し込み前（サブ戦略除外済み v3.8）:
    - 総損益 `+6,629,149円`
    - 平均区間リターン `+8.84%`
    - 年率換算Sharpe `1.26`
- 判断:
  - 現時点では **v3.8候補** とする。
  - サンプルが `15件` とまだ少ないため、正式採用前にフル再計算版でも確認する。

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

## 統合バックテスト
- 出力:
  - `output/backtest_v38_acquired_intervals_candidate/dataset_summary_consolidated.csv`
  - `output/backtest_v38_acquired_intervals_candidate/trade_log_consolidated.csv`
  - `output/backtest_v38_acquired_intervals_candidate/component_summary_consolidated.csv`
  - `output/backtest_v38_acquired_intervals_candidate/summary_consolidated.json`
- 統合方法:
  - 通常部は `v35 core` を土台にする
  - ただし `crash` と `post_major_active` の新規エントリー窓は通常部から除外する
  - その窓にだけ、採用済みの `v3.8` メソッドを差し込む
  - `q2_defensive_entry` と不採用の `v3.7` crash主戦略は含めない
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
  - `phase_name != surge`
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

## 追加検証: post_major_active=False の移行区間
- `post_major_crash_mode=True` だが `post_major_active=False` の区間は、通常復帰の橋渡しとして別管理する。
- 追加出力:
  - `output/optimize_v38_post_major_transition_candidate/variant_summary.csv`
  - `output/optimize_v38_post_major_transition_candidate/variant_trade_details.csv`
  - `output/optimize_v38_post_major_transition_candidate/dataset_variant_summary.csv`
  - `output/optimize_v38_post_major_transition_candidate/effective_regime_variant_summary.csv`
  - `output/optimize_v38_post_major_transition_candidate/no_lookahead_audit.csv`

### 状態の実態
- 取得済み区間では、この移行区間は `34日`。
- 内訳:
  - `concentrated`: `30日`
    - `post_crash_surge`
    - `post_crash_settling`
    - `post_crash_stable`
    - `post_crash_uptrend`
  - `dispersed`: `4日`
    - すべて `post_crash_uptrend`

### concentrated 移行区間の候補
- method候補: `v38_post_crash_transition_concentrated_etf_entry`
- 元の通常コアはこの区間から除外し、ETF候補に置換する。
- 条件:
  - `post_major_crash_mode=True`
  - `post_major_active=False`
  - `sector_mode=concentrated`
  - `phase_name != crash`
  - `surge` は除外
  - SSA履歴がある期間は `ssa_recovery_confirm=True` を要求
- 単体検証:
  - variant: `inactive_etf_ssa_when_available_no_surge`
  - 件数 `3`
  - 勝率 `66.67%`
  - 損益 `+87,829円`
  - 平均リターン `+0.58%`
  - 最大同時保有額 `4,995,600円`
  - 最大同時保有本数 `1`

### dispersed 移行区間の候補
- method候補: `v38_post_crash_transition_dispersed_core_entry`
- 別途作成した小型 top1 候補より、既存コアが移行期 `dispersed/uptrend` で拾っていた銘柄の方が有効だった。
- そのため、通常コアへ戻した扱いにはせず、`post_crash_transition_dispersed_core_continuation` としてラベルを分けて継続する。
- 条件:
  - `post_major_crash_mode=True`
  - `post_major_active=False`
  - `sector_mode=dispersed`
  - `phase_name = uptrend`
  - 現行コアの個別株エントリーを利用
- 統合上の実績:
  - 件数 `2`
  - 勝率 `100%`
  - 損益 `+342,162円`
  - 平均リターン `+9.39%`

### 統合差し込み確認
- 出力:
  - `output/backtest_v38_acquired_intervals_candidate/with_post_major_inactive_transition/summary.json`
  - `output/backtest_v38_acquired_intervals_candidate/with_post_major_inactive_transition/component_summary.csv`
  - `output/backtest_v38_acquired_intervals_candidate/with_post_major_inactive_transition/regime_component_summary.csv`
  - `output/backtest_v38_acquired_intervals_candidate/with_post_major_inactive_transition/excluded_transition_core_trades.csv`
- 差し込み前:
  - 総損益 `+7,283,863円`
  - 平均区間リターン `+9.71%`
  - 年率換算Sharpe `1.30`
  - 最大同時保有額 `9,970,269円`
- 差し込み後:
  - 総損益 `+7,432,442円`
  - 平均区間リターン `+9.91%`
  - 年率換算Sharpe `1.27`
  - 最大同時保有額 `10,490,446円`
  - 損失区間: `2024_2q`, `2024_4q`

### 判断
- `post_major_active=False` の区間は、通常局面へ全面復帰させるより別ラベルで保守的に扱う。
- 収益は改善したが、Sharpeはわずかに低下し、最大資金拘束も増加した。
- 現時点の候補は:
  - `concentrated`: 通常コアを外してETFに置換
  - `dispersed`: 既存コアを通常復帰扱いにせず、移行期専用ラベルとして継続
- 正式実装前の課題:
  - 最大資金拘束が500万円を超えるため、ETF・デイトレ・個別株の同日配分制約を追加する。
  - サンプルが薄いため、`post_crash_transition_*` は候補扱いに留める。

### 資金配分ルール候補
- `post_crash_transition_*` は正式採用せず、候補扱いで継続検証する。
- 通常復帰前の v3.8 系メソッドが同日に重なる場合、優先順位は次の候補とする。
  - `1`: デイトレ
  - `2`: ETF
  - `3`: 個別株
- 理由:
  - デイトレは当日引けで手仕舞うため、overnight risk が最も小さい。
  - ETFは個別株より分散されているが、保有型で資金拘束が大きい。
  - 個別株は上振れ余地はあるが、銘柄固有リスクと決算・材料リスクが残る。
- 暫定制約:
  - 総投入上限は `5,000,000円`。
  - デイトレは最優先で `1銘柄あたり500,000円`、最大 `2銘柄` まで予約する。
  - ETFは資金の約6割、`3,000,000円` を上限にする。
  - 個別株はETF後の残余資金でのみ実行し、残余が小さい場合は見送る。
- 注意:
  - 優先順位は新規エントリー時の資金配分に使う。
  - 既存保有を無理に売ってデイトレへ振り替える設計にはしない。
  - 次回検証では、この配分制約を入れた上で最大資金拘束・Sharpe・損失区間を再計算する。

### 資金配分ルール適用後の再計算
- 出力:
  - `output/backtest_v38_acquired_intervals_candidate/with_post_major_inactive_transition_allocated/summary.json`
  - `output/backtest_v38_acquired_intervals_candidate/with_post_major_inactive_transition_allocated/dataset_summary.csv`
  - `output/backtest_v38_acquired_intervals_candidate/with_post_major_inactive_transition_allocated/component_summary.csv`
  - `output/backtest_v38_acquired_intervals_candidate/with_post_major_inactive_transition_allocated/skipped_trades.csv`
  - `output/backtest_v38_acquired_intervals_candidate/with_post_major_inactive_transition_allocated/allocation_audit.csv`
- 前提:
  - 総投入上限 `5,000,000円`
  - 新規エントリー優先順位は `デイトレ > ETF > 個別株`
  - 既存保有は強制売却しない
  - 既存保有の売却予定日当日は保守的にまだ資金拘束しているものとして扱う
- 結果:
  - 総損益 `+5,611,333円`
  - 平均区間リターン `+8.02%`
  - 年率換算Sharpe `1.60`
  - 最大同時保有額 `4,999,849円`
  - 最大同時保有本数 `2`
  - 損失区間: `2022_4q`, `2024_2q`, `2024_4q`
  - 採用トレード `737件`
  - 資金不足による見送り `76件`
- コンポーネント別:
  - `v35_core_no_crash_no_transition`: `702件`, `+3,236,239円`
  - `v38_post_crash_concentrated_etf`: `9件`, `+1,499,270円`
  - `v38_post_crash_rebound_daytrade`: `12件`, `+454,914円`
  - `v38_post_crash_dispersed`: `10件`, `+204,725円`
- 解釈:
  - 500万円上限は守れるようになった。
  - Sharpeは無制約版より改善した。
  - ただし、ETF保有中は後続のデイトレ・個別株・transition候補が資金不足で見送られやすい。
  - `post_crash_transition_*` 候補は一部約定したがサンプルが薄いため、正式実装にはまだ入れない。

### ETF上限を下げた再検証
- 出力:
  - `output/backtest_v38_acquired_intervals_candidate/allocation_etf_cap_search/summary_compare.csv`
  - `output/backtest_v38_acquired_intervals_candidate/allocation_etf_cap_search/component_compare.csv`
  - `output/backtest_v38_acquired_intervals_candidate/allocation_etf_cap_search/dataset_compare.csv`
- 比較対象:
  - ETF上限 `3,000,000円`
  - ETF上限 `3,500,000円`
  - ETF上限 `4,000,000円`
  - ETF上限 `4,500,000円`
  - ETF上限 `5,000,000円`
- 結果:
  - ETF上限 `3,000,000円`: 総損益 `+5,611,333円`, 平均区間リターン `+8.02%`, Sharpe `1.60`, 損失区間 `2022_4q / 2024_2q / 2024_4q`
  - ETF上限 `3,500,000円`: 総損益 `+5,721,128円`, 平均区間リターン `+8.17%`, Sharpe `1.57`, 損失区間 `2022_4q / 2024_2q / 2024_4q`
  - ETF上限 `4,000,000円`: 総損益 `+5,851,720円`, 平均区間リターン `+8.36%`, Sharpe `1.52`, 損失区間 `2022_4q / 2024_2q / 2024_4q / 2025_2q`
  - ETF上限 `4,500,000円`: 総損益 `+5,967,308円`, 平均区間リターン `+8.52%`, Sharpe `1.48`, 損失区間 `2022_4q / 2024_2q / 2024_4q / 2025_2q`
  - ETF上限 `5,000,000円`: 総損益 `+5,909,646円`, 平均区間リターン `+8.44%`, Sharpe `1.41`, 損失区間 `2022_4q / 2024_2q / 2024_4q / 2025_2q`
- 判断:
  - 安全性・Sharpe重視では ETF上限 `3,000,000円` が最有力。
  - 総損益重視では ETF上限 `4,500,000円` が最大だが、`2025_2q` の損失が残る。
  - `2025_2q` はETF上限を `3,000,000円` に下げると `+67,271円` まで改善し、損失区間から外れる。
  - 理由は、ETF満額保有で潰れていた `post_crash_dispersed` と `post_crash_transition_dispersed_core_continuation` が一部約定できるため。
- 暫定本命:
  - ETF上限 `3,000,000円`
  - 500万円運用時にETFへ全額投入しない。
  - 残り約200万円をデイトレ・個別株・transition候補のために残す。
- 採用方針:
  - 500万円運用に対してETFは約6割投入が妥当と判断し、ETF上限は `3,000,000円` とする。
  - この配分を v3.8 の資金配分ルール正式候補として扱う。

## v3.8候補: post-crash surge 予測デイトレ

### 現在の扱い
- **不採用**。
- `surge` / `post_crash_surge` は基本的に取りに行かない。
- 以下は検証ログとして残すが、ライブ通知・統合バックテストには含めない。

### 目的
- `post_crash_surge` になった翌日に後追いするのではなく、前夜の米国株反発から翌営業日の日本株急反発を読む。
- 寄り付き後の分足確認は使わず、寄り前までに確定している情報だけで候補を作る。
- ETFやテーマ分類ではなく、前営業日までに資金が集中した個別株を出来高・売買代金・値動きで選ぶ。

### 因果条件
- `signal_date`: 日本株の前営業日
- `trade_date`: 翌営業日
- `trade_date` の寄り前に分かっている前夜S&P500リターンを使う。
- `trade_date` の日中高値・終値はバックテスト上の約定・出口判定にだけ使う。

### 実務候補
- method候補: `v38_post_crash_surge_predict_moneyflow_daytrade`
- 対象:
  - `post_major_crash_mode=True`
  - 前営業日の phase が `normal` または `downtrend`
    - 表示上は `post_crash_normal` / `post_crash_downtrend`
  - 前夜 `S&P500 >= +1.0%`
- 銘柄条件:
  - `ret1 >= +5%`
  - `ret5 >= +12%`
  - `vol_ratio20 >= 2.0`
  - `20日高値ブレイク`
  - `close_pos >= 0.60`
  - `upper_shadow_ratio <= 0.25`
  - 資金集中スコア上位 `2銘柄`
- 売買:
  - 翌営業日に前日高値ブレイクで買い
  - 同日引けで売り
  - 1銘柄あたり50万円枠

### 検証結果
- 出力:
  - `output/backtest_predict_post_crash_surge_moneyflow_sp500_candidate/variant_summary.csv`
  - `output/backtest_predict_post_crash_surge_moneyflow_sp500_candidate/best_variant_trades.csv`
- 候補: `breakout_flow_normal_downtrend_top2`
- 結果:
  - 候補 `32件`
  - 約定 `27件`
  - 対象 `4区間`
  - 平均区間リターン `+4.00%`
  - 年率換算Sharpe `3.44`
  - 損失区間 `0`
- 実際に `post_crash_surge` 当日に約定した分:
  - `4件`
  - 平均リターン `+1.40%`

### 除外すべき前日局面
- 前日 `post_crash_high_vol` は全敗。
- 前日 `crash` は平均が弱く、`capitulation_end` への移行で負けやすい。
- したがって、現時点では `normal/downtrend` のみを実務候補とする。

## surge限定候補: 夜間先物主導の資金集中株デイトレ

### 現在の扱い
- **不採用**。
- 夜間先物データが実務的に取得できるまでは、自動トリガー化しない。
- 手動で候補リストを出すことは可能だが、正式なv3.8売買メソッドにはしない。

### 位置づけ
- `post_crash_surge` 当日にETFへ入ると、寄付き時点でETFが大きく上昇済みになりやすい。
- そのため、ETFではなく、前日までに資金が集まっている個別株へデイトレで便乗する。
- method候補: `v38_post_crash_surge_futures_leader_daytrade`
- 正式実装前の候補扱い。

### 外部トリガー
- 主トリガーは夜間の日経先物。
- 補助トリガーとして `S&P500` または米国テーマ株指数を使う。
- 暫定条件:
  - `night_nikkei_futures_ret >= +1.5%`
  - または `S&P500 >= +2.0%` かつ `night_nikkei_futures_ret >= +0.8%`
- 目的:
  - 日経平均が当日 `post_crash_surge` になりそうな日だけを読む。
  - すでに上がったETFを追わず、資金集中株へ寄り付きから短期で乗る。

### 候補銘柄の出し方
- `trade_date` の前営業日までに確定している価格・出来高だけを使う。
- 候補条件:
  - `ret1 >= +4%`
  - `ret5 >= +8%`
  - `vol_ratio20 >= 1.8`
  - `close_pos >= 0.65`
  - `upper_shadow_ratio <= 0.25`
  - `value_traded` が市場内で上位
  - `prev20_high` 近辺またはブレイク済み
- スコア:
  - `ret1_rank`
  - `ret5_rank`
  - `vol_ratio20_rank`
  - `value_traded_rank`
  - `close_pos_rank`
- 例:
  - 半導体相場なら、アドバンテスト、キオクシア、東京エレクトロン、ディスコ等が上位に出るべき。
  - 海運相場なら、日本郵船、商船三井、川崎汽船等が上位に出るべき。
- ただし銘柄名を固定せず、直近の資金集中とテーマクラスタで自動抽出する。

### テーマクラスタ補助
- 個別銘柄だけでなく、同一セクター・テーマ内の資金集中を確認する。
- その日の上位候補が同一テーマに複数出ている場合、そのテーマを優先する。
- テーマ例:
  - 半導体
  - AI/データセンター
  - 海運
  - 防衛
  - 銀行/保険
- 良ファンダ判定は必須にしない。
- surge便乗デイトレでは、ファンダよりも「資金がそこへ集中しているか」を優先する。

### IN方法
- 実務候補は2本で比較する。
- `open_follow`:
  - 夜間先物が強い日は、候補上位1から2銘柄に寄り成で入る。
  - 寄付き価格が前日終値比で極端に高すぎる場合は見送り。
  - 暫定上限: `open_gap <= +12%`
- `prev_high_break`:
  - 前日高値を上抜いたら入る。
  - 寄付きがすでに前日高値を超えている場合は寄付きで約定扱い。
  - 寄り天を避けるため、こちらを保守候補にする。

### EXIT方法
- 原則は同日引け成り。
- リスク管理:
  - デイトレ1銘柄あたり `500,000円`
  - 最大 `2銘柄`
  - 日中損切り候補 `-3%`
  - ただし日足バックテストでは、日中安値到達時の損切りは保守的に別集計する。

### 先読み禁止
- `trade_date` の候補銘柄選定には、`signal_date` までの日本株価格だけを使う。
- 外部トリガーは、寄付き前に取得可能な夜間先物と米国株終値だけを使う。
- `trade_date` の日経 `surge` 判定、日中高値、終値は候補選定には使わない。
- 日中高値・終値は約定判定と出口判定にのみ使う。

### データ上の注意
- 現在ローカルにある `data/external_market/nikkei_futures_daily.csv` は日次先物データであり、真の夜間引け専用データではない。
- したがって初回検証では夜間先物の proxy として扱う。
- 実務運用前には、夜間セッション終値または寄付き前先物気配を取得できる形に更新する。

### 検証方針
- まず過去 `post_major_crash_mode=True` の全区間で検証する。
- 対象は `trade_date` が実際に `post_crash_surge` だった日だけでなく、トリガーが出た全日を集計する。
- 比較軸:
  - `open_follow`
  - `prev_high_break`
  - `night_futures >= +1.0% / +1.5% / +2.0%`
  - `S&P500 >= +1.0% / +2.0%`
  - top1 / top2
  - テーマクラスタ必須 / 不要
- 評価:
  - トレード数
  - 勝率
  - 平均リターン
  - 区間損益
  - 損失区間
  - actual `post_crash_surge` 捕捉率
  - ETFより優れているか

### 初回検証結果
- 出力:
  - `output/backtest_v38_post_crash_surge_futures_leader_daytrade_candidate/variant_summary.csv`
  - `output/backtest_v38_post_crash_surge_futures_leader_daytrade_candidate/best_variant_trades.csv`
  - `output/backtest_v38_post_crash_surge_futures_leader_daytrade_candidate/practical_fut2_extreme_top1_break/trade_log.csv`
  - `output/backtest_v38_post_crash_surge_futures_leader_daytrade_candidate/practical_fut2_extreme_top1_break/dataset_summary.csv`
  - `output/backtest_v38_post_crash_surge_futures_leader_daytrade_candidate/actual_surge_external_trigger_audit.csv`
- 全体Sharpe最良:
  - variant: `sp2_and_fut0p8_strict_break_top2_break_prev_high`
  - 条件: `S&P500 >= +2.0%` かつ `日経先物proxy >= +0.8%`
  - 候補: `ret1 >= +4%`, `ret5 >= +8%`, `vol_ratio20 >= 1.8`, `prev20_high break`, top2
  - IN: 前日高値ブレイク
  - 件数 `7`
  - 勝率 `57.14%`
  - 損益 `+113,757円`
  - 平均リターン `+4.15%`
  - 年率換算Sharpe `7.73`
  - 損失区間 `0`
  - ただし actual `post_crash_surge` 捕捉 `0件`
- surge捕捉を重視した実務候補:
  - variant: `fut_ge_2p0_extreme_top1_break_prev_high_practical`
  - 条件: `日経先物proxy >= +2.0%`
  - 候補:
    - `ret1 >= +8%`
    - `ret5 >= +15%`
    - `vol_ratio20 >= 2.5`
    - `close_pos >= 0.70`
    - `upper_shadow_ratio <= 0.20`
    - `prev20_high break`
    - top1
  - IN: 前日高値ブレイク
  - 件数 `8`
  - 勝率 `62.50%`
  - 損益 `+221,200円`
  - 平均リターン `+5.59%`
  - 年率換算Sharpe `2.21`
  - 損失区間 `0`
  - actual `post_crash_surge` 捕捉 `1件`
  - actual `post_crash_surge` 内リターン `+21.07%`
- 実務候補の約定例:
  - `2024-09-27 4222.T 児玉化学工業`: actual `post_crash_surge`, `+99,400円`, `+21.07%`
  - `2025-05-13 2673.T 夢みつけ隊`: actual `post_crash_normal`, `+132,000円`, `+26.67%`
  - `2024-11-07 3077.T ホリイフードサービス`: actual `post_crash_normal`, `+44,000円`, `+9.35%`
- 悪化した条件:
  - 前日 `surge` も許可し、`sp2_or_fut1p5` で広く拾う版は actual surge 捕捉数は増えるが、損益が悪化した。
  - 寄り成 `open_follow` は捕捉数は増えるが、寄り天・高寄り掴みの負けが増えた。

### 日経先物 + 日経225制約の追加検証
- 背景:
  - 日経平均は寄与度上位銘柄に左右されやすく、TOPIXや個別株全体の動きと乖離することがある。
  - よって日経先物を外部トリガーにする場合、候補銘柄は原則 `日経225採用銘柄` に寄せる。
  - TOPIX proxy にはローカル価格 `1306.T` を使用する。
- 追加した比較軸:
  - current `nikkei225` 採用銘柄制約
  - `TOPIX proxy >= +0.5%`
  - `日経先物proxy - TOPIX proxy >= +1.0%` かつ `TOPIX proxy < +1.0%`
- 注意:
  - current `nikkei225` 採用銘柄リストはライブ運用向けの制約。
  - 過去バックテストにそのまま使うと、厳密な歴史的採用銘柄ではないため、正式な過去検証値としては割り引く。
- 追加検証での最良候補:
  - variant: `fut_ge_1p0_nk225_strict_break_top2_break_prev_high`
  - 条件: `日経先物proxy >= +1.0%`
  - 候補: current `nikkei225` 採用銘柄、`ret1 >= +3%`, `ret5 >= +6%`, `vol_ratio20 >= 1.5`, `prev20_high break`, top2
  - IN: 前日高値ブレイク
  - 件数 `16`
  - 勝率 `50.00%`
  - 損益 `+64,791円`
  - 平均リターン `+0.61%`
  - 損失区間 `0`
  - actual `post_crash_surge` 捕捉 `2件`
  - 最大資金拘束 `1,994,000円`
- 解釈:
  - 小型材料株の極端リターンより収益性は落ちるが、日経先物との整合性は高い。
  - `TOPIX proxy >= +0.5%` を必須にすると、日経225制約版では利益が小さくなる。
  - `日経先物だけ強くTOPIXが弱い` 条件は今回の検証では悪化した。
  - したがって「日経先物が強いから小型材料株に入る」のではなく、「日経先物が強いなら225採用銘柄の直近資金集中株だけを触る」が実務寄り。

### 判断
- 日経先物をトリガーにする場合の本命は、現時点では `日経先物proxy >= +1.0% + 日経225採用銘柄 + 資金集中 top2 + 前日高値ブレイク`。
- 小型材料株版の `日経先物proxy >= +2.0% + 極端資金集中 top1 + 前日高値ブレイク` は参考候補に下げる。
- ETFではなく個別株デイトレに寄せる方針は妥当だが、日経先物連動なら対象は日経225採用銘柄に限定する。
- ただし actual `post_crash_surge` 捕捉はまだ少ない。
- 原因:
  - `nikkei_futures_daily.csv` は2021年以降で、2020年の大きなsurgeを検証できない。
  - 現在の先物データは夜間セッション専用ではなく proxy。
  - actual surge の多くはS&Pだけで立っており、先物proxyがない/弱い日がある。
- 次の課題:
  - 真の夜間先物データを取得する。
  - 米国テーマ株、例: 半導体なら MU / WDC / NVDA / AMD / SOX、海運なら関連米国株またはバルチック指数 proxy を追加する。
  - テーマクラスタ抽出を改善し、半導体・海運・防衛などの資金集中をより正確に拾う。

## v3.8候補: 日経寄与テーマ関連小型デイトレ

### 現在の扱い
- **正式採用しない**。
- 関連小型候補の抽出は可能だが、`crash/surge` を取りに行く目的では使わない。
- 研究ログとして残すが、ライブ通知・統合バックテストには含めない。

### 目的
- 日経先物が強い日に小型株全体へ広げるのではなく、日経225寄与銘柄が動いているテーマの関連小型だけを対象にする。
- 伝播ルートは `日経先物 -> 日経寄与テーマ親銘柄 -> 関連小型株` とする。

### 検証の前提
- 出力:
  - `output/backtest_v38_nikkei_contributor_related_small_daytrade_candidate/variant_summary.csv`
  - `output/backtest_v38_nikkei_contributor_related_small_daytrade_candidate/robust_variant_trades.csv`
  - `output/backtest_v38_nikkei_contributor_related_small_daytrade_candidate/robust_variant_dataset_summary.csv`
  - `output/backtest_v38_nikkei_contributor_related_small_daytrade_candidate/robust_variant_actual_regime_summary.csv`
  - `output/backtest_v38_nikkei_contributor_related_small_daytrade_candidate/robust_variant_ecosystem_summary.csv`
- 関連銘柄マップ:
  - `output/ecosystem_connections_2026_1q_candidate/ecosystem_memberships.csv`
- 運用前提:
  - エコシステム関係は半導体・金融・電力・防衛などの大枠では大きく変わりにくいとみなす。
  - 四半期別のエコシステム再構築は入手性・作業量の観点から実務上困難なため、現行 `2026_1q` 由来のエコシステム接続を全検証区間へ適用する。
  - ただし、親テーマと関連小型の価格・出来高条件は必ず `signal_date` までのデータだけで判定する。

### 親テーマ
- 暫定では次を日経寄与テーマ親銘柄として扱う。
  - 半導体関連: `8035`, `6857`, `6146`, `6723`, `4063`, `6920`, `6971`, `6526`
  - データセンター関連: `9984`, `6701`, `6702`, `6758`, `5801`, `5803`, `6501`
  - 防衛・宇宙関連: `7011`, `7012`, `7013`, `5631`, `6501`
  - 電力・系統関連: `9501`, `9502`, `9503`, `5801`, `5803`, `6501`
  - 金融決済関連: `8306`, `8316`, `8411`, `8604`, `8766`, `8750`
  - 自動車関連: `7203`, `7267`, `7201`, `7269`, `6902`
  - 海運関連: `9101`, `9104`, `9107`

### robust候補
- variant: `sp2_or_fut1p5_parent_soft_related_strict_top2_incl_signal_surge_downtrend_capitulation_surge`
- 外部トリガー:
  - `S&P500 >= +2.0%`
  - または `日経先物proxy >= +1.5%`
- 親銘柄条件:
  - `ret1 >= +1.5%`
  - `ret5 >= +3.0%`
  - `vol_ratio20 >= 1.1`
  - 親銘柄数 `1以上`
- 関連小型条件:
  - 日経225採用銘柄は除外
  - 1単元コスト `500,000円以下`
  - 親テーマと同一エコシステム
  - `ret1 >= +3.5%`
  - `ret5 >= +7.0%`
  - `vol_ratio20 >= 1.6`
  - `close_pos >= 0.62`
  - `upper_shadow_ratio <= 0.28`
  - `prev20_high break`
  - top2
- signal局面:
  - `post_crash_downtrend`
  - `post_crash_capitulation_end`
  - `post_crash_surge`
  - `post_crash_normal` は除外
- 売買:
  - 翌営業日に前日高値ブレイクでIN
  - 当日引けでEXIT
  - 1銘柄 `500,000円`

### robust候補の結果
- トレード数 `9`
- 勝率 `77.78%`
- 損益 `+159,110円`
- 平均リターン `+3.73%`
- 対象区間 `3`
- 平均区間リターン `+1.06%`
- 年率換算Sharpe `5.84`
- 損失区間 `0`
- actual `post_crash_surge` 捕捉 `4件`
- 最大資金拘束 `982,367円`
- 先読み確認:
  - 全9件で `signal_date < trade_date`

### 局面別
- actual `post_crash_downtrend`: `3件`, 勝率 `66.67%`, 損益 `+83,210円`
- actual `crash`: `2件`, 勝率 `100%`, 損益 `+44,400円`
- actual `post_crash_surge`: `4件`, 勝率 `75.00%`, 損益 `+31,500円`

### エコシステム別
- 金融決済関連: `2件`, 勝率 `100%`, 損益 `+87,210円`
- 半導体関連: `7件`, 勝率 `71.43%`, 損益 `+71,900円`

### 判断
- この手法は候補として有望。
- 四半期別エコシステム再構築は行わず、現行エコシステムマップを「概ね不変」とみなして候補評価する。
- `post_crash_normal` では負けが集中したため、この手法では除外する。
- 位置づけは `post_crash_surge / post_crash_downtrend / capitulation_end` の短期便乗デイトレ候補。

## post_major_active=False 移行期間の厳格化

### 現在の採用方針
- `post_major_crash_mode=True` かつ `post_major_active=False` の通常復帰前区間は、通常局面とは別扱いにする。
- 移行期間ETFは不採用。
- 移行期間の個別株は、既存coreの継続エントリーを条件付きで許可するだけに限定する。

### 買い許可条件
- `post_major_crash_mode=True`
- `post_major_active=False`
- `phase_name=uptrend`
- `sector_mode=dispersed`
- 前営業日も `post_major_active=False`
- 前営業日も `phase_name=uptrend`
- 前営業日も `sector_mode=dispersed`
- `price_top3_share <= 0.50`
- `positive_count >= 15`
- 1日1銘柄のみ

### 除外条件
- `crash` / `surge` は常に除外。
- `concentrated` の移行ETFは除外。
- 移行期間のcore候補でも、上記条件を満たさないものは除外する。

### 再検証結果
- 出力:
  - `output/backtest_v38_acquired_intervals_candidate/with_post_major_inactive_transition_allocated_etf300/summary.json`
  - `output/analyze_v38_transition_deep_validation_candidate/scenario_summary.csv`
  - `output/analyze_v38_transition_deep_validation_candidate/transition_trade_details_allocated.csv`
- 500万円、ETF上限300万円、デイトレ優先で再配分。
- 移行メソッドなし:
  - 損益 `+5,590,147円`
  - Sharpe `1.476`
- 厳格化後の現行候補:
  - 損益 `+5,769,904円`
  - Sharpe `1.556`
  - 最大資金拘束 `4,991,060円`
  - 採用移行トレード `1件`
- 採用された移行トレード:
  - `2025_2q 6554.T`
  - `buy_date=2025-05-23`
  - `buy_phase=uptrend`
  - `buy_effective_regime=post_crash_uptrend`
  - 損益 `+179,757円`
- 先読み・対象外混入監査:
  - `buy_phase` に `crash/surge` 混入なし。
  - `buy_effective_regime` に `crash/post_crash_surge` 混入なし。

### 判断
- 移行ETFは資金競合と保有中の崩れが大きく、正式採用しない。
- 移行個別株はサンプルが1件まで減ったため、過剰な期待は置かない。
- ただし、空白期間を完全に放置しないための保守的な候補としては残す。
- 正式運用では `v38_post_crash_transition_dispersed_core_continuation` を厳格条件つき候補として扱う。

## 参照
- `output/analyze_post_crash_three_method_framework_candidate/POST_CRASH_THREE_METHOD_PLAN.md`
- `output/analyze_post_crash_three_method_framework_candidate/post_crash_state_summary.csv`
- `output/analyze_post_crash_three_method_framework_candidate/candidate_method_summary.csv`
- `output/optimize_post_major_dispersed_prev_high_candidate/practical_vs_limited_top1_tp10_sl5.csv`
- `output/optimize_post_major_dispersed_prev_high_candidate/practical_guard_variants_top1.csv`
- `output/optimize_post_major_dispersed_prev_high_candidate/practical_all_post_major_dispersed_top1_tp10_sl5_trades.csv`
- `output/optimize_v38_post_major_transition_candidate/variant_summary.csv`

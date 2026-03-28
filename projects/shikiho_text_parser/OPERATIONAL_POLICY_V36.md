# 運用方針 v3.6

## 概要
- 基本処方は `v3.5 + 50万分割ルール`
- `major_crash` 後だけ、通常の個別株ロジックをそのまま使わず、**価格集中判定**で ETF と個別株を切り替える
- `concentrated` 側は単一ETF固定ではなく、**ETF候補群からシグナル点灯 + 週間強度で選ぶ**
- 日経局面ラベルは `1` 営業日遅れで利用
- 現在の正式候補

## 50万分割ルール
- `100株` の約定金額が `50万円未満`
  - `50万円` を超えない最大株数を購入
- `100株` の約定金額が `50万円以上`
  - `100株` のみ購入

## post-major-crash 状態
### 1. `major_crash`
- `phase_name == crash`
- かつ
  - `ret5 <= -8%`
  - または `dd20 <= -14%`

### 2. `post_major_crash_mode`
- `major_crash` で開始
- `stable / uptrend` かつ `dd20 >= -5%` が `5営業日連続` したら終了

## 集中判定
### `concentrated`
次のどれかを満たす
- `top_share >= 0.45`
- `HHI >= 0.25`
- `sector_count <= 3`
- 直近 `5営業日` リターンプラス銘柄数 `< 4`
- 上位 `3銘柄` の `5営業日` リターン寄与シェア `>= 0.55`

### `dispersed`
- 上記以外

## 売買ルール
### Method名称
- `condition2` : `上昇時メソッド`
- `breakout_1.5` : `安定局面メソッド`
- `q3_post_high_vol` : `反発局面メソッド`
- `no_trade` : `no_trade`
- `post_major_multi_etf_entry` : `crash時ETFメソッド`
- `post_major_prev_high_break_entry` : `暴落後全体上昇時メソッド`

### Condition名称
- `major_crash` : `大暴落`
- `post_major_crash_mode` : `大暴落後回復モード`
- `concentrated` : `集中相場`
- `dispersed` : `分散相場`
- `crash` : `暴落局面`
- `high_vol` : `高ボラ局面`
- `capitulation_end` : `投げ売り終盤局面`
- `settling` : `落ち着き始め局面`
- `normal` : `通常局面`
- `uptrend` : `上昇局面`
- `stable` : `安定局面`
- `downtrend` : `下降局面`
- `surge` : `急騰局面`
- `reversal_up` : `上方反転局面`
- `reversal_down` : `下方反転局面`
- `overheated_range` : `過熱持ち合い局面`
- `weak_uptrend` : `弱い上昇局面`
- `raw_post_crash_high_vol` : `暴落直後高ボラ局面`
- `rebound_confirmed_post_crash_high_vol` : `暴落後反発確認済み高ボラ局面`
- `generic_high_vol` : `一般高ボラ局面`

### `post_major_crash_mode` 中かつ `concentrated`
- ETF候補群:
  - `1306.T`
  - `1321.T`
  - `1328.T`
  - `1489.T`
  - `1593.T`
  - `2516.T`
- 基本は `2516.T`
- 他ETFは
  - `prev_high_break` が点灯
  - 週間強度スコアが上位
  - かつ `2516.T` より一定以上強い
  ときだけ採用
- 利確 `+6%`
- 損切 `-3%`

#### ETF選択ルール
- シグナル:
  - `prev_high_break`
- 強度スコア:
  - `ret5 + 0.5 * reclaim10 + breakout_strength`
- 最低条件:
  - スコア `>= 0.02`
  - 2位との差 `>= 0.005`
- `2516.T` 基準:
  - 他ETFは `2516.T + 0.003` を上回るときだけ採用

### `post_major_crash_mode` 中かつ `dispersed`
- 個別株
- `prev_high_break`
- 当日上位 `2銘柄`
- 利確 `+6%`
- 損切 `-3%`

### それ以外
- `v3.5 + 50万分割ルール`
- 通常局面のMethod対応:
  - `normal / uptrend` : `上昇時メソッド`
  - `stable` : `安定局面メソッド`
  - `settling / reversal_up / capitulation_end / overheated_range / weak_uptrend` : `反発局面メソッド`
  - `surge / downtrend / reversal_down / raw_post_crash_high_vol` : `no_trade`

## フル比較結果
- `q2_2024`: `+5.70%`
- `2025-2Q`: `+8.50%`
- `3Q / 4Q / 4Q-2`: 未再検証

## 補足
- この改善は `post_major_crash` を含む難局面にだけ効くことを狙ったもの
- `q2_2024` では従来 `v3.6` より改善、`2025-2Q` では従来 `v3.6` より低下している
- テーマタグは使わず、価格集中で代替している

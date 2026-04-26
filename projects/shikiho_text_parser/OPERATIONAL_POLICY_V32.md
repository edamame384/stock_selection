# 運用方針 v3.2

## 目的

- `v3.1` の通常局面の強さを維持する
- `crash` 直後の `high_vol` を「選定困難局面」と定義して補正する
- `crash` では新規停止を維持し、`post-crash high_vol` でだけ ETF 反発を使う

## 基本構造

- ベース: `v3.1 + 決算前1営業日売却 + 決算後5営業日見送り`
- `crash`
  - 新規停止
- `crash` 後の `high_vol`
  - `1489.T` を `rebound_open` で使用
- それ以外
  - `v3.1` のまま

## 選定困難局面

- `crash` 後の `high_vol`
- `surge` 後の `high_vol`
- `settling` 初期
- `reversal_up` 初期

現時点の最重要は `crash` 後の `high_vol`

## テーブル方針

- `normal / uptrend / stable`
  - 標準テーブル
- `settling / reversal_up`
  - 中間テーブル
- `crash`
  - 暴落後拡張テーブルを監視用
  - 新規買いはしない
- `crash` 後の `high_vol`
  - 個別株テーブルではなく ETF 反発を優先

## 現時点の採用ETF

- `1489.T`
- シグナル: `rebound_open`

条件:
- `20日高値から -5%以上`
- `5日騰落率 <= -2%`
- `Close > MA5`
- 翌日寄りで買い

## 比較結果

- `v3.1 + pre1_post5_except_crash`: `+17.03%`
- `v3.2`: `+18.63%`

内訳:
- `q2_2024`: `-1.15% -> +5.22%`
- `3Q`: 維持
- `4Q`: 維持
- `4Q-2`: 維持

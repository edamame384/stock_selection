# 運用方針 v3.4

## 概要
- 個別株専用
- ベースは `v3.1` の因果版
- 決算ルール:
  - 決算前 `1` 営業日で売却
  - 決算後 `5` 営業日は新規購入禁止
- 日経局面ラベルは `1` 営業日遅れで利用
- 現在の正式候補

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
- `q2_2024`: `+3.37%`
- `3Q`: `+12.98%`
- `4Q`: `+31.19%`
- `4Q-2`: `+3.50%`
- 合計: `+12.76%`

## 比較
- 対象:
  - `difficult_v7`
  - `v3.4 (difficult_v11)`
- `q2_2024`
  - `difficult_v7`: `+3.37%`
  - `v3.4`: `+3.37%`
- `4Q-2`
  - `difficult_v7`: `+1.20%`
  - `v3.4`: `+3.50%`

## 現時点の整理
- `raw_post_crash_high_vol` は利益を取りに行く局面ではなく、損失抑制局面
- `weak_uptrend` は通常の `uptrend` と分けて扱う必要がある
- これを一本化したものが `v3.4`

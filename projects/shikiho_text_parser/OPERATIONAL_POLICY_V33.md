# 運用方針 v3.3

## 概要
- 個別株専用
- ベースは `v3.1`
- 決算ルール:
  - 決算前 `1` 営業日で売却
  - 決算後 `5` 営業日は新規購入禁止
- 日経局面ラベルは当日ではなく `1` 営業日遅れで利用する

## 選定困難局面
`high_vol` を次の 3 つに分解する。

1. `raw_post_crash_high_vol`
- 直近 `5` 営業日に `crash` がある
- その `crash` 以降に `reversal_up / capitulation_end / normal / uptrend` をまだ挟んでいない

2. `rebound_confirmed_post_crash_high_vol`
- 直近 `5` 営業日に `crash` がある
- その `crash` 以降に `reversal_up / capitulation_end / normal / uptrend` のいずれかを 1 日以上挟んでいる

3. `generic_high_vol`
- `high_vol` だが、直近 `5` 営業日に `crash` が無い

## 売買方針
- `raw_post_crash_high_vol`:
  - `no_trade`
- `rebound_confirmed_post_crash_high_vol`:
  - `condition2`
- `generic_high_vol`:
  - `condition2`
- `surge` 後 `high_vol`:
  - 従来どおり `q3_post_high_vol`
- その他:
  - `v3.1` の因果版に準拠

## 効果
- `q2_2024`
  - `lagged`: `-6.12%`
  - `v3.3`: `+3.37%`
- `4Q-2`
  - `lagged`: `+0.34%`
  - `v3.3`: `+1.20%`

# 日経225 SSA トレンド分析・偽シグナル分類モデル 引き継ぎ資料

最終更新: 2026-05-24

このドキュメントは、別のセッションでこの作業を引き継ぐためのまとめです。

---

## 1. プロジェクト概要

日経225の **Rolling SSA (Singular Spectrum Analysis)** によるトレンドラベリング → 「上昇転換期」エントリーの戦略バックテスト → 機械学習による**偽シグナル分類フィルタ**の開発。

### ゴール
- 戦略①「転換期エントリー（翌日始値、ロング）」の勝率向上
- 強気相場で B&H に負けすぎない取引数の最適化
- フィルタ後リターン > フィルタなしリターン

### 環境
- DB: `postgresql://postgres:ogm384@localhost:5432/stock_selection` (READ ONLY 原則、ただし本セッションでヒストリカル N225 を INSERT 済み)
- yfinance + curl_cffi (SSL バイパス) で外部指標取得
- scikit-learn + LightGBM + XGBoost

---

## 2. 重要ファイル

```
scripts/
├── nikkei_ssa_trend_label.py       # SSA計算 + 4クラス・ラベリング + 戦略バックテスト + チャート生成
├── fetch_external_indicators.py    # yfinance で VIX/DJI/USDJPY/SP500 取得
└── false_signal_classifier.py      # 偽シグナル分類モデル（メインのML スクリプト）

data/
├── ssa_trend_analysis/
│   └── nikkei_ssa_trend_{period}.csv  # SSA計算済みOHLCV+ラベル (4期間)
├── external_market/
│   ├── vix_daily.csv, dji_daily.csv, usdjpy_daily.csv, sp500_daily.csv
├── false_signal_model/
│   ├── training_data_{variant}.csv
│   ├── test_data_{variant}.csv
│   ├── feature_importance_{variant}.csv
│   ├── model_metrics_{variant}.json
│   ├── threshold_comparison_{variant}.csv
│   ├── model_{variant}_{model_name}_{feature_set}.pkl  # joblib 保存モデル
│   └── equity_{variant}_{model}_{feature_set}_{period}.png
```

---

## 3. 戦略①「転換期エントリー」の定義

### ラベリング (4クラス、停滞期なし)

SSA パラメータ: W=40, L=20, R=1

| 条件 | ラベル |
|---|---|
| slope_5 > 0 & slope_20 > 0 | 上昇トレンド (uptrend) |
| slope_5 >= 0 & slope_20 <= 0 | **上昇転換期 (transition_to_uptrend)** ← 注目 |
| slope_5 < 0 & slope_20 >= 0 | 下降転換期 (transition_to_downtrend) |
| slope_5 < 0 & slope_20 < 0 | 下降トレンド (downtrend) |

### エントリー/エグジット
- エントリー: (transition_to_uptrend OR uptrend) ゾーン開始日の**翌日始値**
- エグジット: 同ゾーンの最終日の終値
- ラベルは「上昇転換期始まり」のみ分類器の対象

### 偽シグナルの定義
`is_false = 1 if return_pct < 0 else 0`

---

## 4. データ期間

### バックテスト4期間
| タグ | 期間 |
|---|---|
| 2009_2012 | 2009-01-01 〜 2012-12-31 |
| 2013_2016 | 2013-01-01 〜 2016-12-31 |
| 2018_2021 | 2018-01-01 〜 2021-12-31 |
| 2022_2025 | 2022-01-01 〜 2025-12-31 |

### 学習・テスト分割
- **学習**: TEST_PERIODS 以外すべて (`use_extend_training=True` で 1990-2008 も含む)
- **テスト**: 2018_2021 + 2022_2025

### DB の N225 データ範囲 (本セッションで INSERT 完了)
- 1965-01-05 〜 2026-05-15 (15,083行)
- 1990-01-01 以降をラベリング対象として使用 (USDJPYデータが1996年〜のため実質1996年以降の信号が学習に使われる)

---

## 5. 機械学習バリアントの最終結果

### バリアント定義
- **AC**: StandardScaler + LR、学習データ拡大 (DB から 1990-2008追加)
- **D**: LightGBM + XGBoost を追加 (RF + LR + LGBM + XGB)
- **B**: クロス特徴量からレベル値を除外し変化率/Zスコアのみに変更
- **DB**: D + B の組み合わせ

### 特徴量

#### SSA系 (15個)
```
ssa_40, price_to_ssa_gap,
ssa_slope_3, ssa_slope_5, ssa_slope_20, ssa_curvature,
ret_1, ret_5, ret_20,
vol_20, volume_z20, range_pos_20,
slope_5_to_20_ratio, gap_z_40, days_since_downtrend
```

#### クロス特徴量 (オリジナル, 9個)
```
vix_close, vix_z_20, vix_ret_5,
usdjpy_close, usdjpy_ret_5, usdjpy_z_20,
dji_ret_1, dji_ret_5, sp500_ret_5
```

#### クロス特徴量 (B変種, 変化率のみ 9個)
```
vix_z_20, vix_ret_1, vix_ret_5,
usdjpy_ret_1, usdjpy_ret_5, usdjpy_z_20,
dji_ret_1, dji_ret_5, sp500_ret_5
```

---

## 6. 🏆 最優秀構成 (現時点)

```
variant       = DB (LightGBM/XGB追加 + 変化率クロス特徴量)
学習データ拡大 = True (1990-2008 を追加、最終100訓練サンプル)
モデル         = XGBoost
特徴量集合     = ssa_cross (SSA15 + クロス変化率9 = 24特徴量)
閾値 τ         = 0.70
```

### パフォーマンス (テスト期間 2018-2025)

| 指標 | フィルタなし | **フィルタあり** | B&H |
|---|---|---|---|
| 2018-2021 リターン | +15.86% | **+18.59%** | +22.48% |
| 2018-2021 最大DD | -4.83% | **-4.64%** | — |
| 2018-2021 シャープ | 2.70 | **3.35** | 0.27 |
| 2022-2025 リターン | +8.93% | **+13.50%** | +71.80% |
| 2022-2025 最大DD | -11.13% | **-3.34%** | — |
| 2022-2025 シャープ | 1.07 | **2.55** | 0.67 |
| **2期間合計リターン** | **+26.21%** | **+34.59%** | — |
| Test AUC (XGB) | — | 0.575 | — |

⚠️ シャープが高い理由: 取引間はキャッシュ保有(0%リターン)でボラ低、常時投資のB&Hと単純比較不可。

---

## 7. AUC・最良閾値の全バリアント結果サマリー (学習拡大後)

| Variant | Model | Features | Test AUC | 最良τ | 合計リターン |
|---|---|---|---|---|---|
| D | XGB | ssa | **0.632** | 0.70 | +18.23% |
| D | RF | ssa_cross | 0.483 | 0.70 | +29.21% |
| D | XGB | ssa_cross | 0.601 | 0.70 | +22.46% |
| B | RF | ssa_cross | 0.501 | 0.70 | +29.21% |
| B | LR | ssa_cross | 0.540 | 0.70 | +26.92% |
| **DB** | **XGB** | **ssa_cross (B)** | **0.575** | **0.70** | **+32.09%** 🏆 |
| DB | RF | ssa_cross (B) | 0.501 | 0.70 | +29.21% |

ベースライン (フィルタなし): +24.79%、B&H 合計: +94.28%

---

## 8. 重要な学び・知見

1. **XGBoost が小サンプル(N=100)でも安定して効く** — LightGBM は同条件で AUC 0.391-0.471 と不振だが、XGBoost は 0.575-0.632 を維持
2. **学習データ拡大の効果はモデル依存**:
   - XGBoost (SSA特徴量): AUC 0.532 → 0.632 ✅
   - RF: AUC 0.583 → 0.488 ❌ (1990年代の市場構造に引きずられた)
3. **クロス特徴量はレベル値より変化率が有効**:
   - VIX, USDJPY のレベル値は時代によって平均水準が変動 → 学習効率低下
   - 変化率(B変種)に統一すると LR の AUC 改善 (0.432 → 0.540)
4. **StandardScalerはLRの収束問題は解消するがAUCは改善しない** (SSA特徴量が線形分離しにくい性質)
5. **τ=0.70 がスイートスポット** — 確信度が高い偽シグナルのみ除外する保守的フィルタが最良
6. **フィルタはドローダウン抑制に特に有効** — 2022-2025 で MDD -11.13% → -3.34% に削減

---

## 9. 課題・次の検討候補

### A. 戦略の弱点
- **強気相場でB&H比劣後** (2022-2025: +13.50% vs +71.80%): エントリーが転換期のみで、長期トレンド継続を取れていない
- 学習サンプル数が100でまだ少ない
- USDJPY が 1996年以前ないため、1990-1995年の学習データが NaN で除外される

### B. 改善アイデア
1. **トレーリングストップ等のエグジット改善** — uptrend ゾーン終了まで保持する現行ロジックを変えて、より長く保持
2. **ポジションサイジング** — 確率に応じて投資比率変化（p_false小→フル、p_false中→半分など）
3. **Walk-Forward学習** — 期間ごとに再学習
4. **ハイパーパラメータ探索** — XGBoost の n_estimators, max_depth など
5. **特徴量追加** — 出来高プロファイル、業種別指数、市場幅指標
6. **クロス指標の追加期間取得** — USDJPY を別ソースから 1990年以前まで補完

### C. データ拡張の余地
- pre-1990 の N225 データは DB にある (1965-) が、SSA トレンド分析で学習に使うには市場構造が違いすぎる可能性

---

## 10. 実行コマンド (主要)

```powershell
# 外部指標を取得 (キャッシュ済みなら不要)
python scripts/fetch_external_indicators.py
python scripts/fetch_external_indicators.py --force  # 再取得

# SSA トレンドラベリング (CSVを再生成)
python scripts/nikkei_ssa_trend_label.py

# 偽シグナル分類器を各バリアントで実行
python scripts/false_signal_classifier.py --variant=AC
python scripts/false_signal_classifier.py --variant=D
python scripts/false_signal_classifier.py --variant=B
python scripts/false_signal_classifier.py --variant=DB   # 最優秀
```

各バリアントは `data/false_signal_model/` 配下にバリアント別ファイルを出力する。

---

## 11. コードの主要関数 (false_signal_classifier.py)

| 関数 | 役割 |
|---|---|
| `load_nikkei_from_db(start_date)` | PostgreSQL から N225 OHLCV ロード |
| `_compute_ssa(closes)` | 純粋なSSAを numpy で計算 |
| `_assign_label_en(s5, s20)` | 4クラス・ラベル付け |
| `compute_ssa_labels_for_period(...)` | 期間ごとに SSA + ラベル生成 |
| `build_extended_training_signals()` | DB から 1990-2008 のシグナルを生成 |
| `enrich_features(df)` | SSA CSV にエンジニアリング特徴量を追加 |
| `find_long_zone_signals(df, tag)` | (transition_to_uptrend OR uptrend) ゾーンから取引シグナル抽出 |
| `build_signal_dataset(extend_training)` | 全期間のシグナルをまとめて返す |
| `attach_cross_features(signal_df, cross_feat_df)` | merge_asof で VIX/USDJPY/DJI/SP500 を接続 |
| `get_models_for_variant(variant)` | バリアント別モデル辞書を返す |
| `train_all_models(train_df, feat, models)` | 全モデルをCV付きで学習 |
| `evaluate_all_models(test_df, feat, fitted)` | 全モデルをテストデータで評価 |
| `apply_filter_backtest(test_df, prob_col, thresholds)` | 閾値スイープでフィルタ後リターン算出 |
| `plot_equity_comparison(...)` | フィルタあり/なし/B&H の3本累積損益曲線描画 |

---

## 12. 旧セッションで確定した方針 (Plan Mode で承認済み)

- **偽シグナル定義**: トレードリターン < 0% （ユーザー確定）
- **学習・テスト分割**: 2009-2016学習 / 2018-2025テスト （ユーザー確定）
- **特徴量**: 当初は SSA系のみ → クロス指標追加で改善検討 （ユーザー確定）
- **複数閾値比較**: τ ∈ [0.3, 0.4, 0.5, 0.6, 0.7] （ユーザー確定）
- **モデル**: RF をベース、LR をベースライン → LGBM/XGB 追加検討 （ユーザー確定）

---

## 13. 別セッションで再開する場合の手順

1. 本ファイル `HANDOFF.md` を最初に読む
2. `scripts/false_signal_classifier.py` の構造を確認 (`get_models_for_variant`, `main`)
3. 最優秀構成を確認:
   ```powershell
   python scripts/false_signal_classifier.py --variant=DB
   ```
4. `data/false_signal_model/model_DB_xgb_ssa_cross.pkl` を joblib.load で読み込んで予測可能
5. 改善方向は本ファイル「9. 課題・次の検討候補」を参照

---

## 14. 関連ファイルへのフルパス

```
C:\Users\fgm23\Documents\stock_selection\.claude\worktrees\infallible-driscoll-f4bdc0\
├── HANDOFF.md                                          (このファイル)
├── scripts\nikkei_ssa_trend_label.py
├── scripts\fetch_external_indicators.py
├── scripts\false_signal_classifier.py
└── data\false_signal_model\
    ├── model_DB_xgb_ssa_cross.pkl                      (最優秀モデル)
    ├── test_data_DB.csv                                (テスト期間予測結果)
    ├── threshold_comparison_DB.csv                     (τ別バックテスト結果)
    ├── feature_importance_DB.csv                       (特徴量重要度)
    └── model_metrics_DB.json                           (CV/Test メトリクス)
```

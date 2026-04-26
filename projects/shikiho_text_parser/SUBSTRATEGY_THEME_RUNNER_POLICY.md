# テーマ短期戦略 方針メモ

## 位置づけ
- 旧 `crash` サブ戦略は、検証の結果 `crash` 補助専用に閉じるより、**独立した短期主力戦略**として扱う方が自然。
- 本戦略は `v3.8` 本体の補助ではなく、**テーマ・需給の強い銘柄を短期で取る別系統戦略**とする。

## 現在の基本条件
- 20日高値ブレイク
- 5日騰落率 `>= 12%`
- 出来高倍率 `>= 2.0`
- `close_pos >= 0.60`
- `upper_shadow_ratio <= 0.25`
- `theme_cluster = True` または `hard_detached = True`
- 1日上位 `2銘柄`
- 同一銘柄 5営業日クールダウン

## 全局面バックテスト結果
- 出力:
  - `output/analyze_v38_substrategy_regimes_candidate/summary_allregimes_all.json`
  - `output/analyze_v38_substrategy_regimes_candidate/effective_regime_summary_allregimes_all.csv`
  - `output/analyze_v38_substrategy_regimes_candidate/main_vs_substrategy_summary.csv`

### 全体
- `878件`
- 勝率 `83.03%`
- 平均リターン `+9.51%`
- 中央リターン `+15.0%`
- 最大損失 `-8.0%`

### 主戦略との比較
- `v38_main_core_only`
  - `779件`
  - 勝率 `53.79%`
  - 平均 `+0.93%`
- `substrategy_all_regimes`
  - `878件`
  - 勝率 `83.03%`
  - 平均 `+9.51%`

## どの局面で強いか
- `normal`
  - `291件`
  - 勝率 `85.22%`
  - 平均 `+9.48%`
- `stable`
  - `172件`
  - 勝率 `83.72%`
  - 平均 `+9.86%`
- `post_crash_normal`
  - `86件`
  - 勝率 `89.53%`
  - 平均 `+12.06%`
- `settling`
  - `39件`
  - 勝率 `87.18%`
  - 平均 `+10.53%`
- `high_vol`
  - `24件`
  - 勝率 `91.67%`
  - 平均 `+8.98%`

## 現時点の判断
- 本戦略は、もはや「暴落補助サブ戦略」ではない。
- 実質的には **テーマ・需給ブレイクを取る汎用短期戦略** である。
- したがって今後は:
  - `v3.8` 本体から概念上切り分ける
  - 独立した資金枠で管理する
  - 通常主戦略と重複する局面をどこまで許可するか別途検証する

## 今後の詰めどころ
- `normal / stable / uptrend` を許可し続けるか
- `post_crash_*` 優先で限定するか
- 主戦略と同日重複時の優先順位
- 独立資金枠の上限

## 追加整理: 許可局面と優先順位
- 比較出力:
  - `output/analyze_v38_substrategy_regimes_candidate/main_vs_substrategy_by_regime.csv`
  - `output/analyze_v38_substrategy_regimes_candidate/main_vs_substrategy_same_day_by_regime.csv`

### 方針
- サブ戦略は、主戦略より期待値が高い局面だけを優先採用する。
- `reversal_up` はサブ戦略でも弱いため不採用。

### 優先採用する局面
- `post_crash_normal`
- `post_crash_downtrend`
- `post_crash_high_vol`
- `post_crash_surge`
- `post_crash_uptrend`
- `post_crash_capitulation_end`
- `post_crash_stable`
- `post_crash_settling`
- `surge`
- `downtrend`
- `settling`
- `high_vol`
- `overheated_range`
- `crash`

### 条件付きで許可する局面
- `normal`
- `stable`
- `uptrend`

理由:
- 件数・平均リターンではサブ戦略が主戦略を大きく上回る。
- ただしこの3局面は主戦略の本来の守備範囲とも重なるため、同日重複時は資金配分ルールを明示する。

### 不採用局面
- `reversal_up`

理由:
- 件数 `4`
- 勝率 `25%`
- 平均 `-1.60%`

## 主戦略との優先順位
### 1. 常にサブ戦略優先
- `post_crash_*`
- `surge`
- `downtrend`
- `settling`
- `high_vol`
- `overheated_range`
- `crash`

理由:
- 同局面比較でも、サブ戦略の平均リターンが主戦略を大きく上回る。

### 2. 条件付き優先
- `normal`
- `stable`
- `uptrend`

暫定ルール:
- 同日重複時は **サブ戦略を優先** として問題ない。
- ただし将来、総資金拘束を詰める段階で再検討する。

理由:
- 同日重複日の比較でも
  - `normal`: 主戦略 `+0.79%` / サブ戦略 `+9.63%`
  - `stable`: 主戦略 `-0.16%` / サブ戦略 `+8.88%`
  - `uptrend`: 主戦略 `+1.24%` / サブ戦略 `+9.17%`
  とサブ戦略が優位。

## 現時点の採用案
- サブ戦略は独立短期戦略として運用する。
- 実務初版では、まず以下を採用候補とする。
  - `reversal_up` を除く全局面
  - ただし `normal / stable / uptrend` は主戦略と同日重複時に優先ルールを明示
- 優先順位:
  - **サブ戦略 > 主戦略**
  - 少なくとも現行データではこの順序の方が自然
## 現在の扱い
- この文書は **再設計用メモ** として残す。
- 先読みを除いた因果版バックテストでは、従来のサブ戦略はそのままでは採用できなかった。
- そのため、**現在の `v3.8` 運用版にはこのサブ戦略を含めない**。

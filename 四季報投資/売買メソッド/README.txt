売買メソッド

内容
- scripts/backtest_4q2_signals.py: 現在の標準売買ロジック。condition2 を既定値として保持。
- scripts/select_promising_4q2.py: 4Q-2の四季報テキストと価格データから候補銘柄を選定するスクリプト。
- data/4q2_selected_candidates.csv: 売買対象銘柄。
- data/4q2_scored_universe.csv: 全候補銘柄のスコア一覧。
- results/summary.json: condition2 のバックテスト成績。
- results/trade_log.csv: 約定履歴。
- results/roundtrip_trades.csv: 往復売買単位の損益。
- results/equity_curve.csv: 資産推移。
- results/*.csv: 逆指値幅・利確・損切・閾値の探索結果。

標準条件 condition2
- promising_score_min = 0.66
- trend_r2_min = 0.50
- sector_per_score_min = 0.55
- entry_mode = breakout_up
- entry_limit_pct = 1.5
- take_profit_pct = 8.0
- stop_loss_pct = 5.0

再現性に関する補足
- GitHub共有用の shikiho_text_parser プロジェクトでは、4Q-2 テキスト、4Q-2対象株価、全株価アーカイブ、33業種マスターを同梱済みです。
- そのため、condition2 の銘柄選定とバックテストは、新規データ取得なしで再実行できます。
- 一方で、夜間先物・指数先物の実データCSVはローカル上で未検出のため、この売買メソッドには含まれていません。
- 先物を使う戦略を再現したい場合は、別途その実データの保存と同梱が必要です。

参照先
- ..\その他\reference\bundle_manifest.json
- ..\その他\reference\data_inventory.md
- ..\その他\reference\data_inventory.csv

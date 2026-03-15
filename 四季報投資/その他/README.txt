その他

内容
- reference/README.md: shikiho_text_parser プロジェクトの元README。
- reference/summary_comparison.csv: condition1 と condition2 の成績比較。
- reference/ticker_comparison.csv: 銘柄別比較。

役割
- 売買メソッド・四季報解析メソッドのどちらにも属しにくい比較資料や参照資料を格納。

追加資料
- reference/bundle_manifest.json: GitHub共有向けに同梱したデータ件数の一覧。
- reference/data_inventory.md: 共有時に再現できる処理・できない処理の説明。
- reference/data_inventory.csv: 同内容の一覧CSV。
- reference/futures_log_manifest.json: 先物関連で見つかったログの一覧。
- reference/last_run_futures.log: 先物処理の参照ログ。実データではなくログのみ。

補足
- 全株価データは shikiho_text_parser 側で prices_full として同梱済み。
- 夜間先物・指数先物の実データCSVはローカル上で未検出のため、共有対象には含まれていません。

実行環境
- shikiho_text_parser_runtime: GitHub共有時に再実行するための独立実行環境一式です。
- 入口は shikiho_text_parser_runtime\run_pipeline.py です。
- 詳細は shikiho_text_parser_runtime\README.md を参照してください。

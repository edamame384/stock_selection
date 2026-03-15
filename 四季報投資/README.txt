四季報投資

このフォルダは GitHub 共有用に再構成済みです。

構成
- 売買メソッド: condition2 の説明資料、結果、関連CSV
- 四季報解析メソッド: 四季報テキスト解析結果と採用基準
- その他/shikiho_text_parser_runtime: 再実行可能な実行環境一式

再現実行に使う場所
- その他/shikiho_text_parser_runtime

必要環境
- Python 3.11 以上推奨
- Git LFS

セットアップ
1. git lfs install
2. pip install -r requirements.txt
3. python その他/shikiho_text_parser_runtime/run_pipeline.py --skip-bundle

補足
- 4Q-2 テキスト、4Q-2対象価格、全価格アーカイブ、業種マスターは runtime/data 配下へ同梱済みです。
- 夜間先物や指数先物の実データCSVはローカル上で未検出のため、再現対象には含めていません。
- 詳細は その他/reference と その他/shikiho_text_parser_runtime/README.md を参照してください。

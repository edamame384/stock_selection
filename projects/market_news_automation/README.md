# market_news_automation

Google News RSS を定時取得し、スナップショット蓄積と日次特徴量生成を行う独立プロジェクトです。

## 目的

- PC が起動していない時間帯も含めてニュースを継続取得する
- ライブニュースを JSONL / CSV で蓄積する
- バックテストや当日判定に使える日次特徴量を作る

## 構成

- `collect_news_snapshot.py`
  - 1回分のニュース取得を行い、JSON/JSONL/CSV に追記します
- `aggregate_daily_news.py`
  - スナップショット群から日次特徴量 CSV を作ります
- `config/queries.json`
  - 取得クエリ定義です
- `templates/github_actions_news.yml`
  - GitHub Actions の定時実行テンプレートです

## 推奨運用

- ローカルだけで運用するなら Windows タスクスケジューラ
- PC 非起動時も必要なら GitHub Actions か常時稼働 VPS

## 推奨頻度

- 最低: 1日3回
  - `07:00`
  - `15:30`
  - `23:00`
- 推奨: 1日6回

## ローカル実行

```powershell
python projects/market_news_automation/collect_news_snapshot.py
python projects/market_news_automation/aggregate_daily_news.py
```

依存関係:

```powershell
pip install -r projects/market_news_automation/requirements.txt
```

## 出力先

- スナップショット:
  - `projects/market_news_automation/output/news_snapshots.jsonl`
  - `projects/market_news_automation/output/news_snapshots.csv`
- 日次集計:
  - `projects/market_news_automation/output/daily_news_features.csv`

## GitHub Actions

`templates/github_actions_news.yml` を `.github/workflows/news_collector.yml` として配置すれば、GitHub 上で定時取得できます。

この方式なら、PC 非起動時でも動きます。

## shikiho_text_parser との接続

`projects/shikiho_text_parser/backtest_4q2_signals.py` は、既定で

- `projects/market_news_automation/output/daily_news_features.csv`

を履歴ニュース特徴量として参照できます。

つまり、ニュース蓄積側はこのプロジェクト、売買側は `shikiho_text_parser` という分担で運用できます。

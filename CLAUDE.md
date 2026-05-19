# stock_selection プロジェクト

東証上場銘柄の株価データを管理・分析するシステム。
機械学習（RandomForest）による売買シグナル生成、バックテスト、Bayesian最適化を含む。

## PostgreSQL データベース

**接続文字列**: `postgresql://postgres:ogm384@localhost:5432/stock_selection`
**エンジン**: PostgreSQL 18（ローカル）

### テーブル一覧

#### `stocks` — 銘柄マスター（4,428件）

| カラム | 型 | 説明 |
|--------|-----|------|
| symbol | VARCHAR(20) PK | 銘柄コード（例: `7203.T`, `1306.T`） |
| sector | VARCHAR(100) | 東証33業種分類（下記参照） |

**sector の分類体系**（東証33業種 + 独自追加）:
- 通常企業: `情報・通信業` / `サービス業` / `小売業` / `卸売業` など東証33業種名
- `ETF・ETN`: 1000〜2000番台コード + XXA.T形式の新規ETF（421件）
- `J-REIT`: 3200〜3499番台・8950〜8999番台（53件）
- `インフラファンド`: 9282〜9287番台（5件）
- `金融業`: 8301.T（日本銀行）・8421.T（信金中央金庫）（2件）

主要セクター件数（上位10）:
情報・通信業 633, サービス業 585, ETF・ETN 421, 小売業 349, 卸売業 310,
電気機器 234, 機械 217, 化学 207, 建設業 163, 不動産業 155

#### `stock_prices` — 日次株価（1,474万件）

| カラム | 型 | 説明 |
|--------|-----|------|
| symbol | VARCHAR(20) PK | 銘柄コード（`stocks.symbol` と対応） |
| date | DATE PK | 取引日 |
| adj_close | DOUBLE PRECISION | 調整後終値 |
| close | DOUBLE PRECISION | 終値 |
| high | DOUBLE PRECISION | 高値 |
| low | DOUBLE PRECISION | 安値 |
| open | DOUBLE PRECISION | 始値 |
| volume | BIGINT | 出来高 |

- **期間**: 2008-01-03 〜 2026-05-15（約18年分）
- **インデックス**: `(symbol)`, `(date)`, PRIMARY KEY `(symbol, date)`
- **データソース**: yfinance（Yahoo Finance）

#### `market_indices` — 市場指数（4,487件）

| カラム | 型 | 説明 |
|--------|-----|------|
| symbol | VARCHAR(20) PK | 指数コード（例: `^N225`） |
| date | DATE PK | 取引日 |
| open / high / low / close / adj_close | DOUBLE PRECISION | OHLC + 調整後 |
| volume | BIGINT | 出来高 |

- **格納中の指数**: `^N225`（日経平均）
- **期間**: 2008-01-04 〜 2026-05-15
- **インデックス**: `(date)`, PRIMARY KEY `(symbol, date)`

---

## よく使うクエリ

```sql
-- 銘柄マスターとJOINして株価取得
SELECT p.date, p.close, s.sector
FROM stock_prices p
JOIN stocks s ON p.symbol = s.symbol
WHERE p.symbol = '7203.T'
ORDER BY p.date DESC LIMIT 10;

-- セクター別の銘柄数
SELECT sector, COUNT(*) FROM stocks GROUP BY sector ORDER BY COUNT(*) DESC;

-- 特定日の全銘柄終値
SELECT symbol, close FROM stock_prices WHERE date = '2026-05-15';

-- 日経平均の取得
SELECT date, close FROM market_indices WHERE symbol = '^N225' ORDER BY date DESC LIMIT 5;

-- DB最新日確認
SELECT MAX(date)::date FROM stock_prices;
```

---

## データ更新スクリプト

| スクリプト | 場所 | 用途 |
|-----------|------|------|
| `update_stock_prices_db.py` | `scripts/` | DB の最新日翌日〜昨日の株価を yfinance で取得してINSERT |
| `import_nikkei225_to_db.py` | `scripts/` | 日経平均を指定期間取得して `market_indices` に UPSERT |
| `fetch_all_prices.py` | `scripts/` | CSVファイルへの株価ダウンロード（DB非連携） |

**SSL証明書の設定**（プロキシ環境）:
```python
os.environ["CURL_CA_BUNDLE"] = str(Path(__file__).parents[1] / "win_certs.pem")
```
`win_certs.pem` はプロジェクトルートに存在。yfinance 1.3.0（curl_cffi）で必要。

---

## ディレクトリ構成（主要部分）

```
stock_selection/
├── CLAUDE.md                    # このファイル
├── win_certs.pem                # SSL証明書（プロキシ回避用）
├── src/
│   └── stock_signal.py          # コアモジュール（ML・特徴量・シグナル生成）
├── scripts/
│   ├── update_stock_prices_db.py   # DB株価更新
│   ├── import_nikkei225_to_db.py   # 日経平均DB登録
│   ├── fetch_all_prices.py         # CSV株価ダウンロード
│   ├── run_v36_signal.py           # シグナル生成（v36戦略）
│   └── walkforward_bayes_opt.py    # Bayesian最適化
├── data/
│   ├── prices/           # 個別銘柄CSVファイル（4,411件、約1.2GB）
│   ├── external_market/  # 日経225等の外部市場データCSV
│   ├── sector_master_template.csv  # 銘柄マスターCSV（DBのsource）
│   └── watchlist.csv               # 売買候補銘柄リスト
└── projects/
    ├── quarterly_ranker/        # 四半期ランキングシステム
    ├── market_news_automation/  # ニュース収集
    └── shikiho_text_parser/     # 四季報テキスト解析
```

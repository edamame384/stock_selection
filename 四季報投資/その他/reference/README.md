# Shikiho Text Parser

会社四季報オンラインからコピペしたテキストを、分析向けの最小限の構造化データへ整形する独立プロジェクトです。

## 目的

- ナビゲーションやUI文言を落とす
- 銘柄基本情報、特色、業績、指標、財務などの必要情報だけを残す
- 1銘柄単位でも複数ファイル一括でも処理できるようにする

## 入出力

- 入力: `.txt`
- 出力:
  - `parsed/*.json`
  - `parsed/parsed_summary.csv`

## 使い方

```powershell
python projects/shikiho_text_parser/parse_shikiho_text.py `
  --input "c:\Users\mitsu\OneDrive\ドキュメント\四季報コピペツール\1301.txt" `
  --output-dir "projects/shikiho_text_parser/output"
```

ディレクトリごと処理する場合:

```powershell
python projects/shikiho_text_parser/parse_shikiho_text.py `
  --input "c:\Users\mitsu\OneDrive\ドキュメント\四季報コピペツール" `
  --output-dir "projects/shikiho_text_parser/output"
```

## 主な抽出項目

- `ticker_code`
- `market`
- `company_name`
- `disclosure_date`
- `flags`
- `categories`
- `feature_summary`
- `segment_summary`
- `shikiho_scores`
- `headline_blocks`
- `earnings_rows`
- `guidance_rows`
- `stock_indicators`
- `company_profile`
- `shareholders`
- `financials`


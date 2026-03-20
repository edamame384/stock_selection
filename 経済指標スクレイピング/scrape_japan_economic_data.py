"""日本の経済指標をスクレイピングするスクリプト

APIキー不要のデータソースを使用:
- e-Stat Statistics Dashboard API: GDP、CPI、失業率、鉱工業生産指数
- BOJ Time-Series Data Search API: 政策金利（基準割引率）、コールレート、短観（業況判断DI）

使い方:
    # すべての指標を取得
    python scrape_japan_economic_data.py

    # 特定の指標のみ取得
    python scrape_japan_economic_data.py --indicators cpi unemployment_rate

    # 期間を指定して取得 (e-Stat: YYYY[MM][QQ]0000形式, BOJ: YYYYMM形式)
    python scrape_japan_economic_data.py --time-from 2020010000

    # 指標コードを検索（e-Stat Dashboard API）
    python scrape_japan_economic_data.py --search-indicator "消費者物価"

データソース:
    e-Stat Statistics Dashboard API (登録不要): https://dashboard.e-stat.go.jp/en/static/api
    BOJ Time-Series Data Search API (登録不要): https://www.stat-search.boj.or.jp/index_en.html
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# e-Stat Statistics Dashboard API (登録不要)
# https://dashboard.e-stat.go.jp/en/static/api
# ---------------------------------------------------------------------------

ESTAT_BASE_DATA = "https://dashboard.e-stat.go.jp/api/1.0/Json/getData"
ESTAT_BASE_SEARCH = "https://dashboard.e-stat.go.jp/api/1.0/Json/getIndicatorInfo"

# 指標コード一覧 (19桁の指標コード)
# コード確認方法: --search-indicator オプション、または
# https://dashboard.e-stat.go.jp/ でグラフを表示してURLのindicatorCodeパラメータを確認
ESTAT_INDICATORS: dict[str, dict] = {
    "cpi": {
        "name": "消費者物価指数（総合、前年同月比）",
        "name_en": "CPI (All Items, YoY %)",
        "indicator_code": "0703010401010030000",
        "cycle": 1,  # 月次
        "regional_rank": 2,  # 全国
        "seasonal_adj": 1,  # 原数値
    },
    "cpi_core": {
        "name": "消費者物価指数（生鮮食品除く総合、前年同月比）",
        "name_en": "CPI Core (excl. Fresh Food, YoY %)",
        "indicator_code": "0703010401010030010",
        "cycle": 1,
        "regional_rank": 2,
        "seasonal_adj": 1,
    },
    "cpi_index": {
        "name": "消費者物価指数（総合、指数）",
        "name_en": "CPI (All Items, Index)",
        "indicator_code": "0703010401010090000",
        "cycle": 1,
        "regional_rank": 2,
        "seasonal_adj": 1,
    },
    "unemployment_rate": {
        "name": "完全失業率",
        "name_en": "Unemployment Rate (%)",
        "indicator_code": "0302020000000010000",
        "cycle": 1,  # 月次
        "regional_rank": 2,
        "seasonal_adj": 1,  # 原数値
    },
    "gdp_real": {
        "name": "実質GDP成長率（前期比）",
        "name_en": "Real GDP Growth Rate (QoQ %)",
        "indicator_code": "0704010203000010010",
        "cycle": 2,  # 四半期
        "regional_rank": 2,
        "seasonal_adj": 2,  # 季節調整値
    },
    "gdp_nominal": {
        "name": "名目GDP成長率（前期比）",
        "name_en": "Nominal GDP Growth Rate (QoQ %)",
        "indicator_code": "0704010205000010000",
        "cycle": 2,
        "regional_rank": 2,
        "seasonal_adj": 2,
    },
    "industrial_production": {
        "name": "鉱工業生産指数",
        "name_en": "Industrial Production Index",
        "indicator_code": "0701060100000010010",
        "cycle": 1,  # 月次
        "regional_rank": 2,
        "seasonal_adj": 2,  # 季節調整値
    },
}


def search_estat_indicators(keyword: str, lang: str = "JP") -> pd.DataFrame:
    """e-Stat Statistics Dashboard APIで指標コードをキーワード検索する。

    Args:
        keyword: 検索キーワード（例: "消費者物価", "GDP", "失業率"）
        lang: 言語 ("JP" or "EN")

    Returns:
        検索結果のDataFrame
    """
    params = {"Lang": lang, "SearchWord": keyword}
    print(f"  指標コード検索: '{keyword}'...")
    resp = requests.get(ESTAT_BASE_SEARCH, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    if "GET_STATS" in data and "STATISTICAL_DATA" in data["GET_STATS"]:
        stat_data = data["GET_STATS"]["STATISTICAL_DATA"]
        if "TABLE_INF" in stat_data:
            tables = stat_data["TABLE_INF"]
            if isinstance(tables, dict):
                tables = [tables]
            for t in tables:
                indicator = t.get("INDICATOR", {})
                rows.append({
                    "indicator_code": indicator.get("@indicatorCode", ""),
                    "indicator_name": indicator.get("$", ""),
                    "stat_name": t.get("STAT_NAME", {}).get("$", ""),
                    "cycle": t.get("CYCLE", ""),
                })

    df = pd.DataFrame(rows)
    print(f"    -> {len(df)} 件見つかりました")
    return df


def fetch_estat_indicator(
    indicator_key: str,
    config: dict,
    time_from: str | None = None,
    time_to: str | None = None,
) -> pd.DataFrame:
    """e-Stat Statistics Dashboard APIから指標データを取得する。"""
    params: dict[str, str | int] = {
        "Lang": "JP",
        "IndicatorCode": config["indicator_code"],
        "Cycle": config["cycle"],
        "RegionalRank": config["regional_rank"],
        "IsSeasonalAdjustment": config["seasonal_adj"],
        "MetaGetFlg": "Y",
    }
    if time_from:
        params["TimeFrom"] = time_from
    if time_to:
        params["TimeTo"] = time_to

    print(f"  取得中: {config['name']} ({config['name_en']})...")

    resp = requests.get(ESTAT_BASE_DATA, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # レスポンスからデータを抽出
    rows = []
    if "GET_STATS" in data and "STATISTICAL_DATA" in data["GET_STATS"]:
        stat_data = data["GET_STATS"]["STATISTICAL_DATA"]
        if "DATA_INF" in stat_data:
            data_inf = stat_data["DATA_INF"]
            # DATA_OBJ がリストまたは単一オブジェクトの場合を処理
            data_objs = data_inf.get("DATA_OBJ", [])
            if isinstance(data_objs, dict):
                data_objs = [data_objs]
            for obj in data_objs:
                value_elem = obj.get("VALUE", {})
                if isinstance(value_elem, dict):
                    rows.append({
                        "indicator": indicator_key,
                        "indicator_name": config["name"],
                        "indicator_name_en": config["name_en"],
                        "time": value_elem.get("@time", ""),
                        "value": value_elem.get("$", ""),
                        "unit": value_elem.get("@unit", ""),
                    })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = _parse_estat_time(df)
        df = df.sort_values("date").reset_index(drop=True)
    print(f"    -> {len(df)} 件取得")
    return df


def _parse_estat_time(df: pd.DataFrame) -> pd.DataFrame:
    """e-Statの時間コードを日付に変換する。

    時間コード例:
      月次:     2024100000 -> 2024-10-01
      四半期:   20241Q0000 -> 2024-01-01 (Q1)
      年:       2024CY0000 -> 2024-01-01
      年度:     2024FY0000 -> 2024-04-01
    """
    dates = []
    for t in df["time"]:
        t = str(t).strip()
        if len(t) < 4:
            dates.append(pd.NaT)
            continue

        year = t[:4]

        # 月次: 2024100000 (5,6桁目が月)
        if len(t) >= 6 and t[4:6].isdigit() and t[4:6] != "00":
            month = int(t[4:6])
            if 1 <= month <= 12:
                dates.append(pd.Timestamp(f"{year}-{month:02d}-01"))
                continue

        # 四半期: 20241Q0000 (5桁目が四半期番号、6桁目がQ)
        if len(t) >= 6 and t[5] == "Q" and t[4].isdigit():
            q = int(t[4])
            if 1 <= q <= 4:
                month = (q - 1) * 3 + 1
                dates.append(pd.Timestamp(f"{year}-{month:02d}-01"))
                continue

        # 年度: 2024FY0000
        if len(t) >= 6 and t[4:6] == "FY":
            dates.append(pd.Timestamp(f"{year}-04-01"))
            continue

        # 暦年: 2024CY0000
        if len(t) >= 6 and t[4:6] == "CY":
            dates.append(pd.Timestamp(f"{year}-01-01"))
            continue

        # それ以外は年の先頭
        dates.append(pd.Timestamp(f"{year}-01-01"))

    df["date"] = dates
    return df


# ---------------------------------------------------------------------------
# BOJ Time-Series Data Search API (登録不要、2026年2月開始)
# https://www.stat-search.boj.or.jp/api/v1/
# APIマニュアル: https://www.stat-search.boj.or.jp/info/api_manual_en.pdf
# ---------------------------------------------------------------------------

BOJ_BASE = "https://www.stat-search.boj.or.jp/api/v1/getDataCode"

BOJ_INDICATORS: dict[str, dict] = {
    "policy_rate": {
        "name": "基準割引率および基準貸付利率",
        "name_en": "Basic Discount Rate and Basic Loan Rate",
        "db": "IR01",
        "codes": ["IR01'MADR1Z@D"],
        "frequency": "daily",
    },
    "call_rate_overnight": {
        "name": "無担保コールレート（オーバーナイト物）",
        "name_en": "Call Rate (Uncollateralized Overnight)",
        "db": "FM08",
        "codes": ["FM08'FDSTOR@M"],
        "frequency": "monthly",
    },
    "tankan_manufacturing_large": {
        "name": "短観 業況判断DI 大企業・製造業",
        "name_en": "Tankan DI: Large Enterprises, Manufacturing",
        "db": "CO",
        "codes": ["TK99F1000601GCQ01000"],
        "frequency": "quarterly",
    },
    "tankan_nonmfg_large": {
        "name": "短観 業況判断DI 大企業・非製造業",
        "name_en": "Tankan DI: Large Enterprises, Non-Manufacturing",
        "db": "CO",
        "codes": ["TK99F2000601GCQ01000"],
        "frequency": "quarterly",
    },
    "tankan_all_large": {
        "name": "短観 業況判断DI 大企業・全産業",
        "name_en": "Tankan DI: Large Enterprises, All Industries",
        "db": "CO",
        "codes": ["TK99F0000601GCQ01000"],
        "frequency": "quarterly",
    },
}


def fetch_boj_indicator(
    indicator_key: str,
    config: dict,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """BOJ Time-Series Data Search APIから指標データを取得する。

    Args:
        indicator_key: 指標キー名
        config: BOJ_INDICATORS の設定辞書
        start_date: 開始日 (YYYYMM形式、月次/四半期。日次はYYYYMMDD)
        end_date: 終了日
    """
    params: dict[str, str] = {
        "format": "json",
        "lang": "en",
        "db": config["db"],
        "code": ",".join(config["codes"]),
    }
    if start_date:
        params["startDate"] = start_date
    if end_date:
        params["endDate"] = end_date

    print(f"  取得中: {config['name']} ({config['name_en']})...")

    resp = requests.get(BOJ_BASE, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    # BOJ APIのレスポンス構造を解析
    if isinstance(data, dict):
        for series_code, series_data in data.items():
            if series_code in ("status", "message"):
                continue
            if isinstance(series_data, dict):
                for date_str, val in series_data.items():
                    if date_str in ("description", "name", "unit", "frequency"):
                        continue
                    rows.append({
                        "indicator": indicator_key,
                        "indicator_name": config["name"],
                        "indicator_name_en": config["name_en"],
                        "series_code": series_code,
                        "date_raw": date_str,
                        "value": val,
                    })
            elif isinstance(series_data, list):
                for item in series_data:
                    if isinstance(item, dict):
                        rows.append({
                            "indicator": indicator_key,
                            "indicator_name": config["name"],
                            "indicator_name_en": config["name_en"],
                            "series_code": series_code,
                            "date_raw": item.get("date", ""),
                            "value": item.get("value", ""),
                        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["date"] = _parse_boj_dates(df["date_raw"])
        df = df.sort_values("date").reset_index(drop=True)
    print(f"    -> {len(df)} 件取得")
    return df


def _parse_boj_dates(series: pd.Series) -> pd.Series:
    """BOJ APIの日付文字列をTimestampに変換する。

    YYYYMMDD (日次), YYYYMM (月次), YYYYQQ (四半期, QQ=01-04) に対応。
    """
    dates = []
    for raw in series:
        raw = str(raw).strip()
        if len(raw) == 8 and raw.isdigit():
            # YYYYMMDD
            dates.append(pd.Timestamp(raw))
        elif len(raw) == 6 and raw.isdigit():
            year, mm = raw[:4], int(raw[4:6])
            if 1 <= mm <= 12:
                # YYYYMM (月次)
                dates.append(pd.Timestamp(f"{year}-{mm:02d}-01"))
            elif 1 <= mm <= 4:
                # YYYYQQ (四半期)
                month = (mm - 1) * 3 + 1
                dates.append(pd.Timestamp(f"{year}-{month:02d}-01"))
            else:
                dates.append(pd.NaT)
        else:
            dates.append(pd.NaT)
    return pd.Series(dates, index=series.index)


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------

ALL_INDICATOR_KEYS = list(ESTAT_INDICATORS.keys()) + list(BOJ_INDICATORS.keys())


def scrape_all(
    indicators: list[str] | None = None,
    output_dir: Path = Path("data"),
    time_from: str | None = None,
    time_to: str | None = None,
) -> dict[str, pd.DataFrame]:
    """指定された経済指標を一括取得する。

    Args:
        indicators: 取得する指標キーのリスト。Noneなら全指標。
        output_dir: CSV出力先ディレクトリ。
        time_from: 取得開始期間。
        time_to: 取得終了期間。

    Returns:
        {指標キー: DataFrame} の辞書。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, pd.DataFrame] = {}
    targets = indicators or ALL_INDICATOR_KEYS

    # e-Stat指標の取得
    estat_targets = [k for k in targets if k in ESTAT_INDICATORS]
    if estat_targets:
        print("\n=== e-Stat Statistics Dashboard API ===")
        for key in estat_targets:
            try:
                df = fetch_estat_indicator(
                    key, ESTAT_INDICATORS[key],
                    time_from=time_from, time_to=time_to,
                )
                results[key] = df
                time.sleep(0.5)  # APIへの負荷軽減
            except Exception as e:
                print(f"    [エラー] {key}: {e}")

    # BOJ指標の取得
    boj_targets = [k for k in targets if k in BOJ_INDICATORS]
    if boj_targets:
        print("\n=== BOJ Time-Series Data Search API ===")
        for key in boj_targets:
            try:
                df = fetch_boj_indicator(
                    key, BOJ_INDICATORS[key],
                    start_date=time_from, end_date=time_to,
                )
                results[key] = df
                time.sleep(0.5)
            except Exception as e:
                print(f"    [エラー] {key}: {e}")

    # 結果をCSVに保存
    if results:
        _save_results(results, output_dir)

    return results


def _save_results(results: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """取得結果をCSVファイルに保存する。"""
    print(f"\n=== 保存先: {output_dir.resolve()} ===")

    # 個別CSVの保存
    for key, df in results.items():
        if not df.empty:
            csv_path = output_dir / f"{key}.csv"
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"  {csv_path.name}: {len(df)} 行")

    # 統合CSVの保存（全指標の最新値サマリー）
    summary_rows = []
    for key, df in results.items():
        if df.empty:
            continue
        valid = df.dropna(subset=["value"])
        if valid.empty:
            continue
        latest = valid.iloc[-1]
        summary_rows.append({
            "indicator": key,
            "indicator_name": latest.get("indicator_name", ""),
            "indicator_name_en": latest.get("indicator_name_en", ""),
            "latest_date": latest.get("date", ""),
            "latest_value": latest.get("value", ""),
            "unit": latest.get("unit", ""),
        })
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = output_dir / "summary_latest.csv"
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"  {summary_path.name}: 最新値サマリー ({len(summary_rows)} 指標)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="日本の経済指標をスクレイピング",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # すべての指標を取得
  python scrape_japan_economic_data.py

  # CPI と失業率のみ取得
  python scrape_japan_economic_data.py --indicators cpi unemployment_rate

  # 2020年以降のデータを取得
  python scrape_japan_economic_data.py --time-from 2020010000

  # 指標コードをキーワード検索
  python scrape_japan_economic_data.py --search-indicator "GDP"

利用可能な指標:
  [e-Stat] cpi, cpi_core, cpi_index, unemployment_rate, gdp_real, gdp_nominal, industrial_production
  [BOJ]    policy_rate, call_rate_overnight, tankan_manufacturing_large, tankan_nonmfg_large, tankan_all_large
""",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent / "data",
        help="出力ディレクトリ (デフォルト: data/)",
    )
    parser.add_argument(
        "--indicators", nargs="*", default=None,
        help=f"取得する指標 (デフォルト: すべて) 選択肢: {', '.join(ALL_INDICATOR_KEYS)}",
    )
    parser.add_argument(
        "--time-from", type=str, default=None,
        help="取得開始期間 (例: e-Stat=2020010000, BOJ=202001)",
    )
    parser.add_argument(
        "--time-to", type=str, default=None,
        help="取得終了期間",
    )
    parser.add_argument(
        "--search-indicator", type=str, default=None,
        help="e-Stat Dashboard APIで指標コードをキーワード検索",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("日本経済指標スクレイピング")
    print(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 指標コード検索モード
    if args.search_indicator:
        df = search_estat_indicators(args.search_indicator)
        if not df.empty:
            print("\n検索結果:")
            print(df.to_string(index=False))
        return

    # データ取得モード
    indicators = args.indicators
    if indicators and "all" in indicators:
        indicators = None

    results = scrape_all(
        indicators=indicators,
        output_dir=args.output_dir,
        time_from=args.time_from,
        time_to=args.time_to,
    )

    print("\n" + "=" * 60)
    print(f"完了: {len(results)} 指標を取得しました")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
米国経済指標取得スクリプト (FRED API使用)
- 消費者物価指数 (CPI)
- ISM非製造業景況指数
- ISM製造業景況指数
- 非農業部門雇用者数・失業率

【事前準備】FREDの無料APIキーを取得してください:
  https://fred.stlouisfed.org/docs/api/api_key.html

【実行方法】
  export FRED_API_KEY="your_api_key_here"
  python fetch_us_indicators.py

  または引数で指定:
  python fetch_us_indicators.py --api-key YOUR_KEY
"""

import argparse
import os
import sys

import pandas as pd
from fredapi import Fred

# 保存先ディレクトリ
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "us_indicators")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 取得期間
START_DATE = "1990-01-01"

# FRED シリーズID定義
SERIES = {
    "cpi": {
        "id": "CPIAUCSL",
        "name": "消費者物価指数 (CPI, 全都市・全品目, 季節調整済)",
    },
    "ism_manufacturing": {
        "id": "NAPM",
        "name": "ISM製造業景況指数 (PMI)",
    },
    "ism_nonmanufacturing": {
        "id": "NMFCI",
        "name": "ISM非製造業景況指数 (NMI)",
    },
    "nonfarm_payrolls": {
        "id": "PAYEMS",
        "name": "非農業部門雇用者数 (千人)",
    },
    "unemployment_rate": {
        "id": "UNRATE",
        "name": "失業率 (%)",
    },
}


def main():
    parser = argparse.ArgumentParser(description="米国経済指標をFREDから取得")
    parser.add_argument("--api-key", help="FRED APIキー (環境変数FRED_API_KEYでも可)")
    parser.add_argument("--start", default=START_DATE, help="取得開始日 (YYYY-MM-DD)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("FRED_API_KEY")
    if not api_key:
        print("エラー: FREDのAPIキーが必要です。")
        print("  無料取得: https://fred.stlouisfed.org/docs/api/api_key.html")
        print("  実行方法: export FRED_API_KEY='your_key' && python fetch_us_indicators.py")
        sys.exit(1)

    fred = Fred(api_key=api_key)
    results = {}

    for key, meta in SERIES.items():
        print(f"取得中: {meta['name']} ({meta['id']}) ...", end=" ", flush=True)
        try:
            s = fred.get_series(meta["id"], observation_start=args.start)
            s.name = meta["id"]
            results[key] = s
            out_path = os.path.join(OUTPUT_DIR, f"{key}.csv")
            s.to_frame().to_csv(out_path, index_label="Date")
            print(f"OK  ({s.index[0].date()} ~ {s.index[-1].date()}, {len(s)}件)")
        except Exception as e:
            print(f"NG: {e}")

    if results:
        combined = pd.DataFrame(results)
        combined.index.name = "Date"
        combined.to_csv(os.path.join(OUTPUT_DIR, "us_indicators_combined.csv"))
        print(f"\n--- 最新6件 ---")
        print(combined.tail(6).to_string())
        print(f"\n保存先: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

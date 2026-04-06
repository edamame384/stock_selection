"""
vix_correlation.py
米国VIX と 日本VIX (日経平均ボラティリティ指数) の相関分析

【データ準備】
  以下を Yahoo Finance からダウンロードして
  results/vix/ フォルダに置いてください:
    - us_vix.csv  : ^VIX の履歴データ (2001年〜)
    - jp_vix.csv  : ^JNIV の履歴データ (2010年〜)
  ※ Yahoo Finance の CSV は "Date,Open,High,Low,Close,Adj Close,Volume" 形式

注意: 日経VIは2010年1月から公表開始のため、相関は2010年以降のみ
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

matplotlib.rcParams['font.family'] = 'IPAGothic'
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

OUT_DIR = Path('/home/user/stock_selection/四季報投資/売買メソッド/results/vix')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_vix_csv(path: Path, col_name: str) -> pd.Series:
    """Yahoo Finance ダウンロードCSVを読み込む"""
    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
    # 'Close' または 'Adj Close' を使用
    close_col = 'Close' if 'Close' in df.columns else df.columns[0]
    s = df[close_col].dropna()
    s = pd.to_numeric(s, errors='coerce').dropna()
    s.name = col_name
    return s


def main():
    print("=" * 55)
    print("VIX 相関分析")
    print("=" * 55)

    us_path = OUT_DIR / 'us_vix.csv'
    jp_path = OUT_DIR / 'jp_vix.csv'

    if not us_path.exists() or not jp_path.exists():
        print(f"\n【エラー】CSVファイルが見つかりません。")
        print(f"  {us_path}  → 存在: {us_path.exists()}")
        print(f"  {jp_path}  → 存在: {jp_path.exists()}")
        print("\nYahoo Finance から以下をダウンロードして配置してください:")
        print("  ^VIX  → us_vix.csv")
        print("  ^JNIV → jp_vix.csv")
        return

    us_vix = load_vix_csv(us_path, 'US_VIX')
    jp_vix = load_vix_csv(jp_path, 'JP_VIX')

    print(f"米国VIX: {len(us_vix)}件 ({us_vix.index[0].date()} ~ {us_vix.index[-1].date()})")
    print(f"日本VIX: {len(jp_vix)}件 ({jp_vix.index[0].date()} ~ {jp_vix.index[-1].date()})")

    # 共通日付で結合
    combined = pd.DataFrame({'US_VIX': us_vix, 'JP_VIX': jp_vix}).dropna()
    combined.index.name = 'Date'
    combined.to_csv(OUT_DIR / 'vix_combined.csv', encoding='utf-8-sig')
    print(f"\n統合CSV保存: vix_combined.csv ({len(combined)}行)")
    print(f"  共通期間: {combined.index[0].date()} ~ {combined.index[-1].date()}")

    # ── 相関係数 ────────────────────────────────────
    corr_all = combined['US_VIX'].corr(combined['JP_VIX'])
    corr_log = np.log(combined['US_VIX']).corr(np.log(combined['JP_VIX']))
    chg = combined[['US_VIX', 'JP_VIX']].pct_change().dropna()
    corr_chg = chg['US_VIX'].corr(chg['JP_VIX'])

    combined['Year'] = combined.index.year
    yearly_corr = combined.groupby('Year').apply(
        lambda g: g['US_VIX'].corr(g['JP_VIX']) if len(g) > 20 else np.nan
    ).dropna()

    print(f"\n{'='*45}")
    print("相関係数")
    print(f"{'='*45}")
    print(f"  水準値の相関 (Pearson) : {corr_all:.4f}")
    print(f"  対数値の相関           : {corr_log:.4f}")
    print(f"  日次変化率の相関       : {corr_chg:.4f}")
    print(f"{'='*45}")
    print("\n年別相関係数:")
    for yr, c in yearly_corr.items():
        bar = '█' * int(abs(c) * 20)
        print(f"  {yr}: {c:+.4f}  {bar}")

    # ── チャート ────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    fig.suptitle('米国VIX vs 日本VIX (日経平均ボラティリティ指数)', fontsize=15)

    ax1 = axes[0]
    ax1.plot(us_vix.index, us_vix.values, color='#e74c3c', lw=1.2, alpha=0.9, label='米国VIX')
    ax1_r = ax1.twinx()
    ax1_r.plot(jp_vix.index, jp_vix.values, color='#2980b9', lw=1.2, alpha=0.9, label='日本VIX')
    ax1.set_ylabel('米国VIX', fontsize=12, color='#e74c3c')
    ax1_r.set_ylabel('日本VIX', fontsize=12, color='#2980b9')
    ax1.tick_params(labelsize=11)
    ax1_r.tick_params(labelsize=11)
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    lines = [plt.Line2D([0],[0],color='#e74c3c',lw=1.5,label='米国VIX'),
             plt.Line2D([0],[0],color='#2980b9',lw=1.5,label='日本VIX')]
    ax1.legend(handles=lines, fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'VIX推移 (共通期間: {combined.index[0].date()} ~ {combined.index[-1].date()})', fontsize=12)

    ax2 = axes[1]
    ax2.scatter(combined['US_VIX'], combined['JP_VIX'], alpha=0.3, s=8, color='#555')
    m, b = np.polyfit(combined['US_VIX'], combined['JP_VIX'], 1)
    x_line = np.linspace(combined['US_VIX'].min(), combined['US_VIX'].max(), 100)
    ax2.plot(x_line, m*x_line+b, color='red', lw=1.5, label=f'回帰直線 (y={m:.2f}x+{b:.2f})')
    ax2.set_xlabel('米国VIX', fontsize=12)
    ax2.set_ylabel('日本VIX', fontsize=12)
    ax2.tick_params(labelsize=11)
    ax2.set_title(f'散布図  Pearson相関={corr_all:.4f} / 変化率相関={corr_chg:.4f}', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    colors_bar = ['#27ae60' if c >= 0.7 else '#e74c3c' for c in yearly_corr.values]
    ax3.bar(yearly_corr.index, yearly_corr.values, color=colors_bar, alpha=0.8)
    ax3.axhline(corr_all, color='navy', linestyle='--', lw=1.5,
                label=f'全期間平均 {corr_all:.4f}')
    ax3.set_ylabel('相関係数', fontsize=12)
    ax3.set_title('年別相関係数', fontsize=12)
    ax3.tick_params(labelsize=11)
    ax3.set_ylim(0, 1.05)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    chart_path = OUT_DIR / 'vix_correlation.png'
    plt.savefig(chart_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nチャート保存: {chart_path}")

    rows = [
        {'項目': '水準値Pearson相関', '値': round(corr_all, 4)},
        {'項目': '対数値相関',         '値': round(corr_log, 4)},
        {'項目': '日次変化率相関',     '値': round(corr_chg, 4)},
    ]
    for yr, c in yearly_corr.items():
        rows.append({'項目': f'{yr}年相関', '値': round(c, 4)})
    pd.DataFrame(rows).to_csv(OUT_DIR / 'vix_correlation.csv', index=False, encoding='utf-8-sig')
    print(f"相関CSV保存: {OUT_DIR / 'vix_correlation.csv'}")


if __name__ == '__main__':
    main()

import time
import warnings
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime

matplotlib.rcParams['font.family'] = 'IPAGothic'
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

OUT_DIR = Path('/home/user/stock_selection/四季報投資/売買メソッド/results/vix')
OUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = '2001-01-01'
END_DATE   = '2025-12-31'

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
}


def fetch_yahoo(ticker: str, start: str, end: str) -> pd.Series | None:
    """Yahoo Finance v8 API から日次終値を取得"""
    p1 = int(datetime.strptime(start, '%Y-%m-%d').timestamp())
    p2 = int(datetime.strptime(end,   '%Y-%m-%d').timestamp())

    url = (
        f'https://query1.finance.yahoo.com/v8/finance/chart/{ticker}'
        f'?interval=1d&period1={p1}&period2={p2}&events=history'
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
        result = data['chart']['result'][0]
        timestamps = result['timestamp']
        closes     = result['indicators']['quote'][0]['close']
        dates = pd.to_datetime(timestamps, unit='s').normalize()
        s = pd.Series(closes, index=dates, name=ticker)
        s = s.dropna()
        print(f"  {ticker}: {len(s)}件 ({s.index[0].date()} ~ {s.index[-1].date()})")
        return s
    except Exception as e:
        print(f"  ERROR [{ticker}]: {e}")
        return None


def main():
    print("=" * 55)
    print("VIX データ取得")
    print(f"期間: {START_DATE} ~ {END_DATE}")
    print("=" * 55)

    # ── データ取得 ──────────────────────────────────
    print("\n米国VIX (^VIX) 取得中...")
    us_vix = fetch_yahoo('^VIX', START_DATE, END_DATE)
    time.sleep(1)

    print("日本VIX (^JNIV) 取得中...")
    jp_vix = fetch_yahoo('^JNIV', START_DATE, END_DATE)

    if us_vix is None or jp_vix is None:
        print("データ取得失敗")
        return

    # ── CSV 保存 ────────────────────────────────────
    us_df = us_vix.reset_index()
    us_df.columns = ['Date', 'US_VIX']
    us_df.to_csv(OUT_DIR / 'us_vix.csv', index=False, encoding='utf-8-sig')
    print(f"\n米国VIX保存: us_vix.csv ({len(us_df)}行)")

    jp_df = jp_vix.reset_index()
    jp_df.columns = ['Date', 'JP_VIX']
    jp_df.to_csv(OUT_DIR / 'jp_vix.csv', index=False, encoding='utf-8-sig')
    print(f"日本VIX保存: jp_vix.csv ({len(jp_df)}行)")

    # ── 共通日付で結合 ──────────────────────────────
    combined = pd.DataFrame({'US_VIX': us_vix, 'JP_VIX': jp_vix}).dropna()
    combined.index.name = 'Date'
    combined.to_csv(OUT_DIR / 'vix_combined.csv', encoding='utf-8-sig')
    print(f"統合CSV保存: vix_combined.csv ({len(combined)}行)")
    print(f"  共通期間: {combined.index[0].date()} ~ {combined.index[-1].date()}")

    # ── 相関係数 ────────────────────────────────────
    corr_all  = combined['US_VIX'].corr(combined['JP_VIX'])
    corr_log  = np.log(combined['US_VIX']).corr(np.log(combined['JP_VIX']))

    # 年別相関
    combined['Year'] = combined.index.year
    yearly_corr = combined.groupby('Year').apply(
        lambda g: g['US_VIX'].corr(g['JP_VIX']) if len(g) > 20 else np.nan
    ).dropna()

    # 前日比変化率の相関
    chg = combined[['US_VIX', 'JP_VIX']].pct_change().dropna()
    corr_chg = chg['US_VIX'].corr(chg['JP_VIX'])

    print(f"\n{'='*45}")
    print("相関係数")
    print(f"{'='*45}")
    print(f"  水準値の相関 (Pearson) : {corr_all:.4f}")
    print(f"  対数値の相関           : {corr_log:.4f}")
    print(f"  日次変化率の相関       : {corr_chg:.4f}")
    print(f"{'='*45}")
    print("\n年別相関係数:")
    for yr, c in yearly_corr.items():
        bar = '█' * int(abs(c) * 20)
        print(f"  {yr}: {c:+.4f}  {bar}")

    # ── チャート ────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    fig.suptitle(f'米国VIX vs 日本VIX (日経平均ボラティリティ指数)', fontsize=15)

    # 上段: 時系列
    ax1 = axes[0]
    ax1.plot(us_vix.index, us_vix.values, color='#e74c3c', lw=1.2, label='米国VIX (^VIX)', alpha=0.9)
    ax1_r = ax1.twinx()
    ax1_r.plot(jp_vix.index, jp_vix.values, color='#2980b9', lw=1.2, label='日本VIX (^JNIV)', alpha=0.9)
    ax1.set_ylabel('米国VIX', fontsize=12, color='#e74c3c')
    ax1_r.set_ylabel('日本VIX', fontsize=12, color='#2980b9')
    ax1.tick_params(labelsize=11)
    ax1_r.tick_params(labelsize=11)
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    lines1 = [plt.Line2D([0],[0],color='#e74c3c',lw=1.5,label='米国VIX'),
              plt.Line2D([0],[0],color='#2980b9',lw=1.5,label='日本VIX')]
    ax1.legend(handles=lines1, fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'VIX推移 (共通期間: {combined.index[0].date()} ~ {combined.index[-1].date()})', fontsize=12)

    # 中段: 散布図
    ax2 = axes[1]
    ax2.scatter(combined['US_VIX'], combined['JP_VIX'], alpha=0.3, s=8, color='#555')
    m, b = np.polyfit(combined['US_VIX'], combined['JP_VIX'], 1)
    x_line = np.linspace(combined['US_VIX'].min(), combined['US_VIX'].max(), 100)
    ax2.plot(x_line, m*x_line+b, color='red', lw=1.5, label=f'回帰直線 (y={m:.2f}x+{b:.2f})')
    ax2.set_xlabel('米国VIX', fontsize=12)
    ax2.set_ylabel('日本VIX', fontsize=12)
    ax2.tick_params(labelsize=11)
    ax2.set_title(f'散布図  Pearson相関={corr_all:.4f} / 変化率相関={corr_chg:.4f}', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    # 下段: 年別相関
    ax3 = axes[2]
    colors_bar = ['#e74c3c' if c < 0.7 else '#27ae60' for c in yearly_corr.values]
    ax3.bar(yearly_corr.index, yearly_corr.values, color=colors_bar, alpha=0.8)
    ax3.axhline(corr_all, color='navy', linestyle='--', lw=1.5,
                label=f'全期間平均 {corr_all:.4f}')
    ax3.set_ylabel('相関係数', fontsize=12)
    ax3.set_title('年別相関係数', fontsize=12)
    ax3.tick_params(labelsize=11)
    ax3.set_ylim(0, 1.05)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    chart_path = OUT_DIR / 'vix_correlation.png'
    plt.savefig(chart_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nチャート保存: {chart_path}")

    # 相関サマリーCSV
    rows = [
        {'項目': '水準値Pearson相関', '値': round(corr_all, 4)},
        {'項目': '対数値相関',         '値': round(corr_log, 4)},
        {'項目': '日次変化率相関',     '値': round(corr_chg, 4)},
    ]
    for yr, c in yearly_corr.items():
        rows.append({'項目': f'{yr}年相関', '値': round(c, 4)})
    pd.DataFrame(rows).to_csv(OUT_DIR / 'vix_correlation.csv', index=False, encoding='utf-8-sig')
    print(f"相関CSV保存: {OUT_DIR / 'vix_correlation.csv'}")


if __name__ == '__main__':
    main()

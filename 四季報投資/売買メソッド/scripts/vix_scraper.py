"""
vix_scraper.py
米国VIX と 日本市場ボラティリティ の取得・相関分析

データソース:
  米国VIX : GitHub (datasets/finance-vix) からスクレイピング
  日本VIX代替: 保有済み日本株価データから30日実現ボラティリティを算出
               (日経VI/^JNIVは外部サイト全てブロックのため代替計算)
"""

import warnings
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from io import StringIO

matplotlib.rcParams['font.family'] = 'IPAGothic'
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

BASE_DIR  = Path('/home/user/stock_selection/四季報投資')
PRICE_DIR = BASE_DIR / 'その他/shikiho_text_parser_runtime/data/prices_full'
SECTOR_CSV = BASE_DIR / 'その他/shikiho_text_parser_runtime/data/reference/sector_master_template.csv'
OUT_DIR   = BASE_DIR / '売買メソッド/results/vix'
OUT_DIR.mkdir(parents=True, exist_ok=True)

START = '2001-01-01'
END   = '2025-12-31'
RVOL_WINDOW = 30   # 実現ボラティリティの計算窓 (日)


# ── 米国VIX: GitHubから取得 ───────────────────────────────────
def fetch_us_vix() -> pd.Series:
    url = 'https://raw.githubusercontent.com/datasets/finance-vix/master/data/vix-daily.csv'
    print(f"  米国VIX取得中: {url}")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    df['DATE'] = pd.to_datetime(df['DATE'], format='%m/%d/%Y')
    df = df.set_index('DATE').sort_index()
    s = df['CLOSE'].loc[START:END].dropna()
    s.name = 'US_VIX'
    print(f"  → {len(s)}件 ({s.index[0].date()} ~ {s.index[-1].date()})")
    return s


# ── 日本市場 実現ボラティリティ: 既存株価データから計算 ─────────
def compute_japan_rvol() -> pd.Series:
    """
    全銘柄の日次リターンを等ウェイト合成した市場インデックスを作成し、
    30日ローリング実現ボラティリティ (年率%) を返す
    """
    print("  日本市場ボラティリティ計算中...")
    master = pd.read_csv(SECTOR_CSV, encoding='utf-8-sig')
    master = master[master['sector'] != '-'].dropna(subset=['sector'])
    master['ticker'] = master['symbol'].str.replace('.T', '', regex=False)
    tickers = master['ticker'].tolist()

    all_dates = pd.date_range(start=START, end=END, freq='B')
    price_list = []
    loaded = 0

    for ticker in tickers:
        fpath = PRICE_DIR / f'{ticker}_T.csv'
        if not fpath.exists():
            continue
        try:
            df = pd.read_csv(fpath, parse_dates=['Date'], index_col='Date',
                              usecols=['Date', 'Close'])
            df = df['Close'].dropna()
            df = pd.to_numeric(df, errors='coerce').dropna()
            if len(df) < 500:
                continue
            price_list.append(df)
            loaded += 1
        except Exception:
            continue

    print(f"  → {loaded}銘柄のデータ読み込み完了")

    # 価格行列を作成して日次リターンを計算
    price_df = pd.DataFrame(price_list).T
    price_df = price_df.reindex(all_dates).ffill()
    returns  = price_df.pct_change().dropna(how='all')

    # 等ウェイト市場リターン
    market_ret = returns.mean(axis=1)

    # 30日ローリング実現ボラティリティ (年率%)
    rvol = market_ret.rolling(RVOL_WINDOW, min_periods=RVOL_WINDOW).std() * np.sqrt(252) * 100
    rvol = rvol.loc[START:END].dropna()
    rvol.name = 'JP_RVOL'
    print(f"  → {len(rvol)}件 ({rvol.index[0].date()} ~ {rvol.index[-1].date()})")
    return rvol


# ── 相関分析 & チャート ──────────────────────────────────────
def analyze_and_plot(us_vix: pd.Series, jp_rvol: pd.Series):
    # 共通日付で結合
    combined = pd.DataFrame({'US_VIX': us_vix, 'JP_RVOL': jp_rvol}).dropna()
    combined.index.name = 'Date'

    print(f"\n共通期間: {combined.index[0].date()} ~ {combined.index[-1].date()} ({len(combined)}営業日)")

    # CSV 保存
    us_vix.reset_index().rename(columns={'DATE':'Date'}).to_csv(
        OUT_DIR / 'us_vix.csv', index=False, encoding='utf-8-sig')
    jp_rvol.reset_index().rename(columns={'index':'Date'}).to_csv(
        OUT_DIR / 'jp_rvol.csv', index=False, encoding='utf-8-sig')
    combined.to_csv(OUT_DIR / 'vix_combined.csv', encoding='utf-8-sig')
    print(f"CSV保存: us_vix.csv / jp_rvol.csv / vix_combined.csv")

    # 相関係数
    corr_all = combined['US_VIX'].corr(combined['JP_RVOL'])
    corr_log = np.log(combined['US_VIX']).corr(np.log(combined['JP_RVOL']))
    chg = combined.pct_change().dropna()
    corr_chg = chg['US_VIX'].corr(chg['JP_RVOL'])

    combined['Year'] = combined.index.year
    yearly_corr = combined.groupby('Year').apply(
        lambda g: g['US_VIX'].corr(g['JP_RVOL']) if len(g) > 20 else np.nan
    ).dropna()

    print(f"\n{'='*50}")
    print("相関係数 (米国VIX vs 日本市場実現ボラティリティ)")
    print(f"{'='*50}")
    print(f"  水準値 Pearson相関  : {corr_all:+.4f}")
    print(f"  対数値相関          : {corr_log:+.4f}")
    print(f"  日次変化率の相関    : {corr_chg:+.4f}")
    print(f"{'='*50}")
    print("\n年別相関係数:")
    for yr, c in yearly_corr.items():
        bar = '█' * max(0, int(c * 20))
        print(f"  {yr}: {c:+.4f}  {bar}")

    # ── チャート ──────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    fig.suptitle(
        '米国VIX vs 日本市場ボラティリティ (30日実現ボラティリティ)\n'
        '2001年〜2025年',
        fontsize=15
    )

    # 上段: 時系列比較
    ax1 = axes[0]
    ax1.plot(combined.index, combined['US_VIX'],
             color='#e74c3c', lw=1.2, alpha=0.9, label='米国VIX')
    ax1_r = ax1.twinx()
    ax1_r.plot(combined.index, combined['JP_RVOL'],
               color='#2980b9', lw=1.2, alpha=0.9, label='日本実現ボラティリティ')
    ax1.set_ylabel('米国VIX', fontsize=12, color='#e74c3c')
    ax1_r.set_ylabel('日本RVol (%/年率)', fontsize=12, color='#2980b9')
    ax1.tick_params(labelsize=11)
    ax1_r.tick_params(labelsize=11)
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    # 主要イベントの注釈
    events = {
        '2001-09-11': '9/11',
        '2003-03-20': 'イラク戦争',
        '2008-09-15': 'リーマン',
        '2011-03-11': '東日本大震災',
        '2020-03-16': 'コロナ',
    }
    for date_str, label in events.items():
        dt = pd.Timestamp(date_str)
        if dt in combined.index or combined.index[(combined.index >= dt)].shape[0] > 0:
            ax1.axvline(dt, color='gray', linestyle=':', lw=0.8, alpha=0.6)
            ax1.text(dt, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 80,
                     label, fontsize=7, rotation=90, va='top', color='gray')
    lines = [plt.Line2D([0],[0],color='#e74c3c',lw=1.5,label='米国VIX'),
             plt.Line2D([0],[0],color='#2980b9',lw=1.5,label='日本RVol')]
    ax1.legend(handles=lines, fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'時系列比較 (共通期間: {combined.index[0].date()} ~ {combined.index[-1].date()})', fontsize=12)

    # 中段: 散布図
    ax2 = axes[1]
    # 年代別に色分け
    color_map = {
        range(2001, 2010): '#e74c3c',
        range(2010, 2020): '#2980b9',
        range(2020, 2026): '#27ae60',
    }
    label_map = {
        range(2001, 2010): '2001-2009',
        range(2010, 2020): '2010-2019',
        range(2020, 2026): '2020-2025',
    }
    plotted = set()
    for yr_range, color in color_map.items():
        mask = combined['Year'].apply(lambda y: y in yr_range)
        sub = combined[mask]
        lbl = label_map[yr_range]
        ax2.scatter(sub['US_VIX'], sub['JP_RVOL'],
                    alpha=0.4, s=8, color=color,
                    label=lbl if lbl not in plotted else '')
        plotted.add(lbl)
    m, b = np.polyfit(combined['US_VIX'], combined['JP_RVOL'], 1)
    x_line = np.linspace(combined['US_VIX'].min(), combined['US_VIX'].max(), 100)
    ax2.plot(x_line, m*x_line+b, color='black', lw=1.5,
             label=f'回帰直線 (y={m:.2f}x+{b:.2f})')
    ax2.set_xlabel('米国VIX', fontsize=12)
    ax2.set_ylabel('日本実現ボラティリティ (%)', fontsize=12)
    ax2.tick_params(labelsize=11)
    ax2.set_title(
        f'散布図  Pearson相関={corr_all:.4f} / 対数相関={corr_log:.4f} / 変化率相関={corr_chg:.4f}',
        fontsize=12
    )
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 下段: 年別相関
    ax3 = axes[2]
    bar_colors = ['#27ae60' if c >= 0.6 else ('#f39c12' if c >= 0.4 else '#e74c3c')
                  for c in yearly_corr.values]
    ax3.bar(yearly_corr.index, yearly_corr.values, color=bar_colors, alpha=0.8)
    ax3.axhline(corr_all, color='navy', linestyle='--', lw=1.5,
                label=f'全期間平均 {corr_all:.4f}')
    ax3.axhline(0, color='gray', lw=0.8)
    ax3.set_ylabel('相関係数', fontsize=12)
    ax3.set_xlabel('年', fontsize=12)
    ax3.set_title('年別相関係数 (緑≥0.6 / 橙≥0.4 / 赤<0.4)', fontsize=12)
    ax3.tick_params(labelsize=11)
    ax3.set_ylim(-0.2, 1.1)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')
    for yr, c in yearly_corr.items():
        ax3.text(yr, c + 0.02, f'{c:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    chart_path = OUT_DIR / 'vix_correlation.png'
    plt.savefig(chart_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nチャート保存: {chart_path}")

    # 相関サマリーCSV
    rows = [
        {'項目': '水準値Pearson相関',   '値': round(corr_all, 4)},
        {'項目': '対数値相関',           '値': round(corr_log, 4)},
        {'項目': '日次変化率相関',       '値': round(corr_chg, 4)},
    ]
    for yr, c in yearly_corr.items():
        rows.append({'項目': f'{yr}年相関', '値': round(c, 4)})
    pd.DataFrame(rows).to_csv(
        OUT_DIR / 'vix_correlation.csv', index=False, encoding='utf-8-sig'
    )
    print(f"相関CSV保存: {OUT_DIR / 'vix_correlation.csv'}")


def main():
    print("=" * 55)
    print("VIX 相関分析")
    print("  米国VIX     : GitHub (datasets/finance-vix)")
    print("  日本ボラティリティ: 日本株4000銘柄 30日実現ボラティリティ")
    print("=" * 55)

    print("\n【米国VIX】")
    us_vix  = fetch_us_vix()

    print("\n【日本市場ボラティリティ】")
    jp_rvol = compute_japan_rvol()

    print("\n【相関分析】")
    analyze_and_plot(us_vix, jp_rvol)


if __name__ == '__main__':
    main()

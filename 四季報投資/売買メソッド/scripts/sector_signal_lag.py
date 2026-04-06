"""
sector_signal_lag.py
セクターインデックスの買いシグナルに対し、
「その業種内で最初に買いシグナルが出た個別銘柄」との
ラグ日数と価格上昇率を分析する

MA(25/100) / 全期間 (2001-2025)
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

matplotlib.rcParams['font.family'] = 'IPAGothic'
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from sector_index_backtest import (
    TRAIN_START, TEST_END, BASE_DIR,
    build_sector_data, build_sector_index,
)

OUT_DIR = BASE_DIR / '売買メソッド/results/signal_lag'
OUT_DIR.mkdir(parents=True, exist_ok=True)

FAST = 25
SLOW = 100
# セクターシグナルの何日前まで個別株シグナルを探すか
LOOKBACK_DAYS = 200


def calc_buy_signals(price: pd.Series) -> pd.Series:
    """MA(FAST/SLOW)ゴールデンクロス日を返す (値=1, それ以外=0)"""
    maf = price.rolling(FAST, min_periods=FAST).mean()
    mas = price.rolling(SLOW, min_periods=SLOW).mean()
    sig = pd.Series(0, index=price.index)
    sig[(maf > mas) & (maf.shift(1) <= mas.shift(1))] = 1
    return sig


def analyze_sector(sector_name: str, info: dict, all_dates: pd.DatetimeIndex) -> list[dict]:
    """
    1セクターについてセクターシグナル日 × 個別銘柄シグナルの
    ラグと価格上昇率を計算して返す
    """
    sector_idx = build_sector_index(info, all_dates)
    if len(sector_idx) < SLOW + 10:
        return []

    # セクターインデックスの買いシグナル日
    sector_sig = calc_buy_signals(sector_idx)
    sector_buy_dates = sector_sig[sector_sig == 1].index.tolist()
    if not sector_buy_dates:
        return []

    # 各個別銘柄の買いシグナルを事前計算
    stock_signals = {}   # {ticker: [signal_dates]}
    for ticker, price_s in info['prices'].items():
        if len(price_s) < SLOW + 10:
            continue
        sig = calc_buy_signals(price_s)
        dates = sig[sig == 1].index.tolist()
        if dates:
            stock_signals[ticker] = dates

    rows = []
    for sec_date in sector_buy_dates:
        # セクターシグナル日より前 LOOKBACK_DAYS 日以内に
        # 買いシグナルが出た個別銘柄を探す
        window_start = sec_date - pd.Timedelta(days=LOOKBACK_DAYS)

        earliest_ticker = None
        earliest_date   = None
        earliest_price_at_signal   = None
        earliest_price_at_sec_date = None

        for ticker, sig_dates in stock_signals.items():
            price_s = info['prices'][ticker]
            for sd in sig_dates:
                if window_start <= sd <= sec_date:
                    if earliest_date is None or sd < earliest_date:
                        # 個別株シグナル日の翌営業日終値 (実際のエントリー価格)
                        entry_dates = price_s.index[price_s.index > sd]
                        if len(entry_dates) == 0:
                            continue
                        ep = float(price_s.loc[entry_dates[0]])
                        # セクターシグナル日時点の価格
                        sec_avail = price_s.index[price_s.index >= sec_date]
                        if len(sec_avail) == 0:
                            continue
                        sp = float(price_s.loc[sec_avail[0]])
                        earliest_ticker  = ticker
                        earliest_date    = sd
                        earliest_price_at_signal   = ep
                        earliest_price_at_sec_date = sp

        if earliest_date is None:
            continue   # この業種シグナルには先行個別株シグナルなし

        # ラグ (営業日数)
        biz_days_between = len(pd.bdate_range(earliest_date, sec_date)) - 1

        # 価格上昇率 (個別株エントリー価格 → セクターシグナル日)
        pct_change = (earliest_price_at_sec_date - earliest_price_at_signal) \
                     / earliest_price_at_signal * 100 if earliest_price_at_signal else np.nan

        rows.append({
            'sector':            sector_name,
            'sector_signal_date': sec_date,
            'first_stock':       earliest_ticker,
            'stock_signal_date': earliest_date,
            'lag_bdays':         biz_days_between,
            'stock_entry_price': round(earliest_price_at_signal, 2),
            'price_at_sector_signal': round(earliest_price_at_sec_date, 2),
            'pct_change_by_sector_signal': round(pct_change, 2),
        })

    return rows


def main():
    print("=" * 60)
    print("セクターシグナルの先行分析")
    print(f"MA({FAST}/{SLOW}) / 全期間 ({TRAIN_START} ~ {TEST_END})")
    print("=" * 60)

    sector_info = build_sector_data()
    all_dates   = pd.date_range(start=TRAIN_START, end=TEST_END, freq='B')

    all_rows = []
    total = len(sector_info)
    for i, (sector_name, info) in enumerate(sorted(sector_info.items(),
                                                    key=lambda x: x[1]['code'])):
        if len(info['prices']) < 2:
            continue
        print(f"[{i+1}/{total}] {sector_name} ({len(info['prices'])}銘柄)...", end=' ', flush=True)
        rows = analyze_sector(sector_name, info, all_dates)
        all_rows.extend(rows)
        print(f"{len(rows)}シグナル")

    if not all_rows:
        print("データなし")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / 'signal_lag_detail.csv', index=False, encoding='utf-8-sig')
    print(f"\n詳細CSV保存: {len(df)}件")

    # ── 統計サマリー ──────────────────────────────────
    print("\n" + "=" * 75)
    print(f"{'セクター':<20} {'シグナル数':>8} {'平均ラグ(日)':>12} {'中央値ラグ':>10} {'平均上昇率%':>12} {'中央値%':>9}")
    print("-" * 75)

    summary_rows = []
    for sector_name in sorted(df['sector'].unique()):
        sub = df[df['sector'] == sector_name]
        avg_lag  = sub['lag_bdays'].mean()
        med_lag  = sub['lag_bdays'].median()
        avg_pct  = sub['pct_change_by_sector_signal'].mean()
        med_pct  = sub['pct_change_by_sector_signal'].median()
        print(f"{sector_name:<20} {len(sub):>8} {avg_lag:>11.1f}日 {med_lag:>9.1f}日 "
              f"{avg_pct:>+11.1f}% {med_pct:>+8.1f}%")
        summary_rows.append({
            'セクター': sector_name, 'シグナル数': len(sub),
            '平均ラグ(営業日)': round(avg_lag, 1),
            '中央値ラグ(営業日)': round(med_lag, 1),
            '平均上昇率(%)': round(avg_pct, 2),
            '中央値上昇率(%)': round(med_pct, 2),
        })

    print("=" * 75)
    total_avg_lag = df['lag_bdays'].mean()
    total_med_lag = df['lag_bdays'].median()
    total_avg_pct = df['pct_change_by_sector_signal'].mean()
    total_med_pct = df['pct_change_by_sector_signal'].median()
    print(f"{'【全セクター合計】':<20} {len(df):>8} {total_avg_lag:>11.1f}日 "
          f"{total_med_lag:>9.1f}日 {total_avg_pct:>+11.1f}% {total_med_pct:>+8.1f}%")
    print("=" * 75)

    # ── CSV ──────────────────────────────────────────
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / 'signal_lag_summary.csv',
                                       index=False, encoding='utf-8-sig')

    # ── チャート ──────────────────────────────────────
    import matplotlib.dates as mdates

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(
        f'セクターシグナルと個別銘柄シグナルのラグ分析\n'
        f'MA({FAST}/{SLOW}) / 全期間 2001-2025 / {len(df)}シグナル',
        fontsize=14
    )

    # 左: ラグ分布
    ax1 = axes[0]
    ax1.hist(df['lag_bdays'], bins=40, color='#2980b9', alpha=0.8, edgecolor='white')
    ax1.axvline(total_avg_lag, color='red', lw=2, linestyle='--',
                label=f'平均 {total_avg_lag:.1f}日')
    ax1.axvline(total_med_lag, color='orange', lw=2, linestyle='-.',
                label=f'中央値 {total_med_lag:.1f}日')
    ax1.set_xlabel('ラグ (営業日数)', fontsize=12)
    ax1.set_ylabel('件数', fontsize=12)
    ax1.set_title('最初の個別株シグナル → セクターシグナルまでのラグ分布', fontsize=11)
    ax1.tick_params(labelsize=11)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 右: 価格上昇率分布
    ax2 = axes[1]
    clip = df['pct_change_by_sector_signal'].clip(-30, 60)
    ax2.hist(clip, bins=40, color='#e67e22', alpha=0.8, edgecolor='white')
    ax2.axvline(total_avg_pct, color='red', lw=2, linestyle='--',
                label=f'平均 {total_avg_pct:+.1f}%')
    ax2.axvline(total_med_pct, color='navy', lw=2, linestyle='-.',
                label=f'中央値 {total_med_pct:+.1f}%')
    ax2.axvline(0, color='black', lw=1)
    ax2.set_xlabel('価格変化率 (%)', fontsize=12)
    ax2.set_ylabel('件数', fontsize=12)
    ax2.set_title('個別株シグナル後 → セクターシグナルまでの価格変化率', fontsize=11)
    ax2.tick_params(labelsize=11)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = OUT_DIR / 'signal_lag_distribution.png'
    plt.savefig(chart_path, dpi=120, bbox_inches='tight')
    plt.close()

    # セクター別棒グラフ
    sum_df = pd.DataFrame(summary_rows).sort_values('平均ラグ(営業日)', ascending=False)
    fig2, axes2 = plt.subplots(2, 1, figsize=(16, 12))
    fig2.suptitle('セクター別 ラグ & 上昇率', fontsize=14)

    ax3 = axes2[0]
    bars = ax3.barh(sum_df['セクター'], sum_df['平均ラグ(営業日)'],
                    color='#2980b9', alpha=0.8)
    ax3.barh(sum_df['セクター'], sum_df['中央値ラグ(営業日)'],
             color='#85c1e9', alpha=0.6, label='中央値')
    for bar, val in zip(bars, sum_df['平均ラグ(営業日)']):
        ax3.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                 f'{val:.0f}日', va='center', fontsize=8)
    ax3.set_xlabel('営業日数', fontsize=12)
    ax3.set_title('平均ラグ (青:平均 / 水色:中央値)', fontsize=12)
    ax3.tick_params(labelsize=10)
    ax3.grid(True, alpha=0.3, axis='x')

    ax4 = axes2[1]
    colors_pct = ['#e74c3c' if v > 5 else ('#27ae60' if v < 0 else '#f39c12')
                  for v in sum_df['平均上昇率(%)']]
    bars2 = ax4.barh(sum_df['セクター'], sum_df['平均上昇率(%)'],
                     color=colors_pct, alpha=0.8)
    ax4.axvline(0, color='black', lw=1)
    ax4.axvline(total_avg_pct, color='navy', lw=1.5, linestyle='--',
                label=f'全体平均 {total_avg_pct:+.1f}%')
    for bar, val in zip(bars2, sum_df['平均上昇率(%)']):
        x = bar.get_width()
        ax4.text(x + (0.3 if x >= 0 else -0.3), bar.get_y() + bar.get_height()/2,
                 f'{val:+.1f}%', va='center',
                 ha='left' if x >= 0 else 'right', fontsize=8)
    ax4.set_xlabel('価格変化率 (%)', fontsize=12)
    ax4.set_title('平均上昇率 (赤>+5% / 橙0~5% / 緑マイナス)', fontsize=12)
    ax4.tick_params(labelsize=10)
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    chart2_path = OUT_DIR / 'signal_lag_by_sector.png'
    plt.savefig(chart2_path, dpi=120, bbox_inches='tight')
    plt.close()

    print(f"\nチャート保存: {chart_path.name} / {chart2_path.name}")
    print(f"結果保存先: {OUT_DIR}")


if __name__ == '__main__':
    main()

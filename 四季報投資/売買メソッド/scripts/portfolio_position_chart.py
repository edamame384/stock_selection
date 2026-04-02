"""
portfolio_position_chart.py
テスト期間(2022-2025)のポジション金額推移を可視化

固定-10%損切り / MA(25/100) / 同時4ポジション
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from sector_index_backtest import (
    TRAIN_START, TRAIN_END, TEST_START, TEST_END,
    BASE_DIR, build_sector_data, build_sector_index,
    get_largest_cap_series, find_global_best_ma, MAX_POSITIONS,
)
from sector_stoploss_compare import run_portfolio_compare

warnings.filterwarnings('ignore')

OUT_DIR = BASE_DIR / '売買メソッド/results/sector_backtest'


def run_with_position_tracking(sector_results, fast, slow, start, end,
                                 stop_mode='fixed', stop_param=0.10,
                                 initial_capital=1_000_000.0):
    """run_portfolio_compare と同じロジック + 日次ポジション明細を記録"""
    from sector_stoploss_compare import compute_atr
    all_dates_full = pd.date_range(start=start, end=end, freq='B')

    sector_signals = {}
    sector_ma_fast = {}
    sector_ma_slow = {}
    for sname, info in sector_results.items():
        idx = info['index'].loc[start:end]
        if len(idx) < slow + 5:
            continue
        maf = idx.rolling(fast, min_periods=fast).mean()
        mas = idx.rolling(slow, min_periods=slow).mean()
        sig = pd.Series(0, index=idx.index)
        sig[(maf > mas) & (maf.shift(1) <= mas.shift(1))] =  1
        sig[(maf < mas) & (maf.shift(1) >= mas.shift(1))] = -1
        sector_signals[sname] = sig
        sector_ma_fast[sname] = maf
        sector_ma_slow[sname] = mas

    active_sectors = list(sector_signals.keys())
    ref_idx = sector_signals[active_sectors[0]]
    dates = [d for d in all_dates_full if d in ref_idx.index]

    def ma_strength(sname, dt):
        maf = sector_ma_fast[sname]
        mas = sector_ma_slow[sname]
        if dt not in maf.index or pd.isna(maf.get(dt)) or pd.isna(mas.get(dt)) or mas.get(dt) == 0:
            return 0.0
        return float((maf.loc[dt] - mas.loc[dt]) / mas.loc[dt])

    slot_capital  = initial_capital / MAX_POSITIONS
    cash          = initial_capital
    positions     = []
    pending_exits = []
    pending_entries = []
    trades        = []

    # 日次記録
    daily_cash      = []
    daily_pos_vals  = []   # list of list: [[val_pos0, val_pos1, ...]] per day
    daily_pos_info  = []   # list of list: [{'sector','ticker','alloc'}] per day
    equity_dates    = []

    def current_price(ticker, sector, dt):
        ps = sector_results[sector]['prices'].get(ticker)
        if ps is None:
            return None
        avail = ps.index[ps.index >= dt]
        return float(ps.loc[avail[0]]) if len(avail) > 0 else None

    for i, dt in enumerate(dates):
        is_last = (i == len(dates) - 1)

        # 決済
        still_open = []
        for pe in pending_exits:
            pos, reason = pe['pos'], pe['reason']
            cp = current_price(pos['ticker'], pos['sector'], dt)
            if cp is not None:
                ret = (cp - pos['entry_price']) / pos['entry_price']
                cash += pos['alloc'] * (1 + ret)
                trades.append({
                    'sector': pos['sector'], 'ticker': pos['ticker'],
                    'entry_date': pos['entry_date'], 'exit_date': dt,
                    'entry_price': pos['entry_price'], 'exit_price': cp,
                    'return_pct': ret * 100, 'exit_reason': reason,
                })
            else:
                still_open.append(pe)
        exiting = {pe['pos']['ticker'] for pe in pending_exits if pe not in still_open}
        positions = [p for p in positions if p['ticker'] not in exiting]
        pending_exits = still_open

        # エントリー
        for pe in pending_entries[:]:
            if MAX_POSITIONS - len(positions) <= 0:
                break
            sname, ticker = pe['sector'], pe['ticker']
            if any(p['sector'] == sname for p in positions):
                pending_entries.remove(pe)
                continue
            cp = current_price(ticker, sname, dt)
            if cp and cp > 0 and cash >= slot_capital:
                stop_price = cp * (1 - stop_param) if stop_mode == 'fixed' else None
                cash -= slot_capital
                positions.append({
                    'sector': sname, 'ticker': ticker,
                    'entry_price': cp, 'entry_date': dt,
                    'alloc': slot_capital, 'peak_price': cp,
                    'stop_price': stop_price,
                })
                pending_entries.remove(pe)

        # シグナル確認 & 損切りチェック
        new_buy_signals = []
        for sname in active_sectors:
            if dt not in sector_signals[sname].index:
                continue
            sig = sector_signals[sname].loc[dt]
            for pos in [p for p in positions if p['sector'] == sname]:
                cp = current_price(pos['ticker'], sname, dt)
                if cp is None:
                    continue
                triggered = False
                if stop_mode != 'none' and pos['stop_price'] is not None:
                    if cp <= pos['stop_price']:
                        if pos not in [pe['pos'] for pe in pending_exits]:
                            pending_exits.append({'pos': pos, 'reason': 'stop_loss'})
                        triggered = True
                if not triggered and (sig == -1 or is_last):
                    if pos not in [pe['pos'] for pe in pending_exits]:
                        pending_exits.append({'pos': pos, 'reason': 'ma_signal' if sig == -1 else 'period_end'})

            if sig == 1:
                if not any(p['sector'] == sname for p in positions) and \
                   not any(pe['sector'] == sname for pe in pending_entries):
                    largest = sector_results[sname]['largest_cap']
                    tn = largest.loc[dt] if dt in largest.index else None
                    if tn and tn in sector_results[sname]['prices']:
                        new_buy_signals.append((ma_strength(sname, dt), sname, tn))

        free = MAX_POSITIONS - len(positions) - len(pending_entries)
        if free > 0 and new_buy_signals:
            new_buy_signals.sort(key=lambda x: x[0], reverse=True)
            for _, sname, tn in new_buy_signals[:free]:
                pending_entries.append({'sector': sname, 'ticker': tn})

        # 日次ポジション評価
        pos_vals = []
        pos_info = []
        for pos in positions:
            cp = current_price(pos['ticker'], pos['sector'], dt)
            val = pos['alloc'] * (cp / pos['entry_price']) if cp else pos['alloc']
            pos_vals.append(val)
            pos_info.append({'sector': pos['sector'], 'ticker': pos['ticker'],
                              'alloc': pos['alloc'], 'val': val})

        daily_cash.append(cash)
        daily_pos_vals.append(pos_vals)
        daily_pos_info.append(pos_info)
        equity_dates.append(dt)

    return {
        'dates': equity_dates,
        'daily_cash': daily_cash,
        'daily_pos_vals': daily_pos_vals,
        'daily_pos_info': daily_pos_info,
        'trades': trades,
        'initial_capital': initial_capital,
    }


def plot_position_chart(data, fast, slow, out_path):
    dates   = data['dates']
    cash    = np.array(data['daily_cash'])
    pos_vals = data['daily_pos_vals']
    trades  = data['trades']
    initial = data['initial_capital']

    # スロット数分の配列を作成 (最大 MAX_POSITIONS)
    slot_vals = np.zeros((MAX_POSITIONS, len(dates)))
    for t_i, pv in enumerate(pos_vals):
        for s_i, v in enumerate(pv[:MAX_POSITIONS]):
            slot_vals[s_i, t_i] = v

    total_equity = cash + slot_vals.sum(axis=0)

    slot_colors = ['#2980b9', '#e67e22', '#27ae60', '#8e44ad']
    slot_labels = [f'ポジション {i+1}' for i in range(MAX_POSITIONS)]

    fig, axes = plt.subplots(3, 1, figsize=(15, 14),
                              gridspec_kw={'height_ratios': [3, 2, 1.5]})
    fig.suptitle(
        f'ポジション金額推移 (テスト期間: 2022-2025)\n'
        f'MA({fast}/{slow}) / 固定損切り-10% / 同時最大{MAX_POSITIONS}ポジション',
        fontsize=12
    )

    # ── 上段: スタック面グラフ ────────────────────────
    ax1 = axes[0]
    bottom = np.zeros(len(dates))
    for s_i in range(MAX_POSITIONS):
        ax1.fill_between(dates, bottom, bottom + slot_vals[s_i],
                         label=slot_labels[s_i], color=slot_colors[s_i], alpha=0.75)
        bottom += slot_vals[s_i]
    # 現金部分
    ax1.fill_between(dates, bottom, bottom + cash,
                     label='現金', color='#bdc3c7', alpha=0.6)
    # 総資産ライン
    ax1.plot(dates, total_equity, color='black', lw=1.5, label='総資産')
    ax1.axhline(initial, color='gray', linestyle='--', lw=0.8, alpha=0.6)

    ax1.set_ylabel('金額 (円)', fontsize=10)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'¥{x/1e4:.0f}万'))
    ax1.legend(loc='upper left', fontsize=8, ncol=3)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('資産内訳 (現金 + ポジション別時価)', fontsize=10)

    # ── 中段: ポジション数と総投資額 ───────────────────
    ax2 = axes[1]
    n_pos = np.array([len(pv) for pv in pos_vals])
    invested = slot_vals.sum(axis=0)
    ax2.bar(dates, n_pos, color='steelblue', alpha=0.6, width=1, label='保有ポジション数')
    ax2.set_ylabel('ポジション数', fontsize=10, color='steelblue')
    ax2.set_ylim(0, MAX_POSITIONS + 1)
    ax2.set_yticks(range(MAX_POSITIONS + 1))
    ax2.legend(loc='upper left', fontsize=8)
    ax2_r = ax2.twinx()
    ax2_r.plot(dates, invested / 1e4, color='darkorange', lw=1.2, label='投資額 (万円)')
    ax2_r.set_ylabel('投資額 (万円)', fontsize=10, color='darkorange')
    ax2_r.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('保有ポジション数 & 投資金額', fontsize=10)

    # ── 下段: 取引ログ (エントリー/エグジット) ──────────
    ax3 = axes[2]
    df_trades = pd.DataFrame(trades)
    if len(df_trades):
        df_trades['entry_date'] = pd.to_datetime(df_trades['entry_date'])
        df_trades['exit_date']  = pd.to_datetime(df_trades['exit_date'])
        # エントリー
        ax3.scatter(df_trades['entry_date'], [1]*len(df_trades),
                    marker='^', color='green', s=60, label='エントリー', zorder=5)
        # 損切りエグジット
        sl = df_trades[df_trades['exit_reason'] == 'stop_loss']
        ax3.scatter(sl['exit_date'], [0]*len(sl),
                    marker='v', color='red', s=60, label='損切り', zorder=5)
        # MAシグナルエグジット
        ma = df_trades[df_trades['exit_reason'] != 'stop_loss']
        ax3.scatter(ma['exit_date'], [0]*len(ma),
                    marker='v', color='gray', s=60, label='MAシグナル', zorder=5)

    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['決済', 'エントリー'], fontsize=8)
    ax3.set_ylim(-0.5, 1.5)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.2)
    ax3.set_title(f'取引タイムライン (合計{len(trades)}件)', fontsize=10)

    for ax in axes:
        ax.set_xlim(pd.Timestamp(TEST_START), pd.Timestamp('2025-12-31'))

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"チャート保存: {out_path}")

    # ── 月次サマリー ──────────────────────────────────
    equity_s = pd.Series(total_equity, index=dates)
    monthly  = equity_s.resample('ME').last()
    monthly_ret = monthly.pct_change() * 100
    print("\n月次リターン (固定-10%):")
    print(monthly_ret.dropna().to_string())


def main():
    print("=" * 60)
    print("ポジション金額推移 可視化")
    print("=" * 60)

    sector_info = build_sector_data()
    all_dates = pd.date_range(start=TRAIN_START, end=TEST_END, freq='B')

    print("\n全業種インデックス構築中...")
    sector_built = {}
    for sector, info in sorted(sector_info.items(), key=lambda x: x[1]['code']):
        if len(info['prices']) < 2:
            continue
        from sector_index_backtest import build_sector_index, get_largest_cap_series
        idx = build_sector_index(info, all_dates)
        if len(idx) < 500:
            continue
        lc = get_largest_cap_series(info, all_dates)
        sector_built[sector] = {
            'code': info['code'], 'index': idx,
            'largest_cap': lc, 'prices': info['prices'],
            'shares': info['shares'], 'n_tickers': len(info['prices']),
        }
    print(f"有効業種数: {len(sector_built)}")

    print("\nMAパラメータ最適化中...")
    fast, slow = find_global_best_ma(sector_built)

    print(f"\nポジション追跡バックテスト実行中 (固定-10%, MA({fast}/{slow}))...")
    data = run_with_position_tracking(
        sector_built, fast, slow, TEST_START, TEST_END,
        stop_mode='fixed', stop_param=0.10,
    )

    out_path = OUT_DIR / 'portfolio_position_breakdown.png'
    plot_position_chart(data, fast, slow, out_path)

    # 取引ログ保存
    df = pd.DataFrame(data['trades'])
    df.to_csv(OUT_DIR / 'portfolio_trades_fixed10.csv', index=False, encoding='utf-8-sig')
    print(f"取引ログ: portfolio_trades_fixed10.csv ({len(df)}件)")

    # 統計
    total = [c + sum(pv) for c, pv in zip(data['daily_cash'], data['daily_pos_vals'])]
    equity = pd.Series(total, index=data['dates'])
    total_ret = (equity.iloc[-1] / data['initial_capital'] - 1) * 100
    max_dd = ((equity - equity.cummax()) / equity.cummax()).min() * 100
    print(f"\n最終リターン: {total_ret:+.2f}%  最大DD: {max_dd:.1f}%")


if __name__ == '__main__':
    main()

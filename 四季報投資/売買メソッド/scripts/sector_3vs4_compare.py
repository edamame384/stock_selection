"""
sector_3vs4_compare.py
業種コード3桁 vs 4桁 ポートフォリオバックテスト比較

戦略:
  買い : MA(25/100) ゴールデンクロス → 翌営業日終値でエントリー
  売り : 買値 -10% を下回った翌営業日終値でのみ決済 (MAシグナルは使わない)
  期末 : テスト期間終了時に強制決済

比較:
  3桁: 東証33業種分類 (33グループ)
  4桁: 主業種+副業種分類 (~165グループ)
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

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

warnings.filterwarnings('ignore')

# ── 3桁スクリプトから共通部品をインポート ────────────────────
from sector_index_backtest import (
    TICKER_BLACKLIST, PRICE_ANOMALY_RATIO,
    SECTOR_CODE_MAP, TRAIN_START, TRAIN_END, TEST_START, TEST_END,
    BASE_DIR, PRICE_DIR, SECTOR_CSV,
    load_price, extract_shares_outstanding,
    build_sector_data, build_sector_index, get_largest_cap_series,
    find_global_best_ma, MAX_POSITIONS,
)
# 4桁スクリプトから4桁マッピングだけインポート
from sector_index_backtest_4digit import (
    build_4digit_sector_mapping,
    build_sector_index    as build_sector_index_4d,
    get_largest_cap_series as get_largest_cap_4d,
)

OUT_DIR = BASE_DIR / '売買メソッド/results/sector_3vs4_compare'
OUT_DIR.mkdir(parents=True, exist_ok=True)

FAST = 25
SLOW = 100
STOP_PCT = 0.10     # 売り: 買値から-10%のみ


# ── ポートフォリオバックテスト (MAシグナル売り なし) ─────────
def run_portfolio_ma_buy_stop_sell(
    sector_results: dict,
    start: str,
    end: str,
    initial_capital: float = 1_000_000.0,
    label: str = '',
) -> dict:
    """
    買い: MA(FAST/SLOW) ゴールデンクロス翌日
    売り: 買値 × (1 - STOP_PCT) を下回った翌日のみ
         (MAデッドクロスでは売らない)
    """
    all_dates_full = pd.date_range(start=start, end=end, freq='B')

    sector_signals_buy  = {}   # 買いシグナルのみ
    sector_ma_fast_map  = {}
    sector_ma_slow_map  = {}

    for sname, info in sector_results.items():
        idx = info['index'].loc[start:end]
        if len(idx) < SLOW + 5:
            continue
        maf = idx.rolling(FAST, min_periods=FAST).mean()
        mas = idx.rolling(SLOW, min_periods=SLOW).mean()
        # 買いシグナルのみ (ゴールデンクロス)
        sig_buy = pd.Series(0, index=idx.index)
        sig_buy[(maf > mas) & (maf.shift(1) <= mas.shift(1))] = 1
        sector_signals_buy[sname]  = sig_buy
        sector_ma_fast_map[sname]  = maf
        sector_ma_slow_map[sname]  = mas

    active_sectors = list(sector_signals_buy.keys())
    if not active_sectors:
        return None
    ref_idx = sector_signals_buy[active_sectors[0]]
    dates   = [d for d in all_dates_full if d in ref_idx.index]
    if not dates:
        return None

    def ma_gap(sname, dt):
        maf = sector_ma_fast_map[sname]
        mas = sector_ma_slow_map[sname]
        if dt not in maf.index or pd.isna(maf.get(dt)) or pd.isna(mas.get(dt)) or mas.get(dt) == 0:
            return 0.0
        return float((maf.loc[dt] - mas.loc[dt]) / mas.loc[dt])

    slot_capital    = initial_capital / MAX_POSITIONS
    cash            = initial_capital
    positions       = []
    pending_exits   = []
    pending_entries = []
    trades          = []
    equity_vals     = []
    equity_dates    = []

    def current_price(ticker, sector, dt):
        ps = sector_results[sector]['prices'].get(ticker)
        if ps is None:
            return None
        avail = ps.index[ps.index >= dt]
        return float(ps.loc[avail[0]]) if len(avail) > 0 else None

    for i, dt in enumerate(dates):
        is_last = (i == len(dates) - 1)

        # ── 決済執行 ──────────────────────────────────
        still_open = []
        for pe in pending_exits:
            pos, reason = pe['pos'], pe['reason']
            cp = current_price(pos['ticker'], pos['sector'], dt)
            if cp is not None:
                ret = (cp - pos['entry_price']) / pos['entry_price']
                cash += pos['alloc'] * (1 + ret)
                trades.append({
                    'sector':      pos['sector'],
                    'ticker':      pos['ticker'],
                    'entry_date':  pos['entry_date'],
                    'exit_date':   dt,
                    'entry_price': pos['entry_price'],
                    'exit_price':  cp,
                    'return_pct':  ret * 100,
                    'exit_reason': reason,
                })
            else:
                still_open.append(pe)
        exiting = {pe['pos']['ticker'] for pe in pending_exits if pe not in still_open}
        positions = [p for p in positions if p['ticker'] not in exiting]
        pending_exits = still_open

        # ── エントリー執行 ────────────────────────────
        for pe in pending_entries[:]:
            if MAX_POSITIONS - len(positions) <= 0:
                break
            sname, ticker = pe['sector'], pe['ticker']
            if any(p['sector'] == sname for p in positions):
                pending_entries.remove(pe)
                continue
            cp = current_price(ticker, sname, dt)
            if cp and cp > 0 and cash >= slot_capital:
                cash -= slot_capital
                positions.append({
                    'sector':      sname,
                    'ticker':      ticker,
                    'entry_price': cp,
                    'entry_date':  dt,
                    'alloc':       slot_capital,
                    'stop_price':  cp * (1 - STOP_PCT),
                })
                pending_entries.remove(pe)

        # ── 損切りチェック & 買いシグナル確認 ──────────
        new_buys = []
        for sname in active_sectors:
            if dt not in sector_signals_buy[sname].index:
                continue
            sig = sector_signals_buy[sname].loc[dt]

            for pos in [p for p in positions if p['sector'] == sname]:
                cp = current_price(pos['ticker'], sname, dt)
                if cp is None:
                    continue
                # ストップロスのみで決済 (MAデッドクロスは無視)
                if cp <= pos['stop_price']:
                    if pos not in [pe['pos'] for pe in pending_exits]:
                        pending_exits.append({'pos': pos, 'reason': 'stop_loss'})
                elif is_last:
                    # 期末: pending_exits 経由では翌日がないので即時決済
                    ret = (cp - pos['entry_price']) / pos['entry_price']
                    cash += pos['alloc'] * (1 + ret)
                    trades.append({
                        'sector':      pos['sector'],
                        'ticker':      pos['ticker'],
                        'entry_date':  pos['entry_date'],
                        'exit_date':   dt,
                        'entry_price': pos['entry_price'],
                        'exit_price':  cp,
                        'return_pct':  ret * 100,
                        'exit_reason': 'period_end',
                    })
                    positions = [p for p in positions if p is not pos]

            # 買いシグナル
            if sig == 1:
                if not any(p['sector'] == sname for p in positions) and \
                   not any(pe['sector'] == sname for pe in pending_entries):
                    largest = sector_results[sname]['largest_cap']
                    tn = largest.loc[dt] if dt in largest.index else None
                    if tn and tn in sector_results[sname]['prices']:
                        new_buys.append((ma_gap(sname, dt), sname, tn))

        free = MAX_POSITIONS - len(positions) - len(pending_entries)
        if free > 0 and new_buys:
            new_buys.sort(key=lambda x: x[0], reverse=True)
            for _, sname, tn in new_buys[:free]:
                pending_entries.append({'sector': sname, 'ticker': tn})

        # ── エクイティ評価 ────────────────────────────
        open_val = sum(
            pos['alloc'] * ((current_price(pos['ticker'], pos['sector'], dt) or pos['entry_price'])
                            / pos['entry_price'])
            for pos in positions
        )
        equity_vals.append(cash + open_val)
        equity_dates.append(dt)

    equity_curve = pd.Series(equity_vals, index=equity_dates)
    total_ret = (equity_curve.iloc[-1] / initial_capital - 1) * 100
    dr        = equity_curve.pct_change().dropna()
    sharpe    = (dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0.0
    max_dd    = ((equity_curve - equity_curve.cummax()) / equity_curve.cummax()).min() * 100
    win_rate  = sum(1 for t in trades if t['return_pct'] > 0) / len(trades) * 100 if trades else 0.0
    stop_cnt  = sum(1 for t in trades if t['exit_reason'] == 'stop_loss')

    if label:
        print(f"  [{label}] リターン:{total_ret:+.2f}% | Sharpe:{sharpe:.3f} | "
              f"最大DD:{max_dd:.1f}% | 勝率:{win_rate:.1f}% | "
              f"取引:{len(trades)} (損切:{stop_cnt})")

    return dict(
        total_return_pct=total_ret, sharpe=sharpe,
        max_drawdown_pct=max_dd, win_rate_pct=win_rate,
        num_trades=len(trades), stop_loss_count=stop_cnt,
        equity_curve=equity_curve, trades=trades,
    )


# ── 4桁セクターデータ構築 ─────────────────────────────────────
def build_4digit_sector_results(all_prices, all_shares, all_dates):
    """
    4桁グループごとに index / largest_cap / prices を構築して返す
    ※ load_price は 3桁スクリプトの版 (TICKER_BLACKLIST 込み) を使用済み
    """
    mapping = build_4digit_sector_mapping()
    groups  = mapping.groupby(['code4', 'group_name'])
    results = {}

    print(f"4桁グループ構築中 ({groups.ngroups} グループ)...")
    for (code4, group_name), grp_df in sorted(groups, key=lambda x: x[0][0]):
        tickers = [t for t in grp_df['ticker'].tolist() if t in all_prices]
        if len(tickers) < 2:
            continue
        idx = build_sector_index_4d(tickers, all_prices, all_shares, all_dates)
        if len(idx) < 500:
            continue
        lc = get_largest_cap_4d(tickers, all_prices, all_shares, all_dates)
        results[group_name] = {
            'code':        code4,
            'index':       idx,
            'largest_cap': lc,
            'prices':      {t: all_prices[t] for t in tickers},
            'n_tickers':   len(tickers),
        }

    print(f"有効4桁グループ数: {len(results)}")
    return results


# ── 比較チャート ────────────────────────────────────────────
def plot_comparison(res3, res4, out_path):
    import matplotlib.dates as mdates

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        f'業種コード 3桁 vs 4桁 ポートフォリオ比較\n'
        f'戦略: 買いMA({FAST}/{SLOW}) / 売り固定-{STOP_PCT*100:.0f}% / 同時最大{MAX_POSITIONS}ポジション',
        fontsize=15
    )

    # ── エクイティカーブ比較 (テスト) ──
    ax = axes[0][0]
    for res, label, color in [(res3['test'], '3桁', '#2980b9'),
                               (res4['test'], '4桁', '#e67e22')]:
        if res:
            ec = res['equity_curve']
            ax.plot(ec.index, ec.values, label=label, color=color, lw=2)
    ax.axhline(1_000_000, color='gray', linestyle='--', lw=0.8)
    ax.set_title('エクイティカーブ (テスト期間: 2022-2025)', fontsize=13)
    ax.set_ylabel('資産', fontsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e4:.0f}万円'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # ── エクイティカーブ比較 (学習) ──
    ax2 = axes[0][1]
    for res, label, color in [(res3['train'], '3桁', '#2980b9'),
                               (res4['train'], '4桁', '#e67e22')]:
        if res:
            ec = res['equity_curve']
            ax2.plot(ec.index, ec.values, label=label, color=color, lw=2)
    ax2.axhline(1_000_000, color='gray', linestyle='--', lw=0.8)
    ax2.set_title('エクイティカーブ (学習期間: 2001-2021)', fontsize=13)
    ax2.set_ylabel('資産', fontsize=12)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e4:.0f}万円'))
    ax2.xaxis.set_major_locator(mdates.YearLocator(5))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.tick_params(labelsize=11)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)

    # ── 指標バー比較 ──
    metrics = ['リターン(%)', 'Sharpe', '最大DD(%)', '勝率(%)']
    vals3_test = [
        res3['test']['total_return_pct'], res3['test']['sharpe'],
        res3['test']['max_drawdown_pct'], res3['test']['win_rate_pct'],
    ] if res3['test'] else [0]*4
    vals4_test = [
        res4['test']['total_return_pct'], res4['test']['sharpe'],
        res4['test']['max_drawdown_pct'], res4['test']['win_rate_pct'],
    ] if res4['test'] else [0]*4

    ax3 = axes[1][0]
    x = np.arange(len(metrics))
    w = 0.35
    ax3.bar(x - w/2, vals3_test, w, label='3桁', color='#2980b9', alpha=0.8)
    ax3.bar(x + w/2, vals4_test, w, label='4桁', color='#e67e22', alpha=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics, fontsize=12)
    ax3.tick_params(labelsize=11)
    ax3.set_title('指標比較 (テスト期間)', fontsize=13)
    ax3.axhline(0, color='gray', linestyle='--', lw=0.8)
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)

    # ── サマリーテキスト ──
    ax4 = axes[1][1]
    ax4.axis('off')

    def fmt(res, period):
        if not res:
            return f'{period}: データなし'
        return (f'{period}:\n'
                f'  リターン  : {res["total_return_pct"]:+.2f}%\n'
                f'  Sharpe    : {res["sharpe"]:.3f}\n'
                f'  最大DD    : {res["max_drawdown_pct"]:.1f}%\n'
                f'  勝率      : {res["win_rate_pct"]:.1f}%\n'
                f'  取引数    : {res["num_trades"]} (損切:{res["stop_loss_count"]})')

    text = (
        '【3桁分類】\n' +
        fmt(res3['train'], '学習') + '\n\n' +
        fmt(res3['test'],  'テスト') + '\n\n\n' +
        '【4桁分類】\n' +
        fmt(res4['train'], '学習') + '\n\n' +
        fmt(res4['test'],  'テスト')
    )
    ax4.text(0.05, 0.95, text, transform=ax4.transAxes,
             fontsize=12, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.8))

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"比較チャート保存: {out_path}")


# ── メイン ──────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("3桁 vs 4桁 ポートフォリオ比較")
    print(f"戦略: 買い MA({FAST}/{SLOW}) / 売り 固定-{STOP_PCT*100:.0f}% のみ")
    print("=" * 65)

    # ── 共通データ読み込み ──
    sector_info = build_sector_data()
    all_dates   = pd.date_range(start=TRAIN_START, end=TEST_END, freq='B')

    # ── 3桁セクター構築 ──
    print("\n【3桁】セクター構築中...")
    sector3 = {}
    for sector, info in sorted(sector_info.items(), key=lambda x: x[1]['code']):
        if len(info['prices']) < 2:
            continue
        idx = build_sector_index(info, all_dates)
        if len(idx) < 500:
            continue
        lc = get_largest_cap_series(info, all_dates)
        sector3[sector] = {
            'code': info['code'], 'index': idx,
            'largest_cap': lc, 'prices': info['prices'],
            'n_tickers': len(info['prices']),
        }
    print(f"有効3桁セクター数: {len(sector3)}")

    # 4桁用の共通価格辞書 (3桁の load_price でフィルタ済み)
    all_prices = {t: s for info in sector_info.values() for t, s in info['prices'].items()}
    all_shares = {t: s for info in sector_info.values() for t, s in info['shares'].items()}

    # ── 4桁セクター構築 ──
    print("\n【4桁】セクター構築中...")
    sector4 = build_4digit_sector_results(all_prices, all_shares, all_dates)

    # ── バックテスト ──
    print(f"\n== 学習期間バックテスト ({TRAIN_START} ~ {TRAIN_END}) ==")
    print("【3桁】")
    res3_train = run_portfolio_ma_buy_stop_sell(sector3, TRAIN_START, TRAIN_END, label='3桁学習')
    print("【4桁】")
    res4_train = run_portfolio_ma_buy_stop_sell(sector4, TRAIN_START, TRAIN_END, label='4桁学習')

    print(f"\n== テスト期間バックテスト ({TEST_START} ~ {TEST_END}) ==")
    print("【3桁】")
    res3_test = run_portfolio_ma_buy_stop_sell(sector3, TEST_START, TEST_END, label='3桁テスト')
    print("【4桁】")
    res4_test = run_portfolio_ma_buy_stop_sell(sector4, TEST_START, TEST_END, label='4桁テスト')

    # ── 結果比較表 ──
    print("\n" + "=" * 70)
    print(f"{'':12} {'リターン':>10} {'Sharpe':>8} {'最大DD':>8} {'勝率':>7} {'取引数':>7} {'損切':>6}")
    print("-" * 70)
    for label, res in [('3桁 学習', res3_train), ('3桁 テスト', res3_test),
                        ('4桁 学習', res4_train), ('4桁 テスト', res4_test)]:
        if res:
            print(f"{label:<12} {res['total_return_pct']:>+9.2f}% "
                  f"{res['sharpe']:>8.3f} "
                  f"{res['max_drawdown_pct']:>7.1f}% "
                  f"{res['win_rate_pct']:>6.1f}% "
                  f"{res['num_trades']:>7} "
                  f"{res['stop_loss_count']:>6}")
    print("=" * 70)

    # ── CSV 保存 ──
    rows = []
    for label, res in [('3桁_学習', res3_train), ('3桁_テスト', res3_test),
                        ('4桁_学習', res4_train), ('4桁_テスト', res4_test)]:
        if res:
            rows.append({
                '分類': label,
                'リターン(%)': round(res['total_return_pct'], 2),
                'Sharpe': round(res['sharpe'], 3),
                '最大DD(%)': round(res['max_drawdown_pct'], 2),
                '勝率(%)': round(res['win_rate_pct'], 1),
                '取引数': res['num_trades'],
                '損切り数': res['stop_loss_count'],
            })
    pd.DataFrame(rows).to_csv(
        OUT_DIR / '3vs4_comparison.csv', index=False, encoding='utf-8-sig'
    )

    # 取引ログ
    for label, res, fname in [
        ('3桁テスト', res3_test, '3digit_trades_test.csv'),
        ('4桁テスト', res4_test, '4digit_trades_test.csv'),
    ]:
        if res and res['trades']:
            pd.DataFrame(res['trades']).to_csv(
                OUT_DIR / fname, index=False, encoding='utf-8-sig'
            )

    # ── 比較チャート ──
    plot_comparison(
        {'train': res3_train, 'test': res3_test},
        {'train': res4_train, 'test': res4_test},
        OUT_DIR / '3vs4_comparison.png',
    )
    print(f"\n結果保存先: {OUT_DIR}")


if __name__ == '__main__':
    main()

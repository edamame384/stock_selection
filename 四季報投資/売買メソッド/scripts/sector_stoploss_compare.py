"""
sector_stoploss_compare.py
損切り方式 比較バックテスト (テスト期間 2022-2025)

比較対象 (MAX_POSITIONS=4, MA25/100):
  1. なし         : MAクロスのみで決済
  2. 固定 -5%     : 買値-5%を下回ったら翌日決済
  3. 固定 -10%    : 買値-10%を下回ったら翌日決済
  4. ATR×2       : 買値 - ATR(14)×2 を下回ったら翌日決済
  5. トレーリング-8%: 保有中の最高値から-8%を下回ったら翌日決済
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── メインモジュールから関数をインポート ──────────────────────────
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from sector_index_backtest import (
    TICKER_BLACKLIST, PRICE_ANOMALY_RATIO, SECTOR_CODE_MAP,
    TRAIN_START, TRAIN_END, TEST_START, TEST_END,
    BASE_DIR, PRICE_DIR, SECTOR_CSV,
    load_price, extract_shares_outstanding,
    build_sector_data, build_sector_index, get_largest_cap_series,
    generate_ma_signals, find_global_best_ma,
    MAX_POSITIONS,
)

warnings.filterwarnings('ignore')

OUT_DIR = BASE_DIR / '売買メソッド/results/sector_backtest'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── ATR 計算用: High/Low も含めた価格を読み込む ──────────────────
def load_hlc(ticker: str) -> pd.DataFrame | None:
    """High / Low / Close を返す (ATR 計算用)"""
    if ticker in TICKER_BLACKLIST:
        return None
    fname = PRICE_DIR / f'{ticker}_T.csv'
    if not fname.exists():
        return None
    try:
        df = pd.read_csv(fname, index_col='Date', parse_dates=True)
        df = df[['High', 'Low', 'Close']].dropna()
        if len(df) < 30:
            return None
        return df
    except Exception:
        return None


def compute_atr(ticker: str, period: int = 14) -> pd.Series | None:
    """ATR(period) を pd.Series で返す"""
    df = load_hlc(ticker)
    if df is None:
        return None
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift(1)).abs(),
        (df['Low']  - df['Close'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


# ── 汎用ポートフォリオバックテスト ───────────────────────────────
def run_portfolio_compare(
    sector_results: dict,
    fast: int,
    slow: int,
    start: str,
    end: str,
    stop_mode: str = 'none',    # 'none' | 'fixed' | 'atr' | 'trailing'
    stop_param: float = 0.0,    # fixed→下落率, atr→乗数, trailing→下落率
    initial_capital: float = 1_000_000.0,
) -> dict:
    """
    stop_mode:
      'none'     : 損切りなし (MAシグナルのみ)
      'fixed'    : stop_param = 0.05 → 買値の-5%
      'atr'      : stop_param = 2.0  → 買値 - ATR(14)×2
      'trailing' : stop_param = 0.08 → 最高値から-8%
    """
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
    if not active_sectors:
        return None
    ref_idx = sector_signals[active_sectors[0]]
    dates = [d for d in all_dates_full if d in ref_idx.index]
    if not dates:
        return None

    def ma_strength(sname, dt):
        maf = sector_ma_fast[sname]
        mas = sector_ma_slow[sname]
        if dt not in maf.index or pd.isna(maf.get(dt)) or pd.isna(mas.get(dt)) or mas.get(dt) == 0:
            return 0.0
        return float((maf.loc[dt] - mas.loc[dt]) / mas.loc[dt])

    # ATR は事前計算してキャッシュ
    atr_cache = {}
    if stop_mode == 'atr':
        print(f"    ATR計算中...", end='', flush=True)
        for sname, info in sector_results.items():
            for ticker in info['prices']:
                if ticker not in atr_cache:
                    atr = compute_atr(ticker)
                    if atr is not None:
                        atr_cache[ticker] = atr
        print(f" {len(atr_cache)} 銘柄完了")

    slot_capital  = initial_capital / MAX_POSITIONS
    cash          = initial_capital
    positions     = []   # {'sector','ticker','entry_price','entry_date','alloc','peak_price','stop_price'}
    pending_exits = []   # {'pos':..., 'reason':...}
    pending_entries = []
    trades        = []
    equity_vals   = []
    equity_dates  = []

    def current_price(ticker, sector, dt):
        ps = sector_results[sector]['prices'].get(ticker)
        if ps is None:
            return None
        avail = ps.index[ps.index >= dt]
        return float(ps.loc[avail[0]]) if len(avail) > 0 else None

    for i, dt in enumerate(dates):
        is_last = (i == len(dates) - 1)

        # ─── 決済執行 ─────────────────────────────
        still_open = []
        for pe in pending_exits:
            pos    = pe['pos']
            reason = pe['reason']
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

        # ─── エントリー執行 ───────────────────────
        for pe in pending_entries[:]:
            if MAX_POSITIONS - len(positions) <= 0:
                break
            sname, ticker = pe['sector'], pe['ticker']
            if any(p['sector'] == sname for p in positions):
                pending_entries.remove(pe)
                continue
            cp = current_price(ticker, sname, dt)
            if cp and cp > 0 and cash >= slot_capital:
                # ATR ストップ価格を計算
                stop_price = None
                if stop_mode == 'fixed':
                    stop_price = cp * (1 - stop_param)
                elif stop_mode == 'atr':
                    atr_s = atr_cache.get(ticker)
                    if atr_s is not None and dt in atr_s.index and not pd.isna(atr_s.loc[dt]):
                        stop_price = cp - atr_s.loc[dt] * stop_param
                    else:
                        stop_price = cp * (1 - 0.10)  # ATR 取得不可時は10%固定
                elif stop_mode == 'trailing':
                    stop_price = cp * (1 - stop_param)  # 初期は買値基準

                cash -= slot_capital
                positions.append({
                    'sector': sname, 'ticker': ticker,
                    'entry_price': cp, 'entry_date': dt,
                    'alloc': slot_capital,
                    'peak_price': cp,
                    'stop_price': stop_price,
                })
                pending_entries.remove(pe)

        # ─── シグナル確認 & 損切りチェック ────────
        new_buy_signals = []
        for sname in active_sectors:
            if dt not in sector_signals[sname].index:
                continue
            sig = sector_signals[sname].loc[dt]

            for pos in [p for p in positions if p['sector'] == sname]:
                cp = current_price(pos['ticker'], sname, dt)
                if cp is None:
                    continue

                # トレーリングストップ: ピーク更新
                if stop_mode == 'trailing' and cp > pos['peak_price']:
                    pos['peak_price'] = cp
                    pos['stop_price'] = cp * (1 - stop_param)

                # ATR ストップ: エントリー時に設定済み (動かさない)

                triggered = False
                if stop_mode != 'none' and pos['stop_price'] is not None:
                    if cp <= pos['stop_price']:
                        if pos not in [pe['pos'] for pe in pending_exits]:
                            pending_exits.append({'pos': pos, 'reason': 'stop_loss'})
                        triggered = True

                if not triggered and (sig == -1 or is_last):
                    if pos not in [pe['pos'] for pe in pending_exits]:
                        label = 'ma_signal' if sig == -1 else 'period_end'
                        pending_exits.append({'pos': pos, 'reason': label})

            if sig == 1:
                already_held    = any(p['sector'] == sname for p in positions)
                already_pending = any(pe['sector'] == sname for pe in pending_entries)
                if not already_held and not already_pending:
                    largest = sector_results[sname]['largest_cap']
                    ticker_now = largest.loc[dt] if dt in largest.index else None
                    if ticker_now and ticker_now in sector_results[sname]['prices']:
                        new_buy_signals.append((ma_strength(sname, dt), sname, ticker_now))

        free = MAX_POSITIONS - len(positions) - len(pending_entries)
        if free > 0 and new_buy_signals:
            new_buy_signals.sort(key=lambda x: x[0], reverse=True)
            for strength, sname, ticker in new_buy_signals[:free]:
                pending_entries.append({'sector': sname, 'ticker': ticker})

        # ─── エクイティ評価 ───────────────────────
        open_val = sum(
            pos['alloc'] * (current_price(pos['ticker'], pos['sector'], dt) or pos['entry_price'])
            / pos['entry_price']
            for pos in positions
        )
        equity_vals.append(cash + open_val)
        equity_dates.append(dt)

    equity_curve = pd.Series(equity_vals, index=equity_dates)
    total_ret = (equity_curve.iloc[-1] / initial_capital - 1) * 100
    dr = equity_curve.pct_change().dropna()
    sharpe = (dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0.0
    max_dd = ((equity_curve - equity_curve.cummax()) / equity_curve.cummax()).min() * 100
    win_rate = sum(1 for t in trades if t['return_pct'] > 0) / len(trades) * 100 if trades else 0.0
    stop_count = sum(1 for t in trades if t['exit_reason'] == 'stop_loss')

    return dict(
        total_return_pct=total_ret, sharpe=sharpe,
        max_drawdown_pct=max_dd, win_rate_pct=win_rate,
        num_trades=len(trades), stop_loss_count=stop_count,
        equity_curve=equity_curve, trades=trades,
    )


# ── メイン ───────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("損切り方式 比較バックテスト")
    print("=" * 60)

    # ── データ構築 ──
    sector_info = build_sector_data()
    all_dates = pd.date_range(start=TRAIN_START, end=TEST_END, freq='B')

    print("\n全業種インデックス構築中...")
    sector_built = {}
    for sector, info in sorted(sector_info.items(), key=lambda x: x[1]['code']):
        if len(info['prices']) < 2:
            continue
        sector_idx = build_sector_index(info, all_dates)
        if len(sector_idx) < 500:
            continue
        largest_cap = get_largest_cap_series(info, all_dates)
        sector_built[sector] = {
            'code': info['code'], 'index': sector_idx,
            'largest_cap': largest_cap, 'prices': info['prices'],
            'shares': info['shares'], 'n_tickers': len(info['prices']),
        }
    print(f"有効業種数: {len(sector_built)}")

    print("\n全業種統一MAパラメータ最適化中...")
    fast, slow = find_global_best_ma(sector_built)

    # ── 比較対象の設定 ──
    VARIANTS = [
        ('なし (MAのみ)',       'none',     0.00),
        ('固定 -5%',           'fixed',    0.05),
        ('固定 -10%',          'fixed',    0.10),
        ('ATR(14)×2',         'atr',      2.00),
        ('トレーリング -8%',    'trailing', 0.08),
    ]

    print(f"\nMA({fast}/{slow}) / MAX_POSITIONS={MAX_POSITIONS} でテスト期間比較")
    print("-" * 70)

    results = {}
    for label, mode, param in VARIANTS:
        print(f"\n  [{label}] (mode={mode}, param={param})")
        res = run_portfolio_compare(
            sector_built, fast, slow, TEST_START, TEST_END,
            stop_mode=mode, stop_param=param,
        )
        results[label] = res
        if res:
            print(f"    リターン: {res['total_return_pct']:+.2f}% | "
                  f"Sharpe: {res['sharpe']:.3f} | "
                  f"最大DD: {res['max_drawdown_pct']:.1f}% | "
                  f"勝率: {res['win_rate_pct']:.1f}% | "
                  f"取引数: {res['num_trades']} (損切り: {res['stop_loss_count']})")

    # ── 比較表 ──
    print("\n" + "=" * 75)
    print(f"{'方式':<20} {'リターン':>9} {'Sharpe':>8} {'最大DD':>8} {'勝率':>7} {'取引数':>7} {'損切':>6}")
    print("-" * 75)
    for label, _, _ in VARIANTS:
        r = results.get(label)
        if r:
            print(f"{label:<20} {r['total_return_pct']:>+8.2f}% "
                  f"{r['sharpe']:>8.3f} "
                  f"{r['max_drawdown_pct']:>7.1f}% "
                  f"{r['win_rate_pct']:>6.1f}% "
                  f"{r['num_trades']:>7} "
                  f"{r['stop_loss_count']:>6}")
    print("=" * 75)

    # ── 比較チャート ──
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    colors = ['#333333', '#e74c3c', '#e67e22', '#2980b9', '#27ae60']
    ax = axes[0]
    for (label, _, _), color in zip(VARIANTS, colors):
        r = results.get(label)
        if r and r['equity_curve'] is not None:
            ec = r['equity_curve']
            ax.plot(ec.index, ec.values, label=label, color=color, lw=1.5)
    ax.axhline(1_000_000, color='gray', linestyle='--', lw=0.8)
    ax.set_title(f'損切り方式比較 エクイティカーブ (テスト期間: 2022-2025) MA({fast}/{slow})',
                 fontsize=12)
    ax.set_ylabel('資産 (円)', fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'¥{x:,.0f}'))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # バー比較チャート
    ax2 = axes[1]
    labels_short = [l for l, _, _ in VARIANTS]
    returns = [results[l]['total_return_pct'] if results[l] else 0 for l in labels_short]
    sharpes = [results[l]['sharpe'] if results[l] else 0 for l in labels_short]
    x = np.arange(len(labels_short))
    w = 0.35
    bars1 = ax2.bar(x - w/2, returns, w, label='リターン (%)', color=colors, alpha=0.8)
    ax2_r = ax2.twinx()
    ax2_r.plot(x, sharpes, 'o--', color='purple', lw=1.5, label='Sharpe')
    ax2_r.set_ylabel('Sharpe ratio', fontsize=9, color='purple')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_short, fontsize=9)
    ax2.set_ylabel('リターン (%)', fontsize=10)
    ax2.set_title('リターン & Sharpe 比較', fontsize=11)
    ax2.axhline(0, color='gray', linestyle='--', lw=0.8)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=8)
    ax2_r.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    out_path = OUT_DIR / 'stoploss_comparison.png'
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"\n比較チャート保存: {out_path}")

    # ── CSV 保存 ──
    rows = []
    for label, mode, param in VARIANTS:
        r = results.get(label)
        if r:
            rows.append({
                '方式': label, 'stop_mode': mode, 'stop_param': param,
                'リターン(%)': round(r['total_return_pct'], 2),
                'Sharpe': round(r['sharpe'], 3),
                '最大DD(%)': round(r['max_drawdown_pct'], 2),
                '勝率(%)': round(r['win_rate_pct'], 1),
                '取引数': r['num_trades'],
                '損切り数': r['stop_loss_count'],
            })
    pd.DataFrame(rows).to_csv(
        OUT_DIR / 'stoploss_comparison.csv', index=False, encoding='utf-8-sig'
    )
    print(f"比較CSV保存: {OUT_DIR / 'stoploss_comparison.csv'}")


if __name__ == '__main__':
    main()

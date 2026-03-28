"""
sector_index_backtest_4digit.py

四季報業種コード4桁（主業種+副業種）分類による
時価総額加重平均チャート＋バックテスト

- 主業種コード上3桁（東証33業種）に副業種で4桁目を付与
- 銘柄数が少ないサブ業種は主業種（3桁）にフォールバック
- シグナル翌営業日の終値で売買執行
- 学習: 2001-2021 / テスト: 2022-2025
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import product
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================
# パス設定
# ============================================================
BASE_DIR  = Path('/home/user/stock_selection/四季報投資')
PRICE_DIR = BASE_DIR / 'その他/shikiho_text_parser_runtime/data/prices_full'
RAW_DIR   = BASE_DIR / 'その他/shikiho_text_parser_runtime/data/raw/4Q-2'
SECTOR_CSV = BASE_DIR / 'その他/shikiho_text_parser_runtime/data/reference/sector_master_template.csv'
OUT_DIR   = BASE_DIR / '売買メソッド/results/sector_backtest_4digit'
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_START = '2001-01-01'
TRAIN_END   = '2021-12-31'
TEST_START  = '2022-01-01'
TEST_END    = '2025-12-31'

MIN_STOCKS_PER_GROUP = 3   # これ未満のサブ業種は主業種にマージ

# 東証33業種 → コード上3桁 マッピング
SECTOR_33_TO_CODE3 = {
    '水産・農林業': '005', '鉱業': '105', '建設業': '205', '食料品': '210',
    '繊維製品': '215', 'パルプ・紙': '220', '化学': '225', '医薬品': '230',
    '石油・石炭製品': '235', 'ゴム製品': '240', 'ガラス・土石製品': '245',
    '鉄鋼': '250', '非鉄金属': '255', '金属製品': '260', '機械': '305',
    '電気機器': '310', '輸送用機器': '315', '精密機器': '320', 'その他製品': '325',
    '電気・ガス業': '330', '陸運業': '335', '海運業': '340', '空運業': '345',
    '倉庫・運輸関連業': '350', '情報・通信業': '355', '卸売業': '405', '小売業': '410',
    '銀行業': '505', '証券、商品先物取引業': '510', '保険業': '515',
    'その他金融業': '520', '不動産業': '605', 'サービス業': '705',
}

# 主業種(rawテキスト表記) → 東証33業種名 マッピング
RAW_MAIN_TO_SECTOR33 = {
    '水産農林': '水産・農林業', '鉱業': '鉱業', '建設': '建設業',
    '食料品': '食料品', '繊維製品': '繊維製品', 'パルプ紙': 'パルプ・紙',
    '化学': '化学', '医薬品': '医薬品', '石油石炭': '石油・石炭製品',
    'ゴム製品': 'ゴム製品', 'ガラス土石': 'ガラス・土石製品', '鉄鋼': '鉄鋼',
    '非鉄金属': '非鉄金属', '金属製品': '金属製品', '機械': '機械',
    '電気機器': '電気機器', '輸送用機器': '輸送用機器', '精密機器': '精密機器',
    'その他製品': 'その他製品', '電力ガス': '電気・ガス業', '陸運': '陸運業',
    '海運': '海運業', '空運': '空運業', '倉庫運輸': '倉庫・運輸関連業',
    '情報通信': '情報・通信業', '卸売': '卸売業', '小売': '小売業',
    '銀行': '銀行業', '証券先物': '証券、商品先物取引業', '保険': '保険業',
    'その他金融': 'その他金融業', '不動産': '不動産業', 'サービス': 'サービス業',
}


# ============================================================
# 1. rawテキストからサブ業種を抽出
# ============================================================
def extract_sectors_from_text(ticker: str) -> tuple[str | None, str | None]:
    txt_path = RAW_DIR / f'{ticker}.txt'
    if not txt_path.exists():
        return None, None
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
    except Exception:
        return None, None

    code_line_idx = -1
    for i, l in enumerate(lines):
        if re.match(r'^\s*\d{3,4}[A-Z0-9]?\s+(東証|名証|福証|札証|東証グロース|東証スタンダード|東証プライム|JASDAQ)', l.strip()):
            code_line_idx = i
            break
    if code_line_idx < 0:
        return None, None

    skip_patterns = [
        r'直近決算', r'^\d{4}', r'https?://', r'会社HP', r'チャート', r'特$', r'色$',
        r'^(業績予想更新あり|優待あり|貸借|信用|前号並み|上方修正|下方修正|注目|増額|減額)$',
        r'^[\d,\.]+$',
    ]

    results = []
    company_found = False
    for l in lines[code_line_idx + 1:code_line_idx + 60]:
        l = l.strip()
        if not l:
            continue
        if any(re.search(p, l) for p in skip_patterns):
            continue
        if not company_found and re.match(r'^[ぁ-ん゛゜ァ-ヶー一-龠Ａ-Ｚ\w・（）()&\-]+$', l) and len(l) <= 25:
            company_found = True
            continue
        if re.match(r'^[ぁ-ん゛゜ァ-ヶー一-龠\w・（）()]+$', l) and len(l) <= 15:
            results.append(l)
        if len(results) >= 2:
            break

    return (results[0] if results else None), (results[1] if len(results) > 1 else None)


# ============================================================
# 2. 発行済株式数抽出
# ============================================================
def extract_shares_outstanding(ticker: str) -> float | None:
    txt_path = RAW_DIR / f'{ticker}.txt'
    if not txt_path.exists():
        return None
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception:
        return None

    stock_section_match = re.search(r'株式\s*\n(.*?)(?:\n\n|\Z)', text, re.DOTALL)
    search_text = stock_section_match.group(0) if stock_section_match else text

    m = re.search(r'([\d,]+)\s*千株', search_text)
    if m:
        return float(m.group(1).replace(',', '')) * 1_000
    m = re.search(r'([\d,]+)\s*万株', search_text)
    if m:
        return float(m.group(1).replace(',', '')) * 10_000
    return None


# ============================================================
# 3. 価格データ読み込み
# ============================================================
def load_price(ticker: str) -> pd.DataFrame | None:
    csv = PRICE_DIR / f'{ticker}_T.csv'
    if not csv.exists():
        return None
    try:
        df = pd.read_csv(csv, parse_dates=['Date'], index_col='Date')
        df = df[['Close']].copy()
        df = df[df['Close'] > 0].dropna()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return None


# ============================================================
# 4. 4桁セクターマッピングの構築
# ============================================================
def build_4digit_sector_mapping() -> pd.DataFrame:
    """
    全銘柄に4桁業種コードを付与する DataFrame を返す
    columns: ticker, sector_33, code3, main_raw, sub_raw, code4, group_name
    """
    print("セクターマスター読み込み中...")
    master = pd.read_csv(SECTOR_CSV, encoding='utf-8-sig')
    master = master[master['sector'] != '-'].dropna(subset=['sector'])
    master['ticker'] = master['symbol'].str.replace('.T', '', regex=False)

    print("rawテキストからサブ業種を抽出中...")
    rows = []
    for _, row in master.iterrows():
        main_raw, sub_raw = extract_sectors_from_text(row['ticker'])
        rows.append({
            'ticker':    row['ticker'],
            'sector_33': row['sector'],
            'code3':     SECTOR_33_TO_CODE3.get(row['sector'], '???'),
            'main_raw':  main_raw,
            'sub_raw':   sub_raw,
        })
    df = pd.DataFrame(rows)

    # 有効なサブ業種グループのみ4桁コードを発行
    valid = df.dropna(subset=['main_raw', 'sub_raw'])
    combo_counts = valid.groupby(['main_raw', 'sub_raw']).size()
    valid_combos = combo_counts[combo_counts >= MIN_STOCKS_PER_GROUP].index

    # コード3桁ごとに副業種に連番を付与
    code4_map = {}
    for code3 in df['code3'].unique():
        if code3 == '???':
            continue
        sub_in_this_group = [
            (m, s) for (m, s) in valid_combos
            if SECTOR_33_TO_CODE3.get(RAW_MAIN_TO_SECTOR33.get(m, ''), '???') == code3
        ]
        for i, (m, s) in enumerate(sorted(sub_in_this_group), start=1):
            code4_map[(m, s)] = f'{code3}{i}'

    # code4 と group_name を設定
    def assign_code4(row):
        key = (row['main_raw'], row['sub_raw'])
        if pd.isna(row['main_raw']) or pd.isna(row['sub_raw']):
            return row['code3'] + '0', row['sector_33']
        c4 = code4_map.get(key)
        if c4:
            return c4, f"{row['main_raw']}/{row['sub_raw']}"
        else:
            return row['code3'] + '0', row['sector_33']

    df[['code4', 'group_name']] = df.apply(
        lambda r: pd.Series(assign_code4(r)), axis=1
    )

    n_groups = df['code4'].nunique()
    print(f"4桁グループ数: {n_groups}  (うちサブ業種: {len(code4_map)}、主業種フォールバック: {n_groups - len(code4_map)})")
    return df


# ============================================================
# 5. セクターインデックス / 時価総額最大銘柄
# ============================================================
def build_sector_index(tickers, prices, shares, date_range):
    if not prices:
        return pd.Series(dtype=float)
    price_df = pd.DataFrame({t: prices[t] for t in tickers if t in prices})
    if price_df.empty:
        return pd.Series(dtype=float)
    price_df = price_df.reindex(date_range).ffill()

    known = {t: s for t, s in shares.items() if t in price_df.columns}
    median_s = np.median(list(known.values())) if known else 1.0
    share_s = pd.Series({t: shares.get(t, median_s) for t in price_df.columns})

    mktcap = price_df.multiply(share_s, axis=1)
    total  = mktcap.sum(axis=1).replace(0, np.nan)
    weight = mktcap.div(total, axis=0)
    wp     = (price_df * weight).sum(axis=1).replace(0, np.nan).dropna()
    if len(wp) == 0:
        return pd.Series(dtype=float)
    return (wp / wp.iloc[0]) * 100


def get_largest_cap_series(tickers, prices, shares, date_range):
    if not prices:
        return pd.Series(dtype=str)
    price_df = pd.DataFrame({t: prices[t] for t in tickers if t in prices})
    if price_df.empty:
        return pd.Series(dtype=str)
    price_df = price_df.reindex(date_range).ffill()
    known = {t: s for t, s in shares.items() if t in price_df.columns}
    median_s = np.median(list(known.values())) if known else 1.0
    share_s = pd.Series({t: shares.get(t, median_s) for t in price_df.columns})
    mktcap = price_df.multiply(share_s, axis=1)
    # 全NaNの行がある場合は前の値で埋めてからidxmax
    mktcap = mktcap.ffill().dropna(how='all')
    return mktcap.idxmax(axis=1, skipna=True)


# ============================================================
# 6. MAクロスシグナル
# ============================================================
def generate_ma_signals(index, fast, slow):
    ma_f = index.rolling(fast, min_periods=fast).mean()
    ma_s = index.rolling(slow, min_periods=slow).mean()
    signal = pd.Series(0, index=index.index)
    signal[(ma_f > ma_s) & (ma_f.shift(1) <= ma_s.shift(1))] =  1
    signal[(ma_f < ma_s) & (ma_f.shift(1) >= ma_s.shift(1))] = -1
    return signal


# ============================================================
# 7. バックテスト (シグナル翌営業日執行)
# ============================================================
def run_backtest(index, largest_cap, prices, fast, slow, start, end, initial_capital=1_000_000.0):
    idx = index.loc[start:end]
    if len(idx) < slow + 10:
        return None

    signal   = generate_ma_signals(idx, fast, slow)
    position = 0
    entry_price, entry_ticker, entry_date = 0.0, None, None
    capital  = initial_capital
    equity, equity_dates = [initial_capital], [idx.index[0]]
    trades   = []

    for i, dt in enumerate(idx.index):
        sig = signal.loc[dt]

        # エグジット: 売りシグナル or 期末 → 翌営業日終値
        if position == 1 and (sig == -1 or i == len(idx) - 1):
            if entry_ticker and entry_ticker in prices:
                ps = prices[entry_ticker]
                exit_dates = ps.index[ps.index > dt]
                if len(exit_dates) > 0:
                    exit_price = ps.loc[exit_dates[0]]
                    ret = (exit_price - entry_price) / entry_price
                    capital *= (1 + ret)
                    trades.append({
                        'ticker':      entry_ticker,
                        'entry_date':  entry_date,
                        'exit_date':   exit_dates[0],
                        'entry_price': entry_price,
                        'exit_price':  float(exit_price),
                        'return_pct':  ret * 100,
                        'capital_after': capital,
                    })
            position = 0
            entry_ticker = None

        # エントリー: 買いシグナル → 翌営業日終値
        if position == 0 and sig == 1:
            ticker_today = (largest_cap.loc[dt] if dt in largest_cap.index else None)
            if ticker_today and ticker_today in prices:
                ps = prices[ticker_today]
                entry_dates = ps.index[ps.index > dt]
                if len(entry_dates) > 0:
                    ep = float(ps.loc[entry_dates[0]])
                    if ep > 0:
                        position, entry_ticker = 1, ticker_today
                        entry_price, entry_date = ep, entry_dates[0]

        equity.append(capital)
        equity_dates.append(dt)

    eq = pd.Series(equity[1:], index=equity_dates[1:])
    if len(eq) < 2:
        return None

    total_ret  = (eq.iloc[-1] / initial_capital - 1) * 100
    dr         = eq.pct_change().dropna()
    sharpe     = (dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0.0
    max_dd     = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    win_rate   = sum(1 for t in trades if t['return_pct'] > 0) / len(trades) * 100 if trades else 0.0

    return dict(total_return_pct=total_ret, sharpe=sharpe,
                max_drawdown_pct=max_dd, win_rate_pct=win_rate,
                num_trades=len(trades), equity_curve=eq, trades=trades,
                fast=fast, slow=slow)


# ============================================================
# 8. MAパラメータ最適化
# ============================================================
FAST_RANGE = [5, 10, 20, 25]
SLOW_RANGE = [25, 50, 75, 100]

def optimize_params(index, largest_cap, prices):
    best_sharpe, best_fast, best_slow = -np.inf, 25, 75
    for fast, slow in product(FAST_RANGE, SLOW_RANGE):
        if fast >= slow:
            continue
        res = run_backtest(index, largest_cap, prices, fast, slow, TRAIN_START, TRAIN_END)
        if res and res['num_trades'] >= 3 and res['sharpe'] > best_sharpe:
            best_sharpe = res['sharpe']
            best_fast, best_slow = fast, slow
    return best_fast, best_slow


# ============================================================
# 9. チャート出力
# ============================================================
def plot_sector_charts(group_name, code4, full_index, train_res, test_res, fast, slow):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1, 1]})
    fig.suptitle(f'{group_name}\n（業種コード4桁: {code4}）  MA({fast}/{slow})', fontsize=11, y=1.01)

    train_idx = full_index.loc[TRAIN_START:TRAIN_END]
    test_idx  = full_index.loc[TEST_START:TEST_END]
    ma_f = full_index.rolling(fast).mean()
    ma_s = full_index.rolling(slow).mean()

    ax = axes[0]
    ax.plot(train_idx.index, train_idx.values, color='steelblue',  lw=1.2, label='学習 (2001-2021)')
    ax.plot(test_idx.index,  test_idx.values,  color='darkorange', lw=1.2, label='テスト (2022-2025)')
    ax.plot(full_index.index, ma_f.values, '--', color='green', lw=0.8, alpha=0.7, label=f'MA{fast}')
    ax.plot(full_index.index, ma_s.values, '--', color='red',   lw=0.8, alpha=0.7, label=f'MA{slow}')
    ax.axvline(pd.Timestamp(TEST_START), color='gray', linestyle=':', lw=1.5)
    ax.set_ylabel('インデックス (base=100)', fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    for ax_i, (prefix, res, color) in enumerate(
        [('学習', train_res, 'steelblue'), ('テスト', test_res, 'darkorange')], 1
    ):
        ax_cur = axes[ax_i]
        if res:
            ax_cur.plot(res['equity_curve'].index, res['equity_curve'].values, color=color, lw=1.2)
            ax_cur.set_title(
                f'{prefix}: リターン{res["total_return_pct"]:+.1f}% | '
                f'Sharpe {res["sharpe"]:.2f} | 最大DD {res["max_drawdown_pct"]:.1f}% | '
                f'勝率 {res["win_rate_pct"]:.1f}% | 取引数 {res["num_trades"]}',
                fontsize=8
            )
        ax_cur.set_ylabel('資産 (円)', fontsize=9)
        ax_cur.grid(True, alpha=0.3)
        ax_cur.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'¥{x:,.0f}'))

    axes[2].set_xlabel('日付', fontsize=9)
    plt.tight_layout()
    safe = re.sub(r'[/・、\s]', '_', group_name)
    fname = OUT_DIR / f'sector4_{code4}_{safe[:30]}.png'
    plt.savefig(fname, dpi=100, bbox_inches='tight')
    plt.close()
    return fname


# ============================================================
# 10. メインパイプライン
# ============================================================
def main():
    print("=" * 65)
    print("業種コード4桁（主業種+副業種）バックテスト")
    print("=" * 65)

    # 4桁セクターマッピング構築
    mapping = build_4digit_sector_mapping()

    # 価格・株式数の読み込み
    print("価格データ・発行済株式数を読み込み中...")
    all_prices = {}
    all_shares = {}
    for ticker in mapping['ticker'].unique():
        pf = load_price(ticker)
        if pf is not None and len(pf) >= 100:
            all_prices[ticker] = pf['Close']
        sh = extract_shares_outstanding(ticker)
        if sh and sh > 0:
            all_shares[ticker] = sh
    print(f"価格データ読み込み完了: {len(all_prices)} 銘柄")

    # 全データの日付レンジ
    all_dates = pd.date_range(start=TRAIN_START, end=TEST_END, freq='B')

    # グループ別に処理
    groups = mapping.groupby(['code4', 'group_name'])
    total_groups = groups.ngroups
    print(f"対象グループ数: {total_groups}")

    summary_rows = []
    all_results  = {}

    for idx_g, ((code4, group_name), grp_df) in enumerate(
        sorted(groups, key=lambda x: x[0][0])
    ):
        tickers_in_group = [t for t in grp_df['ticker'].tolist() if t in all_prices]
        n = len(tickers_in_group)
        print(f"\n[{idx_g+1}/{total_groups}] {group_name} (コード: {code4}) | 銘柄数: {n}")

        if n < 2:
            print("  → 銘柄数不足, スキップ")
            continue

        # セクターインデックス
        sec_idx = build_sector_index(
            tickers_in_group, all_prices, all_shares, all_dates
        )
        if len(sec_idx) < 500:
            print("  → インデックスデータ不足, スキップ")
            continue

        # 時価総額最大銘柄
        largest_cap = get_largest_cap_series(
            tickers_in_group, all_prices, all_shares, all_dates
        )

        # MAパラメータ最適化
        print(f"  最適化中...", end='', flush=True)
        fast, slow = optimize_params(sec_idx, largest_cap, all_prices)
        print(f" best: MA({fast}/{slow})")

        # バックテスト
        train_res = run_backtest(sec_idx, largest_cap, all_prices, fast, slow, TRAIN_START, TRAIN_END)
        test_res  = run_backtest(sec_idx, largest_cap, all_prices, fast, slow, TEST_START,  TEST_END)

        # チャート保存
        plot_sector_charts(group_name, code4, sec_idx, train_res, test_res, fast, slow)

        row = {'group_name': group_name, 'code4': code4, 'n_tickers': n,
               'fast_ma': fast, 'slow_ma': slow}
        for prefix, res in [('train', train_res), ('test', test_res)]:
            if res:
                row[f'{prefix}_return_pct']   = round(res['total_return_pct'], 2)
                row[f'{prefix}_sharpe']       = round(res['sharpe'], 3)
                row[f'{prefix}_max_dd_pct']   = round(res['max_drawdown_pct'], 2)
                row[f'{prefix}_win_rate_pct'] = round(res['win_rate_pct'], 2)
                row[f'{prefix}_num_trades']   = res['num_trades']
            else:
                for k in ['return_pct','sharpe','max_dd_pct','win_rate_pct','num_trades']:
                    row[f'{prefix}_{k}'] = None
        summary_rows.append(row)
        all_results[f'{code4}_{group_name}'] = {
            'train': train_res, 'test': test_res, 'index': sec_idx,
            'fast': fast, 'slow': slow
        }

    # サマリーCSV
    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / 'sector4_backtest_summary.csv'
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"\nサマリーCSV: {summary_path}")

    # 全グループ合算エクイティカーブ (テスト期間)
    _plot_combined(all_results)
    # 全グループインデックスグリッド
    _plot_grid(all_results)

    # コンソールサマリー
    print("\n" + "=" * 90)
    print(f"{'グループ名':<30} {'コード':>6} {'銘柄数':>6} {'MA':>8} "
          f"{'学習ﾘﾀｰﾝ':>10} {'学習SR':>7} {'ﾃｽﾄﾘﾀｰﾝ':>10} {'ﾃｽﾄSR':>7} {'ﾃｽﾄ取引':>7}")
    print("-" * 90)
    for row in sorted(summary_rows, key=lambda r: r.get('test_return_pct') or -999, reverse=True):
        tr  = row.get('train_return_pct') or 0
        ts  = row.get('train_sharpe') or 0
        ter = row.get('test_return_pct') or 0
        tes = row.get('test_sharpe') or 0
        tnt = row.get('test_num_trades') or 0
        print(f"{row['group_name']:<30} {row['code4']:>6} {row['n_tickers']:>6} "
              f"MA{row['fast_ma']}/{row['slow_ma']:<3} "
              f"{tr:>+9.1f}% {ts:>7.2f} {ter:>+9.1f}% {tes:>7.2f} {tnt:>7}")
    print("=" * 90)

    test_rets = [r['test_return_pct'] for r in summary_rows if r.get('test_return_pct') is not None]
    if test_rets:
        print(f"\nテスト期間統計 ({len(test_rets)} グループ):")
        print(f"  平均リターン   : {np.mean(test_rets):+.2f}%")
        print(f"  中央値リターン : {np.median(test_rets):+.2f}%")
        print(f"  勝ちグループ数 : {sum(1 for r in test_rets if r > 0)} / {len(test_rets)}")

    print(f"\n結果保存先: {OUT_DIR}")
    return all_results, summary_df


def _plot_combined(all_results):
    eqs = []
    for res in all_results.values():
        if res['test'] and res['test']['equity_curve'] is not None:
            norm = res['test']['equity_curve'] / res['test']['equity_curve'].iloc[0]
            eqs.append(norm)
    if not eqs:
        return
    combined = pd.concat(eqs, axis=1).mean(axis=1)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(combined.index, combined.values * 100, color='navy', lw=1.5)
    ax.axhline(100, color='gray', linestyle='--', lw=0.8)
    ax.set_title('全グループ合算 正規化エクイティカーブ (テスト期間: 2022-2025) [4桁コード]', fontsize=12)
    ax.set_ylabel('パフォーマンス (base=100)', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'combined_equity_test_4digit.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f"合算チャート保存: combined_equity_test_4digit.png")


def _plot_grid(all_results):
    valid = [(k, r) for k, r in all_results.items() if r['index'] is not None and len(r['index']) > 0]
    if not valid:
        return
    n = len(valid)
    ncols = 6
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.8))
    axes_flat = axes.flatten() if nrows > 1 else axes.flatten() if ncols > 1 else [axes]
    for i, (key, res) in enumerate(valid):
        ax = axes_flat[i]
        idx = res['index']
        code4 = key.split('_')[0]
        label = '_'.join(key.split('_')[1:])[:20]
        ax.plot(idx.loc[TRAIN_START:TRAIN_END].index, idx.loc[TRAIN_START:TRAIN_END].values,
                color='steelblue', lw=0.7)
        ax.plot(idx.loc[TEST_START:TEST_END].index,   idx.loc[TEST_START:TEST_END].values,
                color='darkorange', lw=0.7)
        ax.set_title(f'{code4}\n{label}', fontsize=6)
        ax.tick_params(labelsize=5)
        ax.grid(True, alpha=0.3)
        ax.axvline(pd.Timestamp(TEST_START), color='gray', linestyle=':', lw=0.7)
    for j in range(len(valid), len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.suptitle('全グループ 時価総額加重インデックス [4桁コード]', fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'all_sector_indices_4digit.png', dpi=80, bbox_inches='tight')
    plt.close()
    print(f"全グループインデックスグリッド保存: all_sector_indices_4digit.png")


if __name__ == '__main__':
    main()

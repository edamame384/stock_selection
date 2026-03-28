"""
sector_index_backtest.py

業種別加重平均チャート + バックテストシステム

1. 四季報の業種コード(東証33業種)で全銘柄を分類
2. 分類ごとに時価総額加重平均の業種インデックスを作成
3. そのインデックスを使い移動平均クロスシグナルでバックテスト
   - 売買対象: 各業種内で時価総額最大の銘柄
   - 学習期間: 2001-01-01 ~ 2021-12-31
   - テスト期間: 2022-01-01 ~ 2025-12-31
"""

import os
import re
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from itertools import product
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================
# パス設定
# ============================================================
BASE_DIR = Path('/home/user/stock_selection/四季報投資')
PRICE_DIR = BASE_DIR / 'その他/shikiho_text_parser_runtime/data/prices_full'
RAW_DIR   = BASE_DIR / 'その他/shikiho_text_parser_runtime/data/raw/4Q-2'
SECTOR_CSV = BASE_DIR / 'その他/shikiho_text_parser_runtime/data/reference/sector_master_template.csv'
OUT_DIR   = BASE_DIR / '売買メソッド/results/sector_backtest'
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_START = '2001-01-01'
TRAIN_END   = '2021-12-31'
TEST_START  = '2022-01-01'
TEST_END    = '2025-12-31'

# 東証33業種コード（上3桁）マッピング
SECTOR_CODE_MAP = {
    '水産・農林業':        '005',
    '鉱業':              '105',
    '建設業':            '205',
    '食料品':            '210',
    '繊維製品':           '215',
    'パルプ・紙':         '220',
    '化学':              '225',
    '医薬品':            '230',
    '石油・石炭製品':      '235',
    'ゴム製品':           '240',
    'ガラス・土石製品':    '245',
    '鉄鋼':              '250',
    '非鉄金属':           '255',
    '金属製品':           '260',
    '機械':              '305',
    '電気機器':           '310',
    '輸送用機器':         '315',
    '精密機器':           '320',
    'その他製品':         '325',
    '電気・ガス業':       '330',
    '陸運業':            '335',
    '海運業':            '340',
    '空運業':            '345',
    '倉庫・運輸関連業':   '350',
    '情報・通信業':       '355',
    '卸売業':            '405',
    '小売業':            '410',
    '銀行業':            '505',
    '証券、商品先物取引業': '510',
    '保険業':            '515',
    'その他金融業':       '520',
    '不動産業':           '605',
    'サービス業':         '705',
}

# ============================================================
# 1. 四季報テキストから発行済株式数を取得
# ============================================================
def extract_shares_outstanding(ticker: str) -> float | None:
    """生テキストから発行済株式数(千株)を抽出して株数(株)を返す"""
    txt_path = RAW_DIR / f'{ticker}.txt'
    if not txt_path.exists():
        return None
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception:
        return None

    # 株式セクション内の発行済株式数を探す
    # パターン: 数字+千株 (例: 12,078千株)
    # 複数ヒットがある場合、"株式" の後に出てくる最初のものを使う
    stock_section_match = re.search(r'株式\s*\n(.*?)(?:\n\n|\Z)', text, re.DOTALL)
    search_text = stock_section_match.group(0) if stock_section_match else text

    # 千株単位
    m = re.search(r'([\d,]+)\s*千株', search_text)
    if m:
        shares_k = float(m.group(1).replace(',', ''))
        return shares_k * 1_000

    # 万株単位
    m = re.search(r'([\d,]+)\s*万株', search_text)
    if m:
        shares_man = float(m.group(1).replace(',', ''))
        return shares_man * 10_000

    return None


# ============================================================
# 2. 価格データ読み込み
# ============================================================
def load_price(ticker: str) -> pd.DataFrame | None:
    """prices_full/{ticker}_T.csv を読み込み Date をインデックスに返す"""
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
# 3. セクターデータ構築
# ============================================================
def build_sector_data():
    """
    各銘柄の価格データ・発行済株式数を読み込み、
    セクター別に整理する。

    Returns
    -------
    sector_info : dict
        {sector_name: {'tickers': [...], 'shares': {ticker: float}, 'prices': {ticker: pd.Series}}}
    """
    print("セクターマスター読み込み中...")
    master = pd.read_csv(SECTOR_CSV, encoding='utf-8-sig')
    master = master[master['sector'] != '-'].dropna(subset=['sector'])
    # symbol 列は "1301.T" 形式 → ticker は "1301"
    master['ticker'] = master['symbol'].str.replace('.T', '', regex=False)

    sector_info = {}

    for sector in master['sector'].unique():
        tickers = master[master['sector'] == sector]['ticker'].tolist()
        sector_info[sector] = {
            'code': SECTOR_CODE_MAP.get(sector, '???'),
            'tickers': tickers,
            'shares': {},
            'prices': {},
        }

    total = master['ticker'].nunique()
    print(f"対象銘柄数: {total}  セクター数: {len(sector_info)}")

    # 価格・発行済株式数の読み込み
    print("価格データ・株式数を読み込み中...")
    loaded = 0
    for sector, info in sector_info.items():
        for ticker in info['tickers']:
            price_df = load_price(ticker)
            if price_df is None or len(price_df) < 100:
                continue
            shares = extract_shares_outstanding(ticker)
            info['prices'][ticker] = price_df['Close']
            if shares and shares > 0:
                info['shares'][ticker] = shares
            loaded += 1
    print(f"価格データ読み込み完了: {loaded} 銘柄")
    return sector_info


# ============================================================
# 4. 時価総額加重平均業種インデックスの構築
# ============================================================
def build_sector_index(info: dict, date_range: pd.DatetimeIndex) -> pd.Series:
    """
    時価総額加重平均インデックス (base=100 at first valid date)
    発行済株式数が不明な銘柄は等ウェイトで代替
    """
    prices = info['prices']
    shares = info['shares']
    if not prices:
        return pd.Series(dtype=float)

    # 共通日付で価格行列を作成
    price_df = pd.DataFrame({t: s for t, s in prices.items()})
    price_df = price_df.reindex(date_range).ffill()

    # 発行済株式数が不明な銘柄には中央値を充てる
    known_shares = {t: s for t, s in shares.items() if t in price_df.columns}
    if known_shares:
        median_shares = np.median(list(known_shares.values()))
    else:
        median_shares = 1.0

    share_series = pd.Series({
        t: shares.get(t, median_shares)
        for t in price_df.columns
    })

    # 時価総額 = 発行済株式数 × 株価
    mktcap_df = price_df.multiply(share_series, axis=1)

    # ウェイト = 各銘柄の時価総額 / セクター合計時価総額
    total_mktcap = mktcap_df.sum(axis=1).replace(0, np.nan)
    weight_df = mktcap_df.div(total_mktcap, axis=0)

    # 加重平均株価 (円)
    weighted_price = (price_df * weight_df).sum(axis=1)
    weighted_price = weighted_price.replace(0, np.nan).dropna()

    if len(weighted_price) == 0:
        return pd.Series(dtype=float)

    # 正規化 (最初の有効値 = 100)
    base = weighted_price.iloc[0]
    index = (weighted_price / base) * 100
    return index


# ============================================================
# 5. 時価総額最大銘柄の特定（日次）
# ============================================================
def get_largest_cap_series(info: dict, date_range: pd.DatetimeIndex) -> pd.Series:
    """
    各日付における時価総額最大銘柄のティッカーを返す
    発行済株式数不明の銘柄は等ウェイト代替
    """
    prices = info['prices']
    shares = info['shares']
    if not prices:
        return pd.Series(dtype=str)

    price_df = pd.DataFrame({t: s for t, s in prices.items()})
    price_df = price_df.reindex(date_range).ffill()

    if price_df.empty:
        return pd.Series(dtype=str)

    known_shares = {t: s for t, s in shares.items() if t in price_df.columns}
    median_shares = np.median(list(known_shares.values())) if known_shares else 1.0

    share_series = pd.Series({
        t: shares.get(t, median_shares)
        for t in price_df.columns
    })

    mktcap_df = price_df.multiply(share_series, axis=1)
    largest = mktcap_df.idxmax(axis=1)
    return largest


# ============================================================
# 6. MAクロス戦略シグナル生成
# ============================================================
def generate_ma_signals(index: pd.Series, fast: int, slow: int) -> pd.Series:
    """
    fast日MA が slow日MA を上抜け → 買いシグナル(1)
    fast日MA が slow日MA を下抜け → 売りシグナル(-1)
    それ以外 → 0
    """
    ma_fast = index.rolling(fast, min_periods=fast).mean()
    ma_slow = index.rolling(slow, min_periods=slow).mean()

    signal = pd.Series(0, index=index.index)
    cross_up   = (ma_fast > ma_slow) & (ma_fast.shift(1) <= ma_slow.shift(1))
    cross_down = (ma_fast < ma_slow) & (ma_fast.shift(1) >= ma_slow.shift(1))
    signal[cross_up]   =  1
    signal[cross_down] = -1
    return signal


# ============================================================
# 7. バックテスト実行
# ============================================================
def run_backtest(
    index: pd.Series,
    largest_cap: pd.Series,
    prices: dict,
    fast: int,
    slow: int,
    start: str,
    end: str,
    initial_capital: float = 1_000_000.0,
) -> dict:
    """
    指定期間でMAクロス戦略をバックテスト

    Returns
    -------
    dict with keys: returns, sharpe, max_drawdown, win_rate, trades, equity_curve
    """
    idx = index.loc[start:end]
    if len(idx) < slow + 10:
        return None

    signal = generate_ma_signals(idx, fast, slow)

    # 現在ポジション追跡
    position = 0          # 0: 現金, 1: ロング
    entry_price = 0.0
    entry_ticker = None
    capital = initial_capital
    equity = [initial_capital]
    equity_dates = [idx.index[0]]
    trades = []

    dates = idx.index.tolist()

    for i, dt in enumerate(dates):
        sig = signal.loc[dt]

        # エグジット: 売りシグナル or 期末
        if position == 1 and (sig == -1 or i == len(dates) - 1):
            if entry_ticker and entry_ticker in prices:
                price_series = prices[entry_ticker]
                # 翌営業日の始値で成行き (終値で代替)
                exit_dates = price_series.index[price_series.index >= dt]
                if len(exit_dates) > 0:
                    exit_price = price_series.loc[exit_dates[0]]
                    ret = (exit_price - entry_price) / entry_price
                    capital *= (1 + ret)
                    trades.append({
                        'ticker': entry_ticker,
                        'entry_date': entry_date,
                        'exit_date': exit_dates[0],
                        'entry_price': entry_price,
                        'exit_price': float(exit_price),
                        'return_pct': ret * 100,
                        'capital_after': capital,
                    })
            position = 0
            entry_ticker = None

        # エントリー: 買いシグナル
        if position == 0 and sig == 1:
            ticker_today = largest_cap.get(dt) if hasattr(largest_cap, 'get') else (
                largest_cap.loc[dt] if dt in largest_cap.index else None
            )
            if ticker_today and ticker_today in prices:
                price_series = prices[ticker_today]
                entry_dates = price_series.index[price_series.index >= dt]
                if len(entry_dates) > 0:
                    entry_price = float(price_series.loc[entry_dates[0]])
                    if entry_price > 0:
                        position = 1
                        entry_ticker = ticker_today
                        entry_date = entry_dates[0]

        equity.append(capital)
        equity_dates.append(dt)

    equity_curve = pd.Series(equity[1:], index=equity_dates[1:])

    # パフォーマンス指標
    if len(equity_curve) < 2:
        return None

    total_return = (equity_curve.iloc[-1] / initial_capital - 1) * 100
    daily_returns = equity_curve.pct_change().dropna()

    sharpe = 0.0
    if daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

    rolling_max = equity_curve.cummax()
    drawdowns = (equity_curve - rolling_max) / rolling_max
    max_dd = drawdowns.min() * 100

    wins = [t for t in trades if t['return_pct'] > 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0.0

    return {
        'total_return_pct': total_return,
        'sharpe': sharpe,
        'max_drawdown_pct': max_dd,
        'win_rate_pct': win_rate,
        'num_trades': len(trades),
        'equity_curve': equity_curve,
        'trades': trades,
        'fast': fast,
        'slow': slow,
    }


# ============================================================
# 8. MAパラメータ最適化 (学習期間)
# ============================================================
FAST_RANGE = [5, 10, 20, 25]
SLOW_RANGE = [25, 50, 75, 100]

def optimize_params(index: pd.Series, largest_cap: pd.Series, prices: dict) -> tuple[int, int]:
    """学習期間でシャープレシオを最大化するMAパラメータを探索"""
    best_sharpe = -np.inf
    best_fast, best_slow = 25, 75  # デフォルト

    for fast, slow in product(FAST_RANGE, SLOW_RANGE):
        if fast >= slow:
            continue
        res = run_backtest(
            index, largest_cap, prices,
            fast, slow, TRAIN_START, TRAIN_END,
        )
        if res and res['num_trades'] >= 3 and res['sharpe'] > best_sharpe:
            best_sharpe = res['sharpe']
            best_fast, best_slow = fast, slow

    return best_fast, best_slow


# ============================================================
# 9. チャート出力
# ============================================================
def plot_sector_charts(sector_name: str, code: str,
                       full_index: pd.Series,
                       train_result: dict, test_result: dict,
                       fast: int, slow: int):
    """業種別インデックスチャートとエクイティカーブを保存"""

    fig, axes = plt.subplots(3, 1, figsize=(14, 12),
                              gridspec_kw={'height_ratios': [2, 1, 1]})
    fig.suptitle(
        f'{sector_name}（業種コード上3桁: {code}）\n'
        f'MA({fast}/{slow})クロス戦略',
        fontsize=12, y=1.01
    )

    # ---- (1) セクターインデックス ----
    ax = axes[0]
    train_idx = full_index.loc[TRAIN_START:TRAIN_END]
    test_idx  = full_index.loc[TEST_START:TEST_END]

    ax.plot(train_idx.index, train_idx.values, color='steelblue',   lw=1.2, label='学習期間 (2001-2021)')
    ax.plot(test_idx.index,  test_idx.values,  color='darkorange',  lw=1.2, label='テスト期間 (2022-2025)')

    ma_fast = full_index.rolling(fast).mean()
    ma_slow = full_index.rolling(slow).mean()
    ax.plot(full_index.index, ma_fast.values, '--', color='green', lw=0.8, alpha=0.7, label=f'MA{fast}')
    ax.plot(full_index.index, ma_slow.values, '--', color='red',   lw=0.8, alpha=0.7, label=f'MA{slow}')

    ax.axvline(pd.Timestamp(TEST_START), color='gray', linestyle=':', lw=1.5)
    ax.set_ylabel('インデックス (base=100)', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- (2) 学習期間エクイティカーブ ----
    ax2 = axes[1]
    if train_result:
        ec_train = train_result['equity_curve']
        ax2.plot(ec_train.index, ec_train.values, color='steelblue', lw=1.2)
        ax2.set_title(
            f'学習期間: リターン {train_result["total_return_pct"]:.1f}% | '
            f'Sharpe {train_result["sharpe"]:.2f} | '
            f'最大DD {train_result["max_drawdown_pct"]:.1f}% | '
            f'勝率 {train_result["win_rate_pct"]:.1f}% | '
            f'取引数 {train_result["num_trades"]}',
            fontsize=8
        )
    ax2.set_ylabel('資産 (円)', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'¥{x:,.0f}'))

    # ---- (3) テスト期間エクイティカーブ ----
    ax3 = axes[2]
    if test_result:
        ec_test = test_result['equity_curve']
        ax3.plot(ec_test.index, ec_test.values, color='darkorange', lw=1.2)
        ax3.set_title(
            f'テスト期間: リターン {test_result["total_return_pct"]:.1f}% | '
            f'Sharpe {test_result["sharpe"]:.2f} | '
            f'最大DD {test_result["max_drawdown_pct"]:.1f}% | '
            f'勝率 {test_result["win_rate_pct"]:.1f}% | '
            f'取引数 {test_result["num_trades"]}',
            fontsize=8
        )
    ax3.set_ylabel('資産 (円)', fontsize=9)
    ax3.set_xlabel('日付', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'¥{x:,.0f}'))

    plt.tight_layout()
    safe_name = sector_name.replace('/', '_').replace('・', '_').replace('、', '_')
    fname = OUT_DIR / f'sector_{code}_{safe_name}.png'
    plt.savefig(fname, dpi=100, bbox_inches='tight')
    plt.close()
    return fname


# ============================================================
# 10. メインパイプライン
# ============================================================
def main():
    print("=" * 60)
    print("業種別加重平均インデックス バックテスト")
    print("=" * 60)

    # データ構築
    sector_info = build_sector_data()

    # 全データの日付レンジ
    all_dates = pd.date_range(start=TRAIN_START, end=TEST_END, freq='B')

    all_results = {}
    summary_rows = []

    sectors_sorted = sorted(sector_info.items(), key=lambda x: x[1]['code'])
    total_sectors = len(sectors_sorted)

    for idx_s, (sector, info) in enumerate(sectors_sorted):
        code = info['code']
        n_tickers = len(info['prices'])
        print(f"\n[{idx_s+1}/{total_sectors}] {sector} (コード上3桁: {code}) | 銘柄数: {n_tickers}")

        if n_tickers < 2:
            print("  → 銘柄数不足, スキップ")
            continue

        # セクターインデックス構築
        sector_idx = build_sector_index(info, all_dates)
        if len(sector_idx) < 500:
            print("  → インデックスデータ不足, スキップ")
            continue

        # 時価総額最大銘柄（日次）
        largest_cap = get_largest_cap_series(info, all_dates)

        # MAパラメータ最適化
        print(f"  MAパラメータ最適化中...", end='')
        fast, slow = optimize_params(sector_idx, largest_cap, info['prices'])
        print(f" best: MA({fast}/{slow})")

        # 学習期間バックテスト
        train_res = run_backtest(
            sector_idx, largest_cap, info['prices'],
            fast, slow, TRAIN_START, TRAIN_END,
        )

        # テスト期間バックテスト
        test_res = run_backtest(
            sector_idx, largest_cap, info['prices'],
            fast, slow, TEST_START, TEST_END,
        )

        # チャート出力
        chart_path = plot_sector_charts(
            sector, code, sector_idx,
            train_res, test_res, fast, slow
        )
        print(f"  チャート保存: {chart_path.name}")

        # サマリー
        row = {
            'sector': sector,
            'code_3digit': code,
            'n_tickers': n_tickers,
            'fast_ma': fast,
            'slow_ma': slow,
        }
        for prefix, res in [('train', train_res), ('test', test_res)]:
            if res:
                row[f'{prefix}_return_pct']    = round(res['total_return_pct'], 2)
                row[f'{prefix}_sharpe']        = round(res['sharpe'], 3)
                row[f'{prefix}_max_dd_pct']    = round(res['max_drawdown_pct'], 2)
                row[f'{prefix}_win_rate_pct']  = round(res['win_rate_pct'], 2)
                row[f'{prefix}_num_trades']    = res['num_trades']
            else:
                for k in ['return_pct', 'sharpe', 'max_dd_pct', 'win_rate_pct', 'num_trades']:
                    row[f'{prefix}_{k}'] = None

        summary_rows.append(row)
        all_results[sector] = {
            'train': train_res,
            'test': test_res,
            'index': sector_idx,
            'fast': fast,
            'slow': slow,
        }

    # ============================================================
    # サマリーCSV出力
    # ============================================================
    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / 'sector_backtest_summary.csv'
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"\nサマリーCSV保存: {summary_path}")

    # ============================================================
    # 全業種合算ポートフォリオ（テスト期間）
    # ============================================================
    print("\n全業種合算エクイティカーブ (テスト期間) 作成中...")
    all_equity = []
    for sector, res in all_results.items():
        if res['test'] and res['test']['equity_curve'] is not None:
            norm = res['test']['equity_curve'] / res['test']['equity_curve'].iloc[0]
            all_equity.append(norm)

    if all_equity:
        combined = pd.concat(all_equity, axis=1).mean(axis=1)
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(combined.index, combined.values * 100, color='navy', lw=1.5)
        ax.axhline(100, color='gray', linestyle='--', lw=0.8)
        ax.set_title('全業種合算 正規化エクイティカーブ (テスト期間: 2022-2025)', fontsize=12)
        ax.set_ylabel('パフォーマンス (base=100)', fontsize=10)
        ax.set_xlabel('日付', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        combined_path = OUT_DIR / 'combined_equity_test.png'
        plt.savefig(combined_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"合算チャート保存: {combined_path}")

    # ============================================================
    # 全業種インデックスチャート
    # ============================================================
    print("全業種インデックスチャート作成中...")
    plot_all_sector_indices(all_results)

    # ============================================================
    # コンソール結果サマリー
    # ============================================================
    print("\n" + "=" * 80)
    print(f"{'業種':<20} {'コード':>6} {'銘柄数':>6} {'MA':>8} "
          f"{'学習ﾘﾀｰﾝ':>10} {'学習SR':>8} "
          f"{'ﾃｽﾄﾘﾀｰﾝ':>10} {'ﾃｽﾄSR':>8} {'ﾃｽﾄ取引数':>8}")
    print("-" * 80)
    for row in sorted(summary_rows, key=lambda r: r.get('test_return_pct') or -999, reverse=True):
        tr = row.get('train_return_pct')
        ts = row.get('train_sharpe')
        ter = row.get('test_return_pct')
        tes = row.get('test_sharpe')
        tnt = row.get('test_num_trades')
        print(
            f"{row['sector']:<20} {row['code_3digit']:>6} {row['n_tickers']:>6} "
            f"MA{row['fast_ma']}/{row['slow_ma']:>3} "
            f"{(tr if tr else 0):>+9.1f}% {(ts if ts else 0):>8.2f} "
            f"{(ter if ter else 0):>+9.1f}% {(tes if tes else 0):>8.2f} "
            f"{(tnt if tnt else 0):>8}"
        )
    print("=" * 80)

    # テスト期間の平均統計
    test_returns = [r['test_return_pct'] for r in summary_rows if r.get('test_return_pct') is not None]
    if test_returns:
        print(f"\nテスト期間統計 ({len(test_returns)} 業種):")
        print(f"  平均リターン : {np.mean(test_returns):+.2f}%")
        print(f"  中央値リターン: {np.median(test_returns):+.2f}%")
        print(f"  勝ち業種数   : {sum(1 for r in test_returns if r > 0)} / {len(test_returns)}")

    print(f"\n結果保存先: {OUT_DIR}")
    return all_results, summary_df


def plot_all_sector_indices(all_results: dict):
    """全業種のインデックスをグリッド表示"""
    valid = [(s, r) for s, r in all_results.items() if r['index'] is not None and len(r['index']) > 0]
    n = len(valid)
    if n == 0:
        return

    ncols = 5
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes_flat = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes.flatten()

    for i, (sector, res) in enumerate(valid):
        ax = axes_flat[i]
        idx = res['index']
        fast = res['fast']
        slow = res['slow']

        train = idx.loc[TRAIN_START:TRAIN_END]
        test  = idx.loc[TEST_START:TEST_END]
        ax.plot(train.index, train.values, color='steelblue', lw=0.8, label='学習')
        ax.plot(test.index,  test.values,  color='darkorange', lw=0.8, label='テスト')
        ax.set_title(f'{sector}\n({SECTOR_CODE_MAP.get(sector,"???")}) MA{fast}/{slow}', fontsize=7)
        ax.set_ylabel('Index', fontsize=6)
        ax.tick_params(labelsize=5)
        ax.grid(True, alpha=0.3)
        ax.axvline(pd.Timestamp(TEST_START), color='gray', linestyle=':', lw=0.8)

    # 余ったサブプロットを非表示
    for j in range(len(valid), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle('全業種 時価総額加重平均インデックス', fontsize=12, y=1.01)
    plt.tight_layout()
    fpath = OUT_DIR / 'all_sector_indices.png'
    plt.savefig(fpath, dpi=80, bbox_inches='tight')
    plt.close()
    print(f"全業種インデックスチャート保存: {fpath}")


if __name__ == '__main__':
    main()

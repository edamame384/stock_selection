"""
ma_param_optimize.py
MAパラメータの網羅的な最適化分析

現行: FAST=[5,10,20,25] × SLOW=[25,50,75,100] → 10組
今回: FAST=5〜60 × SLOW=20〜200 (step5) → 約300組以上
評価: 全業種平均Sharpe / 学習期間 & テスト期間の両方
目的: MA(25/100)が本当に最適かを検証 + 過学習の有無を確認
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from itertools import product
from pathlib import Path

matplotlib.rcParams['font.family'] = 'IPAGothic'
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from sector_index_backtest import (
    TRAIN_START, TRAIN_END, TEST_START, TEST_END,
    BASE_DIR, build_sector_data, build_sector_index,
    get_largest_cap_series, run_backtest,
)

OUT_DIR = BASE_DIR / '売買メソッド/results/ma_optimize'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 探索グリッド
FAST_VALUES = list(range(5, 65, 5))    # 5,10,...,60  (12種)
SLOW_VALUES = list(range(20, 210, 10)) # 20,30,...,200 (19種)


def run_grid_search(sector_built: dict) -> pd.DataFrame:
    """全(fast,slow)組み合わせで全業種平均Sharpeを算出"""
    combos = [(f, s) for f, s in product(FAST_VALUES, SLOW_VALUES) if f < s and s - f >= 10]
    print(f"探索組み合わせ数: {len(combos)}")

    rows = []
    total = len(combos)
    for i, (fast, slow) in enumerate(combos):
        if (i+1) % 30 == 0:
            print(f"  {i+1}/{total} (fast={fast}, slow={slow})", flush=True)

        train_sharpes = []
        test_sharpes  = []
        train_rets    = []
        test_rets     = []

        for sname, info in sector_built.items():
            tr = run_backtest(info['index'], info['largest_cap'], info['prices'],
                              fast, slow, TRAIN_START, TRAIN_END)
            te = run_backtest(info['index'], info['largest_cap'], info['prices'],
                              fast, slow, TEST_START, TEST_END)
            if tr and tr['num_trades'] >= 2:
                train_sharpes.append(tr['sharpe'])
                train_rets.append(tr['total_return_pct'])
            if te and te['num_trades'] >= 2:
                test_sharpes.append(te['sharpe'])
                test_rets.append(te['total_return_pct'])

        if len(train_sharpes) < 5:
            continue

        rows.append({
            'fast': fast, 'slow': slow,
            'train_sharpe_mean': np.mean(train_sharpes),
            'train_sharpe_med':  np.median(train_sharpes),
            'train_return_mean': np.mean(train_rets),
            'test_sharpe_mean':  np.mean(test_sharpes) if test_sharpes else np.nan,
            'test_sharpe_med':   np.median(test_sharpes) if test_sharpes else np.nan,
            'test_return_mean':  np.mean(test_rets) if test_rets else np.nan,
            'n_sectors_train':   len(train_sharpes),
            'n_sectors_test':    len(test_sharpes),
        })

    return pd.DataFrame(rows)


def plot_heatmaps(df: pd.DataFrame):
    """学習/テスト期間の Sharpe ヒートマップと散布図"""
    import matplotlib.dates as mdates
    import matplotlib.colors as mcolors

    # ── ヒートマップ用ピボット ──
    def make_pivot(col):
        return df.pivot(index='slow', columns='fast', values=col)

    train_piv = make_pivot('train_sharpe_mean')
    test_piv  = make_pivot('test_sharpe_mean')

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('MAパラメータ最適化 シャープレシオ ヒートマップ\n(値=全業種平均Sharpe)', fontsize=14)

    for ax, piv, title, cmap in [
        (axes[0], train_piv, f'学習期間 ({TRAIN_START[:4]}〜{TRAIN_END[:4]})', 'RdYlGn'),
        (axes[1], test_piv,  f'テスト期間 ({TEST_START[:4]}〜{TEST_END[:4]})',  'RdYlGn'),
    ]:
        vmax = max(abs(piv.values[~np.isnan(piv.values)]).max(), 0.01)
        im = ax.imshow(piv.values, aspect='auto', cmap=cmap,
                       vmin=-vmax, vmax=vmax, origin='lower')
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels(piv.columns.tolist(), fontsize=8, rotation=45)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index.tolist(), fontsize=8)
        ax.set_xlabel('Fast MA', fontsize=12)
        ax.set_ylabel('Slow MA', fontsize=12)
        ax.set_title(title, fontsize=12)
        plt.colorbar(im, ax=ax, shrink=0.8)

        # 現行(25/100)と最適値にマーカー
        if 25 in piv.columns and 100 in piv.index:
            cx = list(piv.columns).index(25)
            cy = list(piv.index).index(100)
            ax.plot(cx, cy, 'b*', markersize=14, label='現行(25/100)')

        best = piv.stack().idxmax()  # (slow, fast)
        bx = list(piv.columns).index(best[1])
        by = list(piv.index).index(best[0])
        ax.plot(bx, by, 'w^', markersize=10, label=f'最適({best[1]}/{best[0]})')
        ax.legend(fontsize=9, loc='upper right')

    plt.tight_layout()
    path = OUT_DIR / 'ma_sharpe_heatmap.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"ヒートマップ保存: {path.name}")

    # ── 学習 vs テスト散布図 ──
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
    fig2.suptitle('学習期間 vs テスト期間 Sharpe 散布図\n(過学習チェック)', fontsize=13)

    ax = axes2[0]
    sc = ax.scatter(df['train_sharpe_mean'], df['test_sharpe_mean'],
                    c=df['fast'], cmap='plasma', s=40, alpha=0.7)
    plt.colorbar(sc, ax=ax, label='Fast MA')
    # 現行(25/100)をハイライト
    cur = df[(df['fast'] == 25) & (df['slow'] == 100)]
    if len(cur):
        ax.scatter(cur['train_sharpe_mean'], cur['test_sharpe_mean'],
                   color='blue', s=150, marker='*', zorder=5, label='現行(25/100)')
    # テスト最良をハイライト
    best_test_idx = df['test_sharpe_mean'].idxmax()
    best_row = df.loc[best_test_idx]
    ax.scatter(best_row['train_sharpe_mean'], best_row['test_sharpe_mean'],
               color='red', s=150, marker='^', zorder=5,
               label=f'テスト最良({int(best_row.fast)}/{int(best_row.slow)})')
    ax.axhline(0, color='gray', lw=0.8)
    ax.axvline(0, color='gray', lw=0.8)
    ax.set_xlabel('学習Sharpe (平均)', fontsize=12)
    ax.set_ylabel('テストSharpe (平均)', fontsize=12)
    ax.set_title('fast MAで色分け', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax2 = axes2[1]
    sc2 = ax2.scatter(df['train_sharpe_mean'], df['test_sharpe_mean'],
                      c=df['slow'], cmap='viridis', s=40, alpha=0.7)
    plt.colorbar(sc2, ax=ax2, label='Slow MA')
    if len(cur):
        ax2.scatter(cur['train_sharpe_mean'], cur['test_sharpe_mean'],
                    color='blue', s=150, marker='*', zorder=5, label='現行(25/100)')
    ax2.scatter(best_row['train_sharpe_mean'], best_row['test_sharpe_mean'],
                color='red', s=150, marker='^', zorder=5,
                label=f'テスト最良({int(best_row.fast)}/{int(best_row.slow)})')
    ax2.axhline(0, color='gray', lw=0.8)
    ax2.axvline(0, color='gray', lw=0.8)
    ax2.set_xlabel('学習Sharpe (平均)', fontsize=12)
    ax2.set_ylabel('テストSharpe (平均)', fontsize=12)
    ax2.set_title('slow MAで色分け', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path2 = OUT_DIR / 'ma_train_vs_test.png'
    plt.savefig(path2, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"散布図保存: {path2.name}")


def main():
    print("=" * 60)
    print("MAパラメータ 網羅的最適化")
    print(f"Fast: {FAST_VALUES}")
    print(f"Slow: {SLOW_VALUES}")
    print("=" * 60)

    sector_info = build_sector_data()
    all_dates   = pd.date_range(start=TRAIN_START, end=TEST_END, freq='B')

    print("\n全業種インデックス構築中...")
    sector_built = {}
    for sector, info in sorted(sector_info.items(), key=lambda x: x[1]['code']):
        if len(info['prices']) < 2:
            continue
        idx = build_sector_index(info, all_dates)
        if len(idx) < 500:
            continue
        lc = get_largest_cap_series(info, all_dates)
        sector_built[sector] = {
            'index': idx, 'largest_cap': lc, 'prices': info['prices'],
        }
    print(f"有効業種数: {len(sector_built)}")

    print("\nグリッドサーチ実行中...")
    df = run_grid_search(sector_built)
    df.to_csv(OUT_DIR / 'ma_grid_results.csv', index=False, encoding='utf-8-sig')
    print(f"グリッドサーチ完了: {len(df)}組")

    # ── 上位ランキング表示 ──
    print("\n=== テスト期間 上位20 (Sharpe降順) ===")
    top_test = df.nlargest(20, 'test_sharpe_mean')[
        ['fast','slow','train_sharpe_mean','test_sharpe_mean',
         'train_return_mean','test_return_mean']
    ]
    print(f"{'fast':>6} {'slow':>6} {'学習Sharpe':>12} {'テストSharpe':>13} "
          f"{'学習リターン%':>13} {'テストリターン%':>14}")
    print("-" * 68)
    for _, r in top_test.iterrows():
        tag = " ← 現行" if r.fast == 25 and r.slow == 100 else ""
        print(f"{int(r.fast):>6} {int(r.slow):>6} "
              f"{r.train_sharpe_mean:>+12.3f} {r.test_sharpe_mean:>+13.3f} "
              f"{r.train_return_mean:>+12.1f}% {r.test_return_mean:>+13.1f}%{tag}")

    print("\n=== 学習期間 上位10 (Sharpe降順) ===")
    top_train = df.nlargest(10, 'train_sharpe_mean')[
        ['fast','slow','train_sharpe_mean','test_sharpe_mean']
    ]
    for _, r in top_train.iterrows():
        tag = " ← 現行" if r.fast == 25 and r.slow == 100 else ""
        print(f"  MA({int(r.fast)}/{int(r.slow)}): 学習={r.train_sharpe_mean:+.3f}  "
              f"テスト={r.test_sharpe_mean:+.3f}{tag}")

    # 現行(25/100)の順位
    cur = df[(df['fast'] == 25) & (df['slow'] == 100)]
    if len(cur):
        cr = cur.iloc[0]
        rank_test  = (df['test_sharpe_mean'] > cr['test_sharpe_mean']).sum() + 1
        rank_train = (df['train_sharpe_mean'] > cr['train_sharpe_mean']).sum() + 1
        print(f"\n現行MA(25/100)の順位:")
        print(f"  学習: {rank_train}/{len(df)}位  (Sharpe={cr['train_sharpe_mean']:+.3f})")
        print(f"  テスト: {rank_test}/{len(df)}位  (Sharpe={cr['test_sharpe_mean']:+.3f})")

    print("\nヒートマップ作成中...")
    plot_heatmaps(df)
    print(f"\n結果保存先: {OUT_DIR}")


if __name__ == '__main__':
    main()

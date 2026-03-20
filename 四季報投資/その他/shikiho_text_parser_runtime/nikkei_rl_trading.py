"""
Nikkei 225 ETF (1321.T) DQN 強化学習売買モデル

【データ】
  1321_T.csv (Nikkei 225 ETF): 2009年〜 ※2001年開始のETFデータが存在しないため調整
  訓練: 2009-01-01 〜 2021-12-31
  テスト: 2022-01-01 〜 2026-12-31

【モデル】
  DQN (Deep Q-Network) with Experience Replay & Target Network

【行動空間】
  0 = ホールド (現ポジション維持)
  1 = 買い    (ロング保有)
  2 = 売り    (ポジション解消)

【状態空間 (14次元)】
  1-5:  1/5/10/20/60日対数リターン
  6:    RSI(14)  正規化
  7:    MACD シグナル差
  8:    ボリンジャーバンド %B
  9:    出来高 z-score (20日)
  10:   ATR比率
  11:   短期/長期 MA クロス (5/20)
  12:   中期/長期 MA クロス (20/60)
  13:   現在ポジション (0 or 1)
  14:   含み損益率

【報酬】
  保有中: 日次対数リターン
  売買時: −0.001 (取引コスト 0.1%)

【使用方法】
  python nikkei_rl_trading.py            # 訓練+テスト実行
  python nikkei_rl_trading.py --test-only  # 保存済みモデルのテストのみ
"""

import argparse
import os
import random
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# ──────────────────────────────────────────────
# パス設定
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
PRICE_FILE = BASE_DIR / "data" / "prices_full" / "1321_T.csv"
OUTPUT_DIR = BASE_DIR / "output" / "nikkei_rl"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = OUTPUT_DIR / "dqn_model.pt"

TRAIN_START = "2009-01-01"
TRAIN_END = "2021-12-31"
TEST_START = "2022-01-01"
TEST_END = "2026-12-31"

TRANSACTION_COST = 0.001   # 取引コスト 0.1%
WINDOW = 60                # 特徴量計算に必要な最低日数

# ──────────────────────────────────────────────
# 特徴量計算
# ──────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV データから状態ベクトル用の特徴量を計算する"""
    close = df["Adj Close"]
    volume = df["Volume"].replace(0, np.nan)

    feat = pd.DataFrame(index=df.index)

    # 対数リターン
    log_ret = np.log(close / close.shift(1))
    for n in [1, 5, 10, 20, 60]:
        feat[f"ret_{n}"] = log_ret.rolling(n).sum()

    # RSI(14)  → [-1, 1]
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    feat["rsi"] = (100 - 100 / (1 + rs)) / 50 - 1

    # MACD (12-26) シグナル差 → z-score
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - signal
    feat["macd"] = (macd_hist - macd_hist.rolling(60).mean()) / (
        macd_hist.rolling(60).std() + 1e-9
    )

    # ボリンジャーバンド %B (20日)
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    feat["bb_b"] = (close - (ma20 - 2 * std20)) / (4 * std20 + 1e-9)
    feat["bb_b"] = feat["bb_b"].clip(0, 1) * 2 - 1   # → [-1, 1]

    # 出来高 z-score
    feat["vol_z"] = (volume - volume.rolling(20).mean()) / (
        volume.rolling(20).std() + 1e-9
    )
    feat["vol_z"] = feat["vol_z"].clip(-3, 3) / 3

    # ATR 比率
    high, low = df["High"], df["Low"]
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    atr = tr.rolling(14).mean()
    feat["atr_ratio"] = (atr / (close + 1e-9) - 0.01) * 20   # 大まかに正規化

    # MA クロス
    ma5 = close.rolling(5).mean()
    ma60 = close.rolling(60).mean()
    feat["cross_5_20"] = (ma5 / (ma20 + 1e-9) - 1) * 20
    feat["cross_20_60"] = (ma20 / (ma60 + 1e-9) - 1) * 20

    return feat.clip(-5, 5)


# ──────────────────────────────────────────────
# 取引環境
# ──────────────────────────────────────────────

class NikkeiTradingEnv:
    N_ACTIONS = 3  # 0=hold, 1=buy, 2=sell

    def __init__(self, df: pd.DataFrame, features: pd.DataFrame):
        # NaN行を除外 (冒頭ウォームアップ期間)
        valid = features.dropna()
        self.features = features.loc[valid.index].values.astype(np.float32)
        self.log_returns = np.log(
            df["Adj Close"] / df["Adj Close"].shift(1)
        ).loc[valid.index].fillna(0).values.astype(np.float32)
        self.n_steps = len(self.features)
        self.n_feat = self.features.shape[1]  # 12次元 (positionとpnlは後付け)
        self.reset()

    @property
    def state_dim(self):
        return self.n_feat + 2   # + position + unrealized_pnl

    def reset(self):
        self.t = 0
        self.position = 0           # 0=フラット, 1=ロング
        self.entry_price_log = 0.0  # ロング開始時の累積対数価格
        self.cum_log_price = 0.0    # 累積対数価格 (≈ ln(price))
        return self._obs()

    def _obs(self):
        base = self.features[self.t].copy()
        unrealized = 0.0
        if self.position == 1:
            unrealized = float(np.clip(self.cum_log_price - self.entry_price_log, -1, 1))
        return np.append(base, [float(self.position), unrealized])

    def step(self, action: int):
        assert 0 <= action < self.N_ACTIONS
        ret = self.log_returns[self.t]
        self.cum_log_price += ret

        reward = 0.0
        cost = 0.0

        if action == 1 and self.position == 0:   # 買い
            self.position = 1
            self.entry_price_log = self.cum_log_price
            cost = TRANSACTION_COST
        elif action == 2 and self.position == 1: # 売り
            self.position = 0
            cost = TRANSACTION_COST

        if self.position == 1:
            reward = float(ret)

        reward -= cost
        self.t += 1
        done = self.t >= self.n_steps
        next_obs = np.zeros(self.state_dim, dtype=np.float32) if done else self._obs()
        return next_obs, reward, done


# ──────────────────────────────────────────────
# DQN ネットワーク
# ──────────────────────────────────────────────

class DQN(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────────
# リプレイバッファ
# ──────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int = 20_000):
        self.buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            torch.FloatTensor(np.array(s)),
            torch.LongTensor(a),
            torch.FloatTensor(r),
            torch.FloatTensor(np.array(ns)),
            torch.FloatTensor(d),
        )

    def __len__(self):
        return len(self.buf)


# ──────────────────────────────────────────────
# 訓練
# ──────────────────────────────────────────────

def train(env: NikkeiTradingEnv, n_episodes: int = 100, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"デバイス: {device}")

    policy_net = DQN(env.state_dim, env.N_ACTIONS).to(device)
    target_net = DQN(env.state_dim, env.N_ACTIONS).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    replay = ReplayBuffer(20_000)

    epsilon = 1.0
    eps_min = 0.01
    eps_decay = (epsilon - eps_min) / (n_episodes * env.n_steps * 0.8)

    BATCH = 64
    GAMMA = 0.99
    TARGET_UPDATE = 200  # ステップ毎にターゲット更新

    step_count = 0
    history = []

    for ep in range(1, n_episodes + 1):
        state = env.reset()
        ep_reward = 0.0
        n_trades = 0

        while True:
            # ε-greedy 行動選択
            if random.random() < epsilon:
                action = random.randint(0, env.N_ACTIONS - 1)
            else:
                with torch.no_grad():
                    q = policy_net(torch.FloatTensor(state).unsqueeze(0).to(device))
                    action = q.argmax().item()

            next_state, reward, done = env.step(action)
            replay.push(state, action, reward, next_state, float(done))
            ep_reward += reward
            if action in (1, 2):
                n_trades += 1

            state = next_state
            epsilon = max(eps_min, epsilon - eps_decay)
            step_count += 1

            # 学習
            if len(replay) >= BATCH:
                s, a, r, ns, d = replay.sample(BATCH)
                s, a, r, ns, d = (x.to(device) for x in (s, a, r, ns, d))

                with torch.no_grad():
                    q_next = target_net(ns).max(1)[0]
                    q_target = r + GAMMA * q_next * (1 - d)

                q_pred = policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                loss = nn.SmoothL1Loss()(q_pred, q_target)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

            if step_count % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        history.append({"ep": ep, "reward": ep_reward, "trades": n_trades, "eps": epsilon})
        if ep % 10 == 0:
            avg_r = np.mean([h["reward"] for h in history[-10:]])
            print(f"  Episode {ep:3d}/{n_episodes}  直近10ep平均報酬={avg_r:.4f}  "
                  f"trades={n_trades}  ε={epsilon:.3f}")

    torch.save(policy_net.state_dict(), MODEL_PATH)
    print(f"\nモデル保存: {MODEL_PATH}")
    return policy_net, history


# ──────────────────────────────────────────────
# バックテスト評価
# ──────────────────────────────────────────────

def backtest(env: NikkeiTradingEnv, policy_net, device, label: str = "", initial_capital: float = 0, reinvest: bool = True, trend_filter: bool = False):
    """
    trend_filter=True のとき: 20日MA < 60日MAの下降トレンド中は強制的に現金保有
      特徴量インデックス: cross_20_60 = env.features[t][11]
      (ma20/ma60 - 1)*20 < 0 → 下降トレンド
    """
    policy_net.eval()
    state = env.reset()
    portfolio_log = [0.0]       # 累積対数リターン（複利）
    actions_log = []

    # 再投資なし用: 毎回 initial_capital 固定額で運用した累積損益
    fixed_pnl = [0.0]          # initial_capital 固定額での累積損益（円）

    with torch.no_grad():
        while True:
            # RLモデルの判断
            q = policy_net(torch.FloatTensor(state).unsqueeze(0).to(device))
            action = q.argmax().item()

            # トレンドフィルター: 20MA < 60MA（下降トレンド）なら強制売り/現金保有
            if trend_filter:
                cross_20_60 = float(env.features[env.t][11])  # (ma20/ma60-1)*20
                if cross_20_60 < 0:
                    # 下降トレンド中はポジション保有禁止
                    if env.position == 1:
                        action = 2  # 強制売り
                    else:
                        action = 0  # 買い禁止 → ホールド（現金）

            next_state, reward, done = env.step(action)
            actions_log.append(action)
            portfolio_log.append(portfolio_log[-1] + reward)
            if initial_capital > 0:
                fixed_pnl.append(fixed_pnl[-1] + initial_capital * reward)
            state = next_state
            if done:
                break

    # 買い持ち戦略との比較
    bnh = np.cumsum(env.log_returns)

    final_rl = portfolio_log[-1]
    final_bnh = bnh[-1]

    n_buy = actions_log.count(1)
    n_sell = actions_log.count(2)
    n_trades = min(n_buy, n_sell)

    # シャープレシオ (日次)
    daily_rl = np.diff(portfolio_log)
    sharpe = (daily_rl.mean() / (daily_rl.std() + 1e-9)) * np.sqrt(252)

    # 最大ドローダウン
    cum = np.array(portfolio_log)
    running_max = np.maximum.accumulate(cum)
    dd = (cum - running_max)
    max_dd = float(dd.min())

    print(f"\n{'='*50}")
    print(f"【{label}バックテスト結果】")
    print(f"  RLモデル累積対数リターン : {final_rl:.4f}  ({np.exp(final_rl)-1:.2%})")
    print(f"  買い持ち累積対数リターン : {final_bnh:.4f}  ({np.exp(final_bnh)-1:.2%})")
    print(f"  シャープレシオ           : {sharpe:.3f}")
    print(f"  最大ドローダウン         : {max_dd:.4f}  ({np.exp(max_dd)-1:.2%})")
    print(f"  売買回数                 : {n_trades}回")
    if initial_capital > 0:
        rl_final_compound = initial_capital * np.exp(final_rl)
        bnh_final = initial_capital * np.exp(final_bnh)
        max_dd_yen = initial_capital * (np.exp(max_dd) - 1)
        rl_final_fixed = initial_capital + fixed_pnl[-1]
        print(f"  ---")
        print(f"  初期資産               : {initial_capital:>14,.0f} 円")
        if reinvest:
            print(f"  RLモデル最終資産[複利] : {rl_final_compound:>14,.0f} 円  (損益 {rl_final_compound-initial_capital:+,.0f} 円)")
        else:
            print(f"  RLモデル最終資産[単利] : {rl_final_fixed:>14,.0f} 円  (損益 {fixed_pnl[-1]:+,.0f} 円)")
            dd_fixed = np.array(fixed_pnl)
            rm_fixed = np.maximum.accumulate(dd_fixed)
            max_dd_fixed = float((dd_fixed - rm_fixed).min())
            print(f"  最大ドローダウン額[単利]: {max_dd_fixed:>13,.0f} 円")
        print(f"  買い持ち最終資産       : {bnh_final:>14,.0f} 円  (損益 {bnh_final-initial_capital:+,.0f} 円)")
        if not reinvest:
            print(f"  ※単利=毎回{initial_capital:,.0f}円固定で運用、利益は再投資しない")
    print(f"{'='*50}")

    # 結果を CSV に保存
    n = len(actions_log)
    out_df = pd.DataFrame({
        "step": range(n),
        "cum_log_rl": portfolio_log[1:n + 1],
        "cum_log_bnh": bnh[:n],
        "action": actions_log,
    })
    out_path = OUTPUT_DIR / f"backtest_{label}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"詳細CSV保存: {out_path}")

    return {
        "label": label,
        "rl_log_return": final_rl,
        "rl_return": np.exp(final_rl) - 1,
        "bnh_log_return": final_bnh,
        "bnh_return": np.exp(final_bnh) - 1,
        "sharpe": sharpe,
        "max_drawdown": np.exp(max_dd) - 1,
        "n_trades": n_trades,
    }


# ──────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────

def load_data(start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(PRICE_FILE, index_col="Date", parse_dates=True)
    df = df.loc[start:end].copy()
    features = compute_features(df)
    return df, features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-only", action="store_true", help="保存済みモデルのテストのみ実行")
    parser.add_argument("--episodes", type=int, default=80, help="訓練エピソード数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--capital", type=float, default=0, help="初期資産（円）。指定時は円建て損益も表示")
    parser.add_argument("--no-reinvest", action="store_true", help="利益を再投資せず固定額で運用")
    parser.add_argument("--trend-filter", action="store_true", help="下降トレンド(20MA<60MA)時は強制現金保有")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 50)
    print("Nikkei 225 ETF (1321.T) DQN 強化学習売買モデル")
    print(f"訓練: {TRAIN_START} 〜 {TRAIN_END}")
    print(f"テスト: {TEST_START} 〜 {TEST_END}")
    print("=" * 50)

    # データ読み込み
    train_df, train_feat = load_data(TRAIN_START, TRAIN_END)
    test_df, test_feat = load_data(TEST_START, TEST_END)
    print(f"\n訓練データ: {len(train_df)}日")
    print(f"テストデータ: {len(test_df)}日")

    train_env = NikkeiTradingEnv(train_df, train_feat)
    test_env = NikkeiTradingEnv(test_df, test_feat)

    if args.test_only:
        if not MODEL_PATH.exists():
            print(f"モデルが見つかりません: {MODEL_PATH}")
            return
        policy_net = DQN(train_env.state_dim, NikkeiTradingEnv.N_ACTIONS).to(device)
        policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("保存済みモデルを読み込みました")
    else:
        print(f"\n【訓練開始】エピソード数: {args.episodes}")
        policy_net, history = train(train_env, n_episodes=args.episodes, seed=args.seed)

        # 訓練履歴保存
        pd.DataFrame(history).to_csv(OUTPUT_DIR / "train_history.csv", index=False)

    # バックテスト
    reinvest = not args.no_reinvest
    train_result = backtest(train_env, policy_net, device, label="訓練期間_", initial_capital=args.capital, reinvest=reinvest, trend_filter=args.trend_filter)
    test_result = backtest(test_env, policy_net, device, label="テスト期間_", initial_capital=args.capital, reinvest=reinvest, trend_filter=args.trend_filter)

    # サマリー保存
    summary = pd.DataFrame([train_result, test_result])
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    print(f"\nサマリー保存: {OUTPUT_DIR}/summary.csv")


if __name__ == "__main__":
    main()

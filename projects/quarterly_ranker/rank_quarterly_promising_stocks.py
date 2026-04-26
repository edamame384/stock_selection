from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_symbol(raw: str) -> str:
    symbol = str(raw).strip().upper()
    if not symbol:
        return symbol
    if symbol.startswith("TYO:"):
        return f"{symbol.split(':', 1)[1]}.T"
    if symbol.endswith(".T"):
        return symbol
    return f"{symbol}.T"


def symbol_to_price_path(symbol: str) -> str:
    return f"{symbol.replace('.', '_')}.csv"


def load_name_master(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["symbol", "company_name", "sector"])
    df = pd.read_csv(path)
    required = {"コード", "銘柄名"}
    if not required.issubset(df.columns):
        raise ValueError(f"name csv must include columns: {required}")
    out = pd.DataFrame(
        {
            "symbol": [normalize_symbol(v) for v in df["コード"].astype(str)],
            "company_name": df["銘柄名"].astype(str),
            "sector": df["33業種"].astype(str) if "33業種" in df.columns else "",
        }
    )
    return out.drop_duplicates(subset=["symbol"], keep="first")


def load_sector_master(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["symbol", "sector_local"])
    df = pd.read_csv(path)
    if not {"symbol", "sector"}.issubset(df.columns):
        raise ValueError("sector master must include columns: symbol, sector")
    return df.rename(columns={"sector": "sector_local"})[["symbol", "sector_local"]].copy()


def calc_r2_log_trend(close: pd.Series) -> float:
    y = np.log(pd.to_numeric(close, errors="coerce").dropna().astype(float))
    if len(y) < 20:
        return np.nan
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y.to_numpy(), 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y.to_numpy() - pred) ** 2))
    ss_tot = float(np.sum((y.to_numpy() - np.mean(y.to_numpy())) ** 2))
    if ss_tot <= 0:
        return np.nan
    return max(0.0, min(1.0, 1.0 - ss_res / ss_tot))


def calc_max_drawdown(close: pd.Series) -> float:
    s = pd.to_numeric(close, errors="coerce").dropna().astype(float)
    if len(s) < 2:
        return np.nan
    dd = s / s.cummax() - 1.0
    return abs(float(dd.min()))


def percentile_score(series: pd.Series, ascending: bool = True) -> pd.Series:
    ranked = series.rank(pct=True, ascending=ascending, method="average")
    return ranked.fillna(0.0)


@dataclass
class QuarterSpec:
    label: str
    start: pd.Timestamp
    end: pd.Timestamp


def build_quarters() -> list[QuarterSpec]:
    return [
        QuarterSpec("2025Q1", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-03-31")),
        QuarterSpec("2025Q2", pd.Timestamp("2025-04-01"), pd.Timestamp("2025-06-30")),
        QuarterSpec("2025Q3", pd.Timestamp("2025-07-01"), pd.Timestamp("2025-09-30")),
        QuarterSpec("2025Q4", pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-31")),
    ]


def describe_health(row: pd.Series) -> str:
    parts: list[str] = []
    if row["trend_r2"] >= 0.75:
        parts.append("上昇トレンドが滑らか")
    if row["max_drawdown_pct"] <= 15.0:
        parts.append("押し目が浅い")
    if row["positive_month_ratio_pct"] >= 65.0:
        parts.append("月次で上昇継続")
    if row["end_to_trailing_high_pct"] >= 95.0:
        parts.append("高値圏を維持")
    if row["persistence_20d_pct"] >= 70.0:
        parts.append("短中期移動平均の上で推移")
    if row["avg_monthly_high_low_change_pct"] <= 25.0:
        parts.append("月間レンジが過度でない")
    return " / ".join(parts[:4]) if parts else "上昇率は高いが変動確認が必要"


def calc_symbol_metrics(symbol: str, price_path: Path, quarter: QuarterSpec) -> dict[str, object] | None:
    df = pd.read_csv(price_path)
    if "Date" not in df.columns or "Close" not in df.columns:
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    if quarter.end not in df.index:
        hist_until_end = df[df.index <= quarter.end]
        if hist_until_end.empty:
            return None
    else:
        hist_until_end = df.loc[: quarter.end]

    qdf = hist_until_end[(hist_until_end.index >= quarter.start) & (hist_until_end.index <= quarter.end)].copy()
    if len(qdf) < 20:
        return None

    trailing = hist_until_end.tail(252).copy()
    if len(trailing) < 120:
        return None

    year_df = df[(df.index >= pd.Timestamp("2025-01-01")) & (df.index <= pd.Timestamp("2025-12-31"))].copy()
    if len(year_df) < 120:
        return None

    monthly = qdf.resample("ME").agg({"High": "max", "Low": "min"})
    monthly = monthly.dropna()
    if monthly.empty:
        return None

    annual_return = float(year_df["Close"].iloc[-1] / year_df["Close"].iloc[0] - 1.0)
    trailing_return = float(trailing["Close"].iloc[-1] / trailing["Close"].iloc[0] - 1.0)
    quarter_return = float(qdf["Close"].iloc[-1] / qdf["Close"].iloc[0] - 1.0)
    avg_monthly_range = float(((monthly["High"] / monthly["Low"]) - 1.0).mean())
    max_monthly_range = float(((monthly["High"] / monthly["Low"]) - 1.0).max())
    trend_r2 = calc_r2_log_trend(trailing["Close"])
    max_drawdown = calc_max_drawdown(trailing["Close"])

    monthly_close = trailing["Close"].resample("ME").last().dropna()
    monthly_ret = monthly_close.pct_change().dropna()
    positive_month_ratio = float((monthly_ret > 0).mean()) if len(monthly_ret) > 0 else np.nan

    ma20 = trailing["Close"].rolling(20).mean()
    ma60 = trailing["Close"].rolling(60).mean()
    last20 = trailing.iloc[-20:].copy()
    ma20_last20 = ma20.reindex(last20.index)
    ma60_last20 = ma60.reindex(last20.index)
    persistence_20d = float(((last20["Close"] > ma20_last20) & (ma20_last20 > ma60_last20)).mean())
    end_to_trailing_high = float(trailing["Close"].iloc[-1] / trailing["High"].max())

    return {
        "quarter": quarter.label,
        "ticker": symbol,
        "annual_return_pct": annual_return * 100.0,
        "trailing_return_pct": trailing_return * 100.0,
        "quarter_return_pct": quarter_return * 100.0,
        "avg_monthly_high_low_change_pct": avg_monthly_range * 100.0,
        "max_monthly_high_low_change_pct": max_monthly_range * 100.0,
        "trend_r2": trend_r2,
        "max_drawdown_pct": max_drawdown * 100.0,
        "positive_month_ratio_pct": positive_month_ratio * 100.0,
        "end_to_trailing_high_pct": end_to_trailing_high * 100.0,
        "persistence_20d_pct": persistence_20d * 100.0,
        "quarter_end_close": float(qdf["Close"].iloc[-1]),
        "price_rows": int(len(df)),
    }


def score_quarter(df: pd.DataFrame) -> pd.DataFrame:
    scored = df[df["annual_return_pct"] > 0].copy()
    if scored.empty:
        return scored
    scored["score"] = (
        0.20 * percentile_score(scored["annual_return_pct"], ascending=True)
        + 0.20 * percentile_score(scored["trailing_return_pct"], ascending=True)
        + 0.14 * percentile_score(scored["quarter_return_pct"], ascending=True)
        + 0.14 * percentile_score(scored["trend_r2"], ascending=True)
        + 0.12 * percentile_score(scored["positive_month_ratio_pct"], ascending=True)
        + 0.12 * percentile_score(scored["end_to_trailing_high_pct"], ascending=True)
        + 0.04 * percentile_score(scored["persistence_20d_pct"], ascending=True)
        + 0.02 * percentile_score(scored["max_drawdown_pct"], ascending=False)
        + 0.02 * percentile_score(scored["avg_monthly_high_low_change_pct"], ascending=False)
    )
    scored = scored.sort_values(
        ["score", "annual_return_pct", "quarter_return_pct", "trend_r2"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    scored["rank"] = np.arange(1, len(scored) + 1)
    scored["healthy_factors"] = scored.apply(describe_health, axis=1)
    return scored


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank promising Japanese stocks by quarter for 2025.")
    parser.add_argument("--price-dir", type=Path, default=Path("data/prices"))
    parser.add_argument(
        "--names-csv",
        type=Path,
        default=Path(r"c:\Users\mitsu\Downloads\四季報2026年1集 - 銘柄.csv"),
    )
    parser.add_argument("--sector-master", type=Path, default=Path("data/sector_master_template.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("projects/quarterly_ranker/output"))
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    name_master = load_name_master(args.names_csv)
    sector_master = load_sector_master(args.sector_master)

    all_rows: list[dict[str, object]] = []
    price_files = sorted(args.price_dir.glob("*.csv"))
    quarters = build_quarters()

    for quarter in quarters:
        quarter_rows: list[dict[str, object]] = []
        for price_path in price_files:
            symbol = price_path.stem.replace("_", ".")
            row = calc_symbol_metrics(symbol=symbol, price_path=price_path, quarter=quarter)
            if row is not None:
                quarter_rows.append(row)

        quarter_df = pd.DataFrame(quarter_rows)
        if quarter_df.empty:
            continue
        scored = score_quarter(quarter_df).head(args.top_n).copy()
        scored = scored.merge(name_master, how="left", left_on="ticker", right_on="symbol")
        scored = scored.merge(sector_master, how="left", left_on="ticker", right_on="symbol", suffixes=("", "_sector"))
        scored["sector"] = scored["sector"].fillna(scored["sector_local"]).fillna("")
        scored["company_name"] = scored["company_name"].fillna("")
        scored = scored.drop(columns=[c for c in ["symbol", "symbol_sector", "sector_local"] if c in scored.columns])

        out_cols = [
            "quarter",
            "rank",
            "ticker",
            "company_name",
            "sector",
            "score",
            "annual_return_pct",
            "quarter_return_pct",
            "avg_monthly_high_low_change_pct",
            "max_monthly_high_low_change_pct",
            "trend_r2",
            "max_drawdown_pct",
            "positive_month_ratio_pct",
            "end_to_trailing_high_pct",
            "persistence_20d_pct",
            "healthy_factors",
            "quarter_end_close",
            "price_rows",
        ]
        scored = scored[out_cols]
        scored.to_csv(args.out_dir / f"promising_stocks_{quarter.label}.csv", index=False, encoding="utf-8-sig")
        all_rows.append(scored)
        print(f"[OUT] {quarter.label} rows={len(scored)}")

    if all_rows:
        combined = pd.concat(all_rows, axis=0, ignore_index=True)
        combined.to_csv(args.out_dir / "promising_stocks_2025_all_quarters.csv", index=False, encoding="utf-8-sig")
        print(f"[OUT] combined rows={len(combined)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

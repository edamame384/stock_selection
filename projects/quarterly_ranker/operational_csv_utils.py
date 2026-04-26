from __future__ import annotations

from pathlib import Path

import pandas as pd


FUTURE_RESULT_COLUMNS = {
    "future_return_2024Q3_pct",
    "realized_start_date",
    "realized_end_date",
    "realized_start_close",
    "realized_end_close",
    "realized_return_pct",
}


def strip_future_result_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [col for col in df.columns if col in FUTURE_RESULT_COLUMNS]
    if not drop_cols:
        return df.copy()
    return df.drop(columns=drop_cols).copy()


def write_operational_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    strip_future_result_columns(df).to_csv(path, index=False, encoding="utf-8-sig")
    return path

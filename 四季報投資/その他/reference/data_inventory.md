# Data Inventory

## Included
- raw 4Q-2 text: 4223 files
- subset price CSV for 4Q-2 universe: 4220 files
- full price CSV archive: 4411 files
- sector master rows: 4428

## Not Included as actual datasets
- night futures / overnight futures CSV: not found in local workspace
- Dow/S&P500/Nikkei futures time series CSV: not found in local workspace

## Reference only
- futures-related run logs if present

## Reproducibility note
- this project can reproduce the current 4Q-2 text parsing, selection, and condition2 backtest without downloading new data
- futures-based strategies cannot be reproduced from this bundle because the underlying futures dataset is not present locally
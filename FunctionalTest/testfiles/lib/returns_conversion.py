import numpy as np
import pandas as pd


# 6.1 and 6.2 Calculate returns and keep each period's ending date.
def return_calculate(prices, method="arithmetic", date_column="Date"):
    if not isinstance(prices, pd.DataFrame):
        raise ValueError("prices must be a pandas DataFrame")
    if not prices.columns.is_unique:
        raise ValueError("prices must have unique column names")
    if not isinstance(method, str):
        raise ValueError("method must be arithmetic or log")
    if date_column not in prices.columns:
        raise ValueError("date column was not found")
    if len(prices) < 2:
        raise ValueError("prices must contain at least two rows")
    columns = [column for column in prices.columns if column != date_column]
    if not columns:
        raise ValueError("prices must contain at least one price column")
    values = prices[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("prices must be positive and finite")

    method = method.lower()
    if method in ("arithmetic", "discrete"):
        with np.errstate(over="ignore"):
            returns = values[1:] / values[:-1] - 1
        if not np.isfinite(returns).all():
            raise ValueError("price ratios are too large to represent")
    elif method == "log":
        # Subtract logs to avoid overflow when the price ratio is very large.
        returns = np.log(values[1:]) - np.log(values[:-1])
    else:
        raise ValueError("method must be arithmetic or log")

    result = pd.DataFrame(returns, columns=columns)
    result[date_column] = prices[date_column].iloc[1:].to_numpy()
    return result[prices.columns]

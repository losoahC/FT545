import numpy as np


def _missing_matrix(data, pairwise, correlation):
    # Rows are observations, columns are variables, and np.nan marks missing values.
    data = np.array(data, dtype=float)
    if data.ndim != 2 or data.shape[1] == 0:
        raise ValueError("data must be a 2-D array with at least one column")
    if np.isinf(data).any():
        raise ValueError("data must not contain infinity; use NaN for missing values")

    # Drop the entire row if any column has a missing value.
    if not pairwise:
        data = data[~np.isnan(data).any(axis=1)]

    columns = data.shape[1]
    result = np.full((columns, columns), np.nan)

    for i in range(columns):
        for j in range(i, columns):
            # Keep observations with nonmissing values in both selected columns.
            valid = ~np.isnan(data[:, i]) & ~np.isnan(data[:, j])
            x = data[valid, i]
            y = data[valid, j]
            n = len(x)
            if n < 2:
                continue

            # Shift first so constant decimal columns have exactly zero variance.
            x_shifted = x - x[0]
            y_shifted = y - y[0]
            x_diff = x_shifted - np.mean(x_shifted)
            y_diff = y_shifted - np.mean(y_shifted)
            cov = np.sum(x_diff * y_diff) / (n - 1)

            if correlation:
                # Use n - 1 for standard deviations to match the sample covariance.
                x_std = np.sqrt(np.sum(x_diff ** 2) / (n - 1))
                y_std = np.sqrt(np.sum(y_diff ** 2) / (n - 1))
                if x_std == 0 or y_std == 0:
                    continue
                value = np.clip(cov / (x_std * y_std), -1, 1)
            else:
                value = cov

            result[i, j] = value
            result[j, i] = value

    return result


# 1.1 Covariance matrix after dropping rows with any missing values.
def covariance_missing_data_skip_missing_rows(data):
    return _missing_matrix(data, pairwise=False, correlation=False)


# 1.2 Correlation matrix after dropping rows with any missing values.
def correlation_missing_data_skip_missing_rows(data):
    return _missing_matrix(data, pairwise=False, correlation=True)


# 1.3 Covariance matrix using pairwise deletion of missing values.
def covariance_missing_data_pairwise(data):
    return _missing_matrix(data, pairwise=True, correlation=False)


# 1.4 Correlation matrix using pairwise deletion of missing values.
def correlation_missing_data_pairwise(data):
    return _missing_matrix(data, pairwise=True, correlation=True)

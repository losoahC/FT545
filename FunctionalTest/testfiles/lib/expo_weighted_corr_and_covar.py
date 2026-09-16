import numpy as np


# 2.1 Exponentially weighted covariance, with the newest observation last.
def ew_covariance(data, decay=0.97):
    data = np.array(data, dtype=float)
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] == 0:
        raise ValueError("data must have at least two rows and one column")
    if not np.isfinite(data).all():
        raise ValueError("data must contain only finite values")
    if not 0 < decay < 1:
        raise ValueError("decay must be between 0 and 1")

    n = len(data)
    weights = decay ** np.arange(n - 1, -1, -1)
    weights = weights / np.sum(weights)

    # Shift first to avoid rounding errors in constant decimal columns.
    shifted = data - data[0]
    mean = np.sum(shifted * weights[:, None], axis=0)
    centered = shifted - mean
    columns = data.shape[1]
    result = np.zeros((columns, columns))
    for i in range(columns):
        for j in range(i, columns):
            # Normalized weights require no additional n - 1 correction.
            value = np.sum(weights * centered[:, i] * centered[:, j])
            result[i, j] = value
            result[j, i] = value
    return result


# 2.2 Convert exponentially weighted covariance to correlation.
def ew_correlation(data, decay=0.94):
    cov = ew_covariance(data, decay)
    std = np.sqrt(np.diag(cov))
    denominator = np.outer(std, std)
    result = np.full(cov.shape, np.nan)
    np.divide(cov, denominator, out=result, where=denominator > 0)
    return np.clip(result, -1, 1)


# 2.3 Combine EW variances and EW correlations with separate decay factors.
def ew_covariance_combined(data, variance_decay=0.97, correlation_decay=0.94):
    cov = ew_covariance(data, variance_decay)
    std = np.sqrt(np.diag(cov))
    corr = ew_correlation(data, correlation_decay)
    result = corr * np.outer(std, std)
    # A constant variable has zero covariance with every variable.
    result[std == 0, :] = 0
    result[:, std == 0] = 0
    return result

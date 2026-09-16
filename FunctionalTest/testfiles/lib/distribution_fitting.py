import numpy as np
from scipy import optimize, stats


def _sample(data, minimum=2):
    data = np.array(data, dtype=float)
    if data.ndim != 1 or len(data) < minimum:
        raise ValueError("data must be a one-dimensional sample of sufficient length")
    if not np.isfinite(data).all():
        raise ValueError("data must contain only finite values")
    if np.all(data == data[0]):
        raise ValueError("data must not be constant")
    return data


# 7.1 Fit a normal distribution using the sample mean and standard deviation.
def fit_normal(data):
    data = _sample(data)
    return {"mu": np.mean(data), "sigma": np.std(data, ddof=1)}


# 7.2 Fit a location-scale Student t distribution by maximum likelihood.
def fit_general_t(data):
    data = _sample(data, minimum=4)
    nu, mu, sigma = stats.t.fit(data)
    if not np.isfinite([nu, mu, sigma]).all() or nu <= 0 or sigma <= 0:
        raise RuntimeError("Student t fitting returned invalid parameters")
    # sigma is the scale parameter, not the distribution's standard deviation.
    return {"mu": mu, "sigma": sigma, "nu": nu}


# 7.3 Fit regression coefficients and Student t error parameters jointly.
def fit_regression_t(x, y):
    x = np.array(x, dtype=float)
    y = _sample(y, minimum=4)
    if x.ndim != 2 or len(x) != len(y) or x.shape[1] == 0:
        raise ValueError("x must be a predictor matrix with one row per response")
    if not np.isfinite(x).all():
        raise ValueError("x must contain only finite values")
    design = np.column_stack([np.ones(len(y)), x])
    if len(y) <= design.shape[1] + 2:
        raise ValueError("not enough observations to fit the regression")
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("predictors must not be linearly dependent")

    # Start with ordinary least squares and a positive residual scale.
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    residuals = y - design @ coefficients
    scale = np.std(residuals, ddof=1)
    if scale <= np.finfo(float).eps * max(1, np.max(np.abs(y))):
        raise ValueError("a perfect fit has no positive residual scale")
    initial = np.r_[coefficients, np.log(scale), np.log(5)]

    def negative_log_likelihood(parameters):
        # Log parameters keep sigma and nu positive during optimization.
        sigma = np.exp(parameters[-2])
        nu = np.exp(parameters[-1])
        residuals = y - design @ parameters[:-2]
        return -np.sum(stats.t.logpdf(residuals, df=nu, loc=0, scale=sigma))

    result = optimize.minimize(
        negative_log_likelihood, initial, method="Nelder-Mead",
        options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-10},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError("Student t regression did not converge")
    parameters = result.x
    fitted = {
        "mu": 0.0,
        "sigma": np.exp(parameters[-2]),
        "nu": np.exp(parameters[-1]),
        "Alpha": parameters[0],
    }
    # The intercept already handles location, so the error location stays zero.
    for i in range(x.shape[1]):
        fitted["B" + str(i + 1)] = parameters[i + 1]
    return fitted


# 7.4 Small-sample corrected Akaike information criterion.
def aicc(log_likelihood, n, k):
    if not isinstance(n, int) or not isinstance(k, int) or k < 1 or n <= k + 1:
        raise ValueError("n and k must be integers with k >= 1 and n > k + 1")
    if not np.isfinite(log_likelihood):
        raise ValueError("log_likelihood must be finite")
    return 2 * k - 2 * log_likelihood + 2 * k * (k + 1) / (n - k - 1)


# 7.5 Fit a normal inverse Gaussian distribution by the method of moments.
def fit_nig_moments(data):
    data = _sample(data, minimum=4)
    variance = np.var(data, ddof=1)
    skew = stats.skew(data, bias=True)
    kurtosis = stats.kurtosis(data, fisher=True, bias=True)
    # A finite NIG fit requires excess kurtosis > (5/3) * skewness squared.
    if kurtosis <= 5 * skew ** 2 / 3:
        raise ValueError("sample moments do not admit a finite NIG fit")

    rho = skew / np.sqrt(3 * kurtosis - 4 * skew ** 2)
    product = 3 * (1 + 4 * rho ** 2) / kurtosis
    alpha = np.sqrt(product / (variance * (1 - rho ** 2) ** 2))
    beta = rho * alpha
    delta = product / (alpha * np.sqrt(1 - rho ** 2))
    mu = np.mean(data) - delta * rho / np.sqrt(1 - rho ** 2)
    return {"mu": mu, "alpha": alpha, "beta": beta, "delta": delta}


# 7.6 Fit the same NIG distribution by maximum likelihood.
def fit_nig_mle(data):
    data = _sample(data, minimum=4)
    a, b, mu, delta = stats.norminvgauss.fit(data)
    # SciPy uses a = alpha * delta and b = beta * delta.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norminvgauss.html
    if not np.isfinite([a, b, mu, delta]).all() or delta <= 0 or a <= abs(b):
        raise RuntimeError("NIG fitting returned invalid parameters")
    return {"mu": mu, "alpha": a / delta, "beta": b / delta, "delta": delta}

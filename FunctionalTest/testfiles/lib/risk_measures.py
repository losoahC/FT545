import numpy as np
from scipy import stats


def _check_parameters(mu, sigma, alpha):
    if not np.isfinite([mu, sigma, alpha]).all() or sigma <= 0 or not 0 < alpha < 1:
        raise ValueError("mu must be finite, sigma positive, and alpha in (0, 1)")


# 8.1 and 8.4: Report losses as positive numbers; alpha is the lower-tail probability.
def normal_risk(mu, sigma, alpha=0.05):
    _check_parameters(mu, sigma, alpha)
    quantile = stats.norm.ppf(alpha)
    var = -sigma * quantile
    es = sigma * stats.norm.pdf(quantile) / alpha
    return {"VaR Absolute": var - mu, "VaR Diff from Mean": var,
            "ES Absolute": es - mu, "ES Diff from Mean": es}


# 8.2 and 8.5: sigma is the Student t scale, not its standard deviation.
def t_risk(mu, sigma, nu, alpha=0.05):
    _check_parameters(mu, sigma, alpha)
    if not np.isfinite(nu) or nu <= 1:
        raise ValueError("nu must exceed one for finite expected shortfall")
    quantile = stats.t.ppf(alpha, nu)
    var = -sigma * quantile
    es = sigma * (nu + quantile ** 2) / (nu - 1) * stats.t.pdf(quantile, nu) / alpha
    return {"VaR Absolute": var - mu, "VaR Diff from Mean": var,
            "ES Absolute": es - mu, "ES Diff from Mean": es}


# 8.3 and 8.6: Use empirical quantiles and the average of the worst alpha fraction.
def sample_risk(samples, alpha=0.05):
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 1 or len(samples) < 2 or not np.isfinite(samples).all():
        raise ValueError("samples must be a finite one-dimensional array")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    ordered = np.sort(samples)
    var = -np.quantile(ordered, alpha)
    # A fractional last observation gives exactly alpha of the empirical mass.
    tail_count = alpha * len(samples)
    count = int(np.floor(tail_count))
    tail_sum = ordered[:count].sum() + (tail_count - count) * ordered[count]
    es = -tail_sum / tail_count
    mean = samples.mean()
    return {"VaR Absolute": var, "VaR Diff from Mean": var + mean,
            "ES Absolute": es, "ES Diff from Mean": es + mean}


def simulate_t_risk(mu, sigma, nu, n_samples=10000, seed=None, alpha=0.05):
    _check_parameters(mu, sigma, alpha)
    if not np.isfinite(nu) or nu <= 1:
        raise ValueError("nu must exceed one")
    if not isinstance(n_samples, (int, np.integer)) or n_samples < 2:
        raise ValueError("n_samples must be an integer of at least two")
    rng = np.random.default_rng(seed)
    samples = stats.t.ppf(rng.random(n_samples), nu, loc=mu, scale=sigma)
    return sample_risk(samples, alpha)

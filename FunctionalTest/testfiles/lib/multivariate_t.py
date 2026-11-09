"""Rank-based multivariate t and elliptical copulas (section 13).

Degrees of freedom follow the reference's two-pass grid in 1/nu, with means
and sample standard deviations fixed, rather than a joint multivariate MLE.
"""
import numpy as np
import pandas as pd
from scipy import stats

from lib.distribution_fitting import fit_general_t
from lib.fix_nonpsd_corr_covar import higham_nearest_psd
from lib.risk_measures import sample_risk
from lib.simulations import simulate_pca


def _observations(data):
    x = np.asarray(data, dtype=float)
    if (x.ndim != 2 or len(x) < 4 or x.shape[1] < 2 or not np.isfinite(x).all()
            or np.any(np.std(x, axis=0) == 0)):
        raise ValueError("data must have at least four finite observations of nonconstant variables")
    return x


def fix_correlation(correlation, tol=1e-8, ridge=1e-8):
    r = np.array(correlation, dtype=float)
    if (r.ndim != 2 or r.shape[0] != r.shape[1] or not r.size
            or not np.isfinite(r).all() or not np.allclose(r, r.T)
            or not np.allclose(np.diag(r), 1) or np.any(np.abs(r) > 1 + tol)):
        raise ValueError("correlation must be symmetric, finite, and have unit diagonal")
    if np.linalg.eigvalsh(r).min() < -tol:
        r = higham_nearest_psd(r)
    r = (r+r.T)/2
    if np.linalg.eigvalsh(r).min() < ridge:
        r = (r + ridge*np.eye(len(r)))/(1+ridge)
    return r


def kendall_correlation(data):
    x = _observations(data)
    r = np.eye(x.shape[1])
    for i in range(x.shape[1]):
        for j in range(i+1, x.shape[1]):
            r[i, j] = r[j, i] = np.sin(np.pi*stats.kendalltau(x[:, i], x[:, j]).statistic/2)
    return fix_correlation(r)


def profile_nu(log_likelihood):
    def best(grid):
        values = np.array([log_likelihood(1/theta) for theta in grid])
        if not np.isfinite(values).all():
            raise RuntimeError("nonfinite profile likelihood")
        return int(values.argmax()), values

    coarse = np.linspace(0.01, 0.49, 200)
    i, _ = best(coarse)
    fine = np.linspace(coarse[max(i-1, 0)], coarse[min(i+1, 199)], 200)
    j, values = best(fine)
    return 1/fine[j], values[j]


def fit_multivariate_t(data):
    x = _observations(data)
    mu, sd, r = x.mean(axis=0), x.std(axis=0, ddof=1), kendall_correlation(x)
    covariance = np.outer(sd, sd)*r

    def scale(nu):
        return covariance*((nu-2)/nu)

    nu, ll = profile_nu(lambda nu: stats.multivariate_t.logpdf(x, loc=mu, shape=scale(nu), df=nu).sum())
    return mu, scale(nu), nu, ll


def _uniforms(uniforms):
    u = _observations(uniforms)
    if np.any((u <= 0) | (u >= 1)):
        raise ValueError("copula observations must be strictly between zero and one")
    return u


def gaussian_copula_ll(uniforms, correlation):
    u = _uniforms(uniforms)
    z = stats.norm.ppf(u)
    return float((stats.multivariate_normal.logpdf(z, cov=correlation) - stats.norm.logpdf(z).sum(axis=1)).sum())


def t_copula_ll(uniforms, correlation, nu):
    u = _uniforms(uniforms)
    if not np.isfinite(nu) or nu <= 0:
        raise ValueError("nu must be positive and finite")
    z = stats.t.ppf(u, nu)
    return float((stats.multivariate_t.logpdf(z, shape=correlation, df=nu) - stats.t.logpdf(z, nu).sum(axis=1)).sum())


def fit_gaussian_copula(uniforms):
    u = _uniforms(uniforms)
    r = kendall_correlation(u)
    return r, gaussian_copula_ll(u, r)


def fit_t_copula(uniforms):
    u = _uniforms(uniforms)
    r = kendall_correlation(u)
    nu, ll = profile_nu(lambda nu: t_copula_ll(u, r, nu))
    return r, nu, ll


def copula_criteria(ll, k, n):
    if (not isinstance(k, int) or k < 0 or not isinstance(n, int) or n <= k+1
            or not np.isfinite(ll)):
        raise ValueError("criteria require finite likelihood, k >= 0, and n > k+1")
    return {"AICC": -2*ll+2*k+2*k*(k+1)/(n-k-1), "BIC": k*np.log(n)-2*ll}


def tail_dependence_t(rho, nu):
    if not np.isfinite([rho, nu]).all() or not -1 <= rho <= 1 or nu <= 0:
        raise ValueError("rho must be in [-1, 1] and nu positive")
    if rho == -1:
        return 0.
    return float(2*stats.t.cdf(-np.sqrt((nu+1)*(1-rho)/(1+rho)), nu+1))


def simulate_t_copula(correlation, nu, n_samples=100000, seed=13):
    if not np.isfinite(nu) or nu <= 0:
        raise ValueError("nu must be positive and finite")
    r = fix_correlation(correlation)
    normal = simulate_pca(r, n_samples, explained=1, seed=seed)
    rng = np.random.default_rng(None if seed is None else seed+1)
    scale = np.sqrt(nu/rng.chisquare(nu, n_samples))
    return stats.t.cdf(normal*scale[:, None], nu)


def fit_t_margins(data):
    x = _observations(data)
    models = [stats.t(p["nu"], loc=p["mu"], scale=p["sigma"])
              for p in (fit_general_t(x[:, j]) for j in range(x.shape[1]))]
    u = np.column_stack([model.cdf(x[:, j]) for j, model in enumerate(models)])
    return models, np.clip(u, np.finfo(float).eps, 1-np.finfo(float).eps)


def t_copula_portfolio_risk(returns, portfolio, n_samples=100000, seed=13,
                            fitted=None):
    required = {"Stock", "currentValue"}
    if (not required.issubset(portfolio.columns) or portfolio.empty
            or not portfolio.Stock.is_unique or "Total" in portfolio.Stock.values
            or not returns.columns.is_unique or not set(portfolio.Stock).issubset(returns.columns)):
        raise ValueError("portfolio requires unique stocks with matching return columns and currentValue")
    values = portfolio.currentValue.to_numpy(float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("current values must be finite and positive")
    stocks = portfolio.Stock.tolist()
    if fitted is None:
        models, u = fit_t_margins(returns[stocks])
        r, nu, _ = fit_t_copula(u)
    else:
        models, r, nu = fitted
        if len(models) != len(stocks) or np.shape(r) != (len(stocks), len(stocks)):
            raise ValueError("fitted model dimensions must match the portfolio")
    if any(model.args[0] <= 1 for model in models):
        raise ValueError("Student t margins require nu > 1 for finite ES")
    uniforms = np.clip(simulate_t_copula(r, nu, n_samples, seed), np.finfo(float).eps, 1-np.finfo(float).eps)
    pnl = np.column_stack([model.ppf(uniforms[:, j])*values[j] for j, model in enumerate(models)])
    pnl = np.column_stack([pnl, pnl.sum(axis=1)])
    rows = []
    for j, (stock, value) in enumerate(zip(stocks+["Total"], np.r_[values, values.sum()])):
        risk = sample_risk(pnl[:, j])
        var, es = risk["VaR Absolute"], risk["ES Absolute"]
        rows.append({"Stock": stock, "VaR95": var, "ES95": es,
                     "VaR95_Pct": var/value, "ES95_Pct": es/value})
    return pd.DataFrame(rows)

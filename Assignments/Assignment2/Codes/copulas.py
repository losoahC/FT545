"""Assignment-specific elliptical copula likelihoods and simulation.

Marginal distribution parameters are fitted separately. R is estimated from
Kendall tau and held fixed while fitting the t copula degrees of freedom.
"""
import numpy as np
from scipy import optimize, stats


def gaussian_log_density(uniforms, correlation):
    """log c(u) = log multivariate-normal pdf(z) - sum log normal pdf(z_i)."""
    z = stats.norm.ppf(uniforms)
    return (stats.multivariate_normal.logpdf(z, cov=correlation)
            - stats.norm.logpdf(z).sum(axis=1))


def t_log_density(uniforms, correlation, nu):
    """A single multivariate t vector uses one shared chi-square scale.

    Its copula density divides the joint density by its univariate t densities;
    the selected *return* margins need not themselves be t distributed.
    """
    z = stats.t.ppf(uniforms, df=nu)
    return (stats.multivariate_t.logpdf(z, shape=correlation, df=nu)
            - stats.t.logpdf(z, df=nu).sum(axis=1))


def fit_t_copula(uniforms, correlation):
    """Use a coarse log-nu search followed by bounded local refinement.

    The range allows finite positive nu below two: a copula requires no finite
    latent variance. R is a scatter/correlation parameter, not a sample covariance
    of those latent scores. Upper bound 1000 approximates the Gaussian limit.
    """
    grid = np.geomspace(0.25, 1000, 60)
    values = np.array([t_log_density(uniforms, correlation, x).sum() for x in grid])
    best = int(np.argmax(values))
    if best in (0, len(grid) - 1):
        raise RuntimeError("t-copula optimum at search boundary; widen the range")
    result = optimize.minimize_scalar(
        lambda log_nu: -t_log_density(uniforms, correlation, np.exp(log_nu)).sum(),
        bounds=np.log(grid[[best - 1, best + 1]]), method="bounded",
        options={"xatol": 1e-8},
    )
    if not result.success:
        raise RuntimeError("t-copula fitting failed")
    return float(np.exp(result.x))


def simulate_uniforms(normal_scores, nu=None, rng=None):
    """Common normal draws reduce noise when comparing Gaussian and t copulas."""
    if nu is None:
        return stats.norm.cdf(normal_scores)
    scale = np.sqrt(rng.chisquare(nu, len(normal_scores)) / nu)
    return stats.t.cdf(normal_scores / scale[:, None], df=nu)

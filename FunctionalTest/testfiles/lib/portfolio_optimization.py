"""Long-only normal risk-budget and maximum-Sharpe portfolios (section 10)."""
import numpy as np
from scipy import optimize


def _covariance(covariance, allow_nonnegative=False):
    cov = np.asarray(covariance, dtype=float)
    if (cov.ndim != 2 or cov.shape[0] != cov.shape[1] or not cov.size
            or not np.isfinite(cov).all() or not np.allclose(cov, cov.T)):
        raise ValueError("covariance must be a finite symmetric square matrix")
    if np.any(np.diag(cov) <= 0) or (np.linalg.eigvalsh(cov).min() < -1e-10
                                    and not (allow_nonnegative and np.all(cov >= 0))):
        raise ValueError("covariance must be positive semidefinite with positive variances")
    return cov


def risk_parity(covariance, risk_budget=None):
    cov = _covariance(covariance)
    n = len(cov)
    budget = np.ones(n) if risk_budget is None else np.asarray(risk_budget, dtype=float)
    if budget.shape != (n,) or not np.isfinite(budget).all() or np.any(budget <= 0):
        raise ValueError("risk budgets must be finite, positive, and match the assets")
    budget = budget / budget.sum()
    # Solve Cx = b/x, the first-order condition of the convex log barrier
    # problem. Log coordinates keep positions positive even for singular C.
    scaled = cov / np.max(np.diag(cov))

    def equations(log_x):
        x = np.exp(log_x)
        return x * (scaled @ x) - budget

    result = optimize.root(equations, np.log(np.sqrt(budget / np.diag(scaled))))
    if not result.success or np.max(np.abs(equations(result.x))) > 1e-8:
        raise RuntimeError("risk parity did not converge")
    weights = np.exp(result.x)
    return weights / weights.sum()


def max_sharpe(covariance, means, risk_free_rate, bounds=None):
    # The workbook deliberately reuses the indefinite matrix from 5.3.
    # Its nonnegative entries and positive diagonal ensure positive variance
    # for every nonzero long-only portfolio, so use it directly as Julia does.
    cov = _covariance(covariance, allow_nonnegative=True)
    n = len(cov)
    means = np.asarray(means, dtype=float)
    if means.shape != (n,) or not np.isfinite(means).all() or not np.isfinite(risk_free_rate):
        raise ValueError("finite means must match the covariance dimensions")
    bounds = np.tile([0., 1.], (n, 1)) if bounds is None else np.asarray(bounds, dtype=float)
    if (bounds.shape != (n, 2) or not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] < 0) or np.any(bounds[:, 1] > 1)
            or np.any(bounds[:, 0] > bounds[:, 1])
            or bounds[:, 0].sum() > 1 or bounds[:, 1].sum() < 1):
        raise ValueError("bounds must admit a fully invested long-only portfolio")
    excess = means - risk_free_rate

    def objective(w):
        variance = w @ cov @ w
        if variance <= 0:
            raise ValueError("portfolio must have positive variance")
        return -(w @ excess) / np.sqrt(variance)

    def gradient(w):
        variance = w @ cov @ w
        return -excess / np.sqrt(variance) + (w @ excess) * (cov @ w) / variance**1.5

    slack = bounds[:, 1] - bounds[:, 0]
    start = bounds[:, 0].copy()
    if slack.sum() > 0:
        start += (1 - start.sum()) * slack / slack.sum()
    result = optimize.minimize(objective, start, jac=gradient, method="SLSQP",
                               bounds=bounds, constraints={"type": "eq", "fun": lambda w: w.sum() - 1,
                                                           "jac": lambda w: np.ones(n)},
                               options={"ftol": 1e-13, "maxiter": 2000})
    if not result.success:
        raise RuntimeError("maximum Sharpe optimization failed: " + result.message)
    return result.x

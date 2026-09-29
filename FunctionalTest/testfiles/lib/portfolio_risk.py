import numpy as np
import pandas as pd
from scipy import stats

from lib.distribution_fitting import fit_normal, fit_general_t
from lib.risk_measures import sample_risk
from lib.simulations import simulate_pca


# 9.1: Gaussian copula with fitted Normal or Student t marginal distributions.
def portfolio_risk(returns, portfolio, n_samples=100000, seed=None, alpha=0.05):
    if alpha != 0.05:
        raise ValueError("this report uses alpha=0.05 for its VaR95 and ES95 columns")
    required = {"Stock", "Holding", "Starting Price", "Distribution"}
    if not required.issubset(portfolio.columns):
        raise ValueError("portfolio is missing required columns")
    if portfolio.empty or not portfolio["Stock"].is_unique or "Total" in portfolio["Stock"].values:
        raise ValueError("portfolio must have unique stocks and no stock named Total")
    stocks = portfolio["Stock"].tolist()
    if not returns.columns.is_unique or not set(stocks).issubset(returns.columns):
        raise ValueError("returns must contain a unique column for every stock")
    values = portfolio["Holding"].to_numpy(float) * portfolio["Starting Price"].to_numpy(float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("current position values must be positive and finite")
    models = []
    for stock, kind in zip(stocks, portfolio["Distribution"]):
        if kind == "Normal":
            fitted = fit_normal(returns[stock])
            model = stats.norm(loc=fitted["mu"], scale=fitted["sigma"])
        elif kind == "T":
            fitted = fit_general_t(returns[stock])
            if fitted["nu"] <= 1:
                raise ValueError("Student t degrees of freedom must exceed one for finite ES")
            model = stats.t(fitted["nu"], loc=fitted["mu"], scale=fitted["sigma"])
        else:
            raise ValueError("Distribution must be Normal or T")
        models.append(model)

    # Monotone marginal CDFs preserve ranks. Match the reference's direct
    # Spearman correlation convention, without an additional sine transform.
    correlation = returns[stocks].corr(method="spearman").to_numpy()
    latent = simulate_pca(correlation, n_samples, explained=1, seed=seed)
    uniforms = np.clip(stats.norm.cdf(latent), np.finfo(float).eps, 1 - np.finfo(float).eps)
    pnl = np.column_stack([model.ppf(uniforms[:, i]) * values[i]
                           for i, model in enumerate(models)])
    all_pnl = np.column_stack([pnl, pnl.sum(axis=1)])
    rows = []
    for i, (stock, value) in enumerate(zip(stocks + ["Total"], np.r_[values, values.sum()])):
        risk = sample_risk(all_pnl[:, i], alpha)
        rows.append({"Stock": stock, "VaR95": risk["VaR Absolute"],
                     "ES95": risk["ES Absolute"],
                     "VaR95_Pct": risk["VaR Absolute"] / value,
                     "ES95_Pct": risk["ES Absolute"] / value})
    return pd.DataFrame(rows)

"""Buy-and-hold ex-post attribution with Carino return linking (section 11)."""
import numpy as np
import pandas as pd


def expost_factor(weights, returns, factor_returns, betas, include_alpha=True):
    w = np.array(weights, dtype=float)
    r = np.asarray(returns, dtype=float)
    f = np.asarray(factor_returns, dtype=float)
    b = np.asarray(betas, dtype=float)
    if (r.ndim != 2 or f.ndim != 2 or len(r) < 2 or len(r) != len(f)
            or w.shape != (r.shape[1],) or b.shape != (r.shape[1], f.shape[1])
            or not all(np.isfinite(a).all() for a in (w, r, f, b))
            or np.any(w < 0) or not np.isclose(w.sum(), 1)
            or np.any(r <= -1) or np.any(f <= -1)):
        raise ValueError("weights, returns, factors, and betas have incompatible or invalid values")
    names = list(factor_returns.columns) if isinstance(factor_returns, pd.DataFrame) else [f"F{i+1}" for i in range(f.shape[1])]
    if len(set(names)) != len(names) or {"Value", "Alpha", "Portfolio"}.intersection(names):
        raise ValueError("factor names must be unique and cannot use report column names")
    portfolio = np.empty(len(r))
    contributions = np.empty_like(f)
    for t in range(len(r)):
        contributions[t] = f[t] * (w @ b)
        updated = w * (1 + r[t])
        portfolio[t] = updated.sum() - 1
        w = updated / updated.sum()
    residual = portfolio - contributions.sum(axis=1)
    components = np.column_stack([contributions, residual])
    total = np.expm1(np.log1p(portfolio).sum())

    def log_ratio(x):
        # Continuous extension log(1+x)/x = 1 at x=0.
        x = np.asarray(x)
        return np.divide(np.log1p(x), x, out=np.ones_like(x), where=x != 0)

    linking = log_ratio(portfolio) / log_ratio(total)
    volatility = np.std(portfolio, ddof=1)
    if volatility == 0:
        vol_attribution = np.zeros(components.shape[1])
    else:
        vol_attribution = ((portfolio - portfolio.mean()) @ (components - components.mean(axis=0))
                           / (len(r) - 1) / volatility)
    raw = np.column_stack([f, residual])
    if np.any(raw <= -1):
        raise ValueError("factor and residual returns must exceed -1 for geometric linking")
    rows = np.vstack([np.expm1(np.log1p(raw).sum(axis=0)),
                      (components * linking[:, None]).sum(axis=0), vol_attribution])
    report = pd.DataFrame(rows, columns=names + ["Alpha"])
    report.insert(0, "Value", ["TotalReturn", "Return Attribution", "Vol Attribution"])
    report["Portfolio"] = [total, total, volatility]
    return report if include_alpha else report.drop(columns="Alpha")

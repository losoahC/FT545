from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lib.portfolio_risk import portfolio_risk

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_workbook_9_1():
    returns = pd.read_csv(DATA_DIR / "test9_1_returns.csv")
    portfolio = pd.read_csv(DATA_DIR / "test9_1_portfolio.csv")
    original = portfolio.copy()
    actual = portfolio_risk(returns, portfolio, seed=9)
    expected = pd.read_csv(DATA_DIR / "testout9_1.csv")
    assert actual.columns.tolist() == expected.columns.tolist()
    assert actual.Stock.tolist() == expected.Stock.tolist()
    # Independent 100,000-draw simulations permit a 4% relative difference.
    np.testing.assert_allclose(actual.iloc[:, 1:], expected.iloc[:, 1:], rtol=0.04)
    values = np.r_[portfolio.Holding * portfolio["Starting Price"], 5000]
    np.testing.assert_allclose(actual.VaR95_Pct, actual.VaR95 / values)
    np.testing.assert_allclose(actual.ES95_Pct, actual.ES95 / values)
    assert np.all(actual.ES95 >= actual.VaR95)
    pd.testing.assert_frame_equal(portfolio, original)


def test_position_scaling_and_reproducibility():
    returns = pd.read_csv(DATA_DIR / "test9_1_returns.csv")
    portfolio = pd.read_csv(DATA_DIR / "test9_1_portfolio.csv")
    first = portfolio_risk(returns, portfolio, n_samples=2000, seed=1)
    portfolio["Holding"] *= 2
    second = portfolio_risk(returns, portfolio, n_samples=2000, seed=1)
    np.testing.assert_allclose(second[["VaR95", "ES95"]], 2 * first[["VaR95", "ES95"]])
    np.testing.assert_allclose(second[["VaR95_Pct", "ES95_Pct"]], first[["VaR95_Pct", "ES95_Pct"]])


def test_unknown_distribution_rejected():
    returns = pd.read_csv(DATA_DIR / "test9_1_returns.csv")
    portfolio = pd.read_csv(DATA_DIR / "test9_1_portfolio.csv")
    portfolio.loc[0, "Distribution"] = "unknown"
    with pytest.raises(ValueError):
        portfolio_risk(returns, portfolio)

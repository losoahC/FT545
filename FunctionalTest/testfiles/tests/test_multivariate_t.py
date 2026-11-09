from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from lib.multivariate_t import (
    kendall_correlation, fix_correlation, fit_multivariate_t, fit_t_margins,
    fit_gaussian_copula, fit_t_copula, gaussian_copula_ll, t_copula_ll,
    copula_criteria, tail_dependence_t, simulate_t_copula, t_copula_portfolio_risk,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def fitted():
    returns = pd.read_csv(DATA_DIR / "test13_returns.csv")
    models, u = fit_t_margins(returns)
    r, nu, ll = fit_t_copula(u)
    return returns, models, u, r, nu, ll


def test_workbook_13_1(fitted):
    returns, _, u, r, _, _ = fitted
    expected = pd.read_csv(DATA_DIR / "testout13_1.csv")
    np.testing.assert_allclose(r, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(kendall_correlation(returns), r, atol=1e-12)
    np.testing.assert_allclose(kendall_correlation(u**3), r, atol=1e-12)
    assert np.linalg.eigvalsh(r).min() > 0


def test_workbook_13_2_through_13_4(fitted):
    returns = fitted[0]
    mu, scale, nu, ll = fit_multivariate_t(returns)
    np.testing.assert_allclose(mu, pd.read_csv(DATA_DIR / "testout13_2.csv").mu, atol=1e-15)
    np.testing.assert_allclose(scale, pd.read_csv(DATA_DIR / "testout13_3.csv"), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose([nu, ll], pd.read_csv(DATA_DIR / "testout13_4.csv").iloc[0], rtol=1e-10)
    np.testing.assert_allclose(np.diag(scale)*nu/(nu-2), returns.var(), rtol=1e-12)
    assert ll == pytest.approx(stats.multivariate_t.logpdf(returns, loc=mu, shape=scale, df=nu).sum())


def test_workbook_13_5_through_13_7(fitted):
    returns, _, u, r, nu, ll_t = fitted
    rg, ll_g = fit_gaussian_copula(u)
    np.testing.assert_allclose(rg, r)
    assert ll_g == pytest.approx(pd.read_csv(DATA_DIR / "testout13_5.csv").ll[0], rel=1e-6)
    np.testing.assert_allclose([nu, ll_t], pd.read_csv(DATA_DIR / "testout13_6.csv").iloc[0], rtol=1e-6)
    expected = pd.read_csv(DATA_DIR / "testout13_7.csv")
    for i, (ll, k) in enumerate([(ll_g, 0), (ll_t, 1)]):
        np.testing.assert_allclose(list(copula_criteria(ll, k, len(returns)).values()),
                                   expected.loc[i, ["AICC", "BIC"]].to_numpy(float), rtol=1e-6)
    assert copula_criteria(ll_t, 1, len(returns))["BIC"] < copula_criteria(ll_g, 0, len(returns))["BIC"]


def test_workbook_13_8(fitted):
    r, nu = fitted[3:5]
    for row in pd.read_csv(DATA_DIR / "testout13_8.csv").itertuples():
        assert r[row.I-1, row.J-1] == pytest.approx(row.Rho, abs=1e-12)
        assert tail_dependence_t(row.Rho, nu) == pytest.approx(row.Lambda, rel=1e-10)
    assert tail_dependence_t(-1, nu) == 0
    assert tail_dependence_t(1, nu) == 1
    assert tail_dependence_t(.5, 5) > tail_dependence_t(.5, 100)


def test_workbook_13_9_and_position_scaling(fitted):
    returns, models, _, r, nu, _ = fitted
    portfolio = pd.read_csv(DATA_DIR / "test13_portfolio.csv")
    original = portfolio.copy()
    fit = (models, r, nu)
    actual = t_copula_portfolio_risk(returns, portfolio, fitted=fit, seed=13)
    expected = pd.read_csv(DATA_DIR / "testout13_9.csv")
    assert actual.columns.tolist() == expected.columns.tolist()
    assert actual.Stock.tolist() == expected.Stock.tolist()
    np.testing.assert_allclose(actual.iloc[:, 1:], expected.iloc[:, 1:], rtol=.04)
    values = np.r_[portfolio.currentValue, portfolio.currentValue.sum()]
    np.testing.assert_allclose(actual.VaR95_Pct, actual.VaR95/values)
    np.testing.assert_allclose(actual.ES95_Pct, actual.ES95/values)
    assert np.all(actual.ES95 > actual.VaR95)
    assert actual.ES95.iloc[-1] < actual.ES95.iloc[:-1].sum()
    first = t_copula_portfolio_risk(returns, portfolio, fitted=fit, n_samples=2000, seed=1)
    scaled = portfolio.copy()
    scaled.currentValue *= 2
    second = t_copula_portfolio_risk(returns, scaled, fitted=fit, n_samples=2000, seed=1)
    np.testing.assert_allclose(second.iloc[:, 1:3], first.iloc[:, 1:3]*2)
    np.testing.assert_allclose(second.iloc[:, 3:], first.iloc[:, 3:])
    pd.testing.assert_frame_equal(portfolio, original)


def test_simulated_uniform_margins_and_joint_tail():
    r = np.array([[1., .4], [.4, 1.]])
    u = simulate_t_copula(r, 6, 60000, seed=545)
    np.testing.assert_allclose(u.mean(axis=0), .5, atol=.004)
    for column in u.T:
        assert stats.kstest(column, "uniform").statistic < .008
    assert np.mean((u[:, 0] < .01) & (u[:, 1] < .01)) > .001
    np.testing.assert_array_equal(simulate_t_copula(r, 6, 10, seed=3), simulate_t_copula(r, 6, 10, seed=3))


def test_copula_density_limits():
    u = np.random.default_rng(1).uniform(.01, .99, (100, 3))
    assert gaussian_copula_ll(u, np.eye(3)) == pytest.approx(0, abs=1e-12)
    r = np.full((3, 3), .3)+np.eye(3)*.7
    assert t_copula_ll(u, r, 100000) == pytest.approx(gaussian_copula_ll(u, r), abs=.01)


def test_correlation_repairs():
    for r in (np.ones((3, 3)), np.array([[1, .9, .9], [.9, 1, -.9], [.9, -.9, 1]])):
        original = r.copy()
        repaired = fix_correlation(r)
        np.testing.assert_allclose(np.diag(repaired), 1)
        assert np.linalg.eigvalsh(repaired).min() > 0
        np.testing.assert_array_equal(r, original)


@pytest.mark.parametrize("data", [np.ones((5, 2)), [[1, 2]], [[1, 2], [2, 3], [3, np.nan], [4, 5]]])
def test_invalid_observations(data):
    with pytest.raises(ValueError):
        fit_multivariate_t(data)


@pytest.mark.parametrize("boundary", [0, 1, np.nan])
def test_invalid_uniforms(boundary):
    u = np.random.default_rng(1).uniform(.1, .9, (10, 2))
    u[0, 0] = boundary
    with pytest.raises(ValueError):
        fit_gaussian_copula(u)
    with pytest.raises(ValueError):
        fit_t_copula(u)


@pytest.mark.parametrize("ll,k,n", [(0, -1, 10), (0, 1, 2), (np.nan, 0, 10)])
def test_invalid_criteria(ll, k, n):
    with pytest.raises(ValueError):
        copula_criteria(ll, k, n)


def test_invalid_portfolio(fitted):
    returns = fitted[0]
    portfolio = pd.read_csv(DATA_DIR / "test13_portfolio.csv")
    portfolio.loc[0, "currentValue"] = -1
    with pytest.raises(ValueError):
        t_copula_portfolio_risk(returns, portfolio)

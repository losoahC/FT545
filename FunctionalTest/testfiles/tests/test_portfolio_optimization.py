from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lib.portfolio_optimization import risk_parity, max_sharpe

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.mark.parametrize("case,budget", [(1, None), (2, [1, 1, 1, 1, .5])])
def test_workbook_risk_parity(case, budget):
    cov = pd.read_csv(DATA_DIR / "test5_2.csv").to_numpy()
    actual = risk_parity(cov, budget)
    expected = pd.read_csv(DATA_DIR / f"testout10_{case}.csv").W
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    assert actual.sum() == pytest.approx(1)
    assert np.all(actual > 0)
    contributions = actual*(cov@actual)
    target = np.ones(5) if budget is None else np.array(budget)
    np.testing.assert_allclose(contributions/contributions.sum(), target/target.sum(), atol=1e-8)


@pytest.mark.parametrize("case,bounds", [(3, None), (4, np.tile([.1, .5], (5, 1)))])
def test_workbook_max_sharpe(case, bounds):
    cov = pd.read_csv(DATA_DIR / "test5_3.csv").to_numpy()
    means = pd.read_csv(DATA_DIR / "test10_3_means.csv").Mean.to_numpy()
    w = max_sharpe(cov, means, .04, bounds)
    expected = pd.read_csv(DATA_DIR / f"testout10_{case}.csv").W
    np.testing.assert_allclose(w, expected, rtol=1e-5, atol=5e-7)
    assert w.sum() == pytest.approx(1)
    limits = np.tile([0., 1.], (5, 1)) if bounds is None else bounds
    assert np.all(w >= limits[:, 0]-1e-10) and np.all(w <= limits[:, 1]+1e-10)
    # No feasible small transfer between two assets improves the objective.
    sharpe = lambda x: (x@means-.04)/np.sqrt(x@cov@x)
    for i in range(5):
        for j in range(5):
            perturbed = w.copy()
            perturbed[i] += 1e-4
            perturbed[j] -= 1e-4
            if np.all(perturbed >= limits[:, 0]) and np.all(perturbed <= limits[:, 1]):
                assert sharpe(perturbed) <= sharpe(w)+1e-10


def test_diagonal_risk_parity_and_scale_invariance():
    cov = np.diag([1., 4., 9.])
    expected = np.array([1, .5, 1/3])
    expected /= expected.sum()
    np.testing.assert_allclose(risk_parity(cov), expected)
    np.testing.assert_allclose(risk_parity(cov*100), expected)


@pytest.mark.parametrize("cov", [np.eye(2)*np.nan, [[1, -2], [-2, 1]], [[1, 0], [1, 1]], [[0, 0], [0, 1]]])
def test_invalid_covariance(cov):
    with pytest.raises(ValueError):
        risk_parity(cov)
    with pytest.raises(ValueError):
        max_sharpe(cov, [.1, .2], .04)


@pytest.mark.parametrize("budget", [[1], [0, 1], [-1, 1], [1, np.nan]])
def test_invalid_budget(budget):
    with pytest.raises(ValueError):
        risk_parity(np.eye(2), budget)


@pytest.mark.parametrize("bounds", [[[.6, 1], [.6, 1]], [[0, .4], [0, .4]], [[.5, .1], [0, 1]]])
def test_infeasible_bounds(bounds):
    with pytest.raises(ValueError):
        max_sharpe(np.eye(2), [.1, .2], .04, bounds)

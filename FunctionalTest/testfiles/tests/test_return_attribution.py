from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lib.return_attribution import expost_factor

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.mark.parametrize("case", [1, 2])
def test_workbook_attribution(case):
    weights = pd.read_csv(DATA_DIR / f"test11_{case}_weights.csv").W
    if case == 1:
        returns = pd.read_csv(DATA_DIR / "test11_1_returns.csv")
        factors, betas = returns, np.eye(3)
    else:
        returns = pd.read_csv(DATA_DIR / "test11_2_stock_returns.csv")
        factors = pd.read_csv(DATA_DIR / "test11_2_factor_returns.csv")
        betas = pd.read_csv(DATA_DIR / "test11_2_beta.csv").iloc[:, 1:]
    original = returns.copy()
    actual = expost_factor(weights, returns, factors, betas, include_alpha=case == 2)
    expected = pd.read_csv(DATA_DIR / f"testout11_{case}.csv")
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(actual.iloc[1:, 1:-1].sum(axis=1), actual.Portfolio.iloc[1:], atol=1e-12)
    terminal_wealth = np.prod(1+returns.to_numpy(), axis=0) @ weights.to_numpy()
    assert actual.Portfolio.iloc[0] == pytest.approx(terminal_wealth-1)
    pd.testing.assert_frame_equal(returns, original)


def test_zero_returns_have_finite_linking():
    r = pd.DataFrame({"A": [0., .1, -1/11], "B": [0., .1, -1/11]})
    report = expost_factor([.5, .5], r, r, np.eye(2))
    assert np.isfinite(report.iloc[:, 1:].to_numpy()).all()
    np.testing.assert_allclose(report.iloc[1, 1:-1].sum(), report.Portfolio.iloc[1], atol=1e-15)
    zero = expost_factor([.5, .5], r*0, r*0, np.eye(2))
    np.testing.assert_allclose(zero.iloc[:, 1:], 0)


@pytest.mark.parametrize("weights,returns,factors,betas", [
    ([.2, .2], np.zeros((3, 2)), np.zeros((3, 1)), np.ones((2, 1))),
    ([.5, .5], np.zeros((3, 2)), np.zeros((2, 1)), np.ones((2, 1))),
    ([.5, .5], np.full((3, 2), -1), np.zeros((3, 1)), np.ones((2, 1))),
    ([.5, .5], np.zeros((3, 2)), np.zeros((3, 1)), np.ones((1, 1))),
])
def test_invalid_inputs(weights, returns, factors, betas):
    with pytest.raises(ValueError):
        expost_factor(weights, returns, factors, betas)

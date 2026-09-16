"""Cross-check public inputs and outputs independently of the reference CSVs."""

import numpy as np
import pandas as pd
import pytest

from lib.chol_psd import chol_psd
from lib.corr_and_covar import (
    covariance_missing_data_skip_missing_rows,
    correlation_missing_data_skip_missing_rows,
    covariance_missing_data_pairwise,
    correlation_missing_data_pairwise,
)
from lib.expo_weighted_corr_and_covar import (
    ew_covariance, ew_correlation, ew_covariance_combined,
)
from lib.fix_nonpsd_corr_covar import near_psd, higham_nearest_psd
from lib.distribution_fitting import fit_normal, fit_general_t, fit_nig_moments, fit_nig_mle
from lib.returns_conversion import return_calculate
from lib.simulations import simulate_normal, simulate_pca


@pytest.mark.parametrize("function, kind, pairwise", [
    (covariance_missing_data_skip_missing_rows, "cov", False),
    (correlation_missing_data_skip_missing_rows, "corr", False),
    (covariance_missing_data_pairwise, "cov", True),
    (correlation_missing_data_pairwise, "corr", True),
])
def test_missing_data_against_pandas(function, kind, pairwise):
    rng = np.random.default_rng(545)
    data = rng.normal(size=(100, 4))
    data[rng.random(data.shape) < 0.15] = np.nan
    original = data.copy()
    frame = pd.DataFrame(data)
    if not pairwise:
        frame = frame.dropna()
    expected = frame.cov() if kind == "cov" else frame.corr()
    actual = function(data)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("function", [
    covariance_missing_data_skip_missing_rows,
    covariance_missing_data_pairwise, ew_covariance, ew_covariance_combined,
])
def test_decimal_constant_has_zero_covariance(function):
    data = np.column_stack([np.full(40, 0.1), np.arange(40)])
    actual = function(data)
    np.testing.assert_array_equal(actual[0, :], 0)
    np.testing.assert_array_equal(actual[:, 0], 0)


@pytest.mark.parametrize("function", [
    correlation_missing_data_skip_missing_rows,
    correlation_missing_data_pairwise, ew_correlation,
])
def test_decimal_constant_has_undefined_correlation(function):
    data = np.column_stack([np.full(40, 0.1), np.arange(40)])
    actual = function(data)
    assert np.isnan(actual[0, :]).all()
    assert np.isnan(actual[:, 0]).all()
    assert actual[1, 1] == pytest.approx(1)


@pytest.mark.parametrize("function", [fit_normal, fit_general_t, fit_nig_moments, fit_nig_mle])
def test_decimal_constant_cannot_be_fitted(function):
    with pytest.raises(ValueError, match="constant"):
        function([0.1] * 40)


def test_ew_against_numpy_weighted_covariance():
    data = np.random.default_rng(545).normal(size=(100, 4))
    original = data.copy()
    weights = 0.92 ** np.arange(99, -1, -1)
    expected = np.cov(data, rowvar=False, aweights=weights, ddof=0)
    np.testing.assert_allclose(ew_covariance(data, 0.92), expected, atol=1e-12)
    # Shifting a variable's level must not change covariance or correlation.
    np.testing.assert_allclose(ew_covariance(data + 100, 0.92), expected, atol=1e-12)
    std = np.sqrt(np.diag(expected))
    np.testing.assert_allclose(ew_correlation(data, 0.92),
                               expected / np.outer(std, std), atol=1e-12)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("function", [near_psd, higham_nearest_psd])
def test_repair_multiple_indefinite_inputs(function):
    rng = np.random.default_rng(545)
    for _ in range(10):
        matrix = rng.uniform(-1.5, 1.5, (4, 4))
        matrix = (matrix + matrix.T) / 2
        np.fill_diagonal(matrix, 1)
        std = rng.uniform(0.1, 3, 4)
        matrix *= np.outer(std, std)
        original = matrix.copy()
        actual = function(matrix)
        assert actual.shape == matrix.shape
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, actual.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(actual), np.diag(matrix), atol=1e-12)
        assert np.linalg.eigvalsh(actual / np.outer(std, std))[0] >= -1.1e-9
        np.testing.assert_array_equal(matrix, original)


def test_cholesky_low_rank_inputs():
    rng = np.random.default_rng(545)
    for _ in range(30):
        loadings = rng.normal(size=(5, 2))
        matrix = loadings @ loadings.T
        original = matrix.copy()
        root = chol_psd(matrix)
        np.testing.assert_allclose(root @ root.T, matrix, rtol=1e-8, atol=1e-10)
        np.testing.assert_array_equal(root, np.tril(root))
        np.testing.assert_array_equal(matrix, original)


@pytest.mark.parametrize("pca", [False, True])
@pytest.mark.parametrize("matrix", [[[4]], [[0]], [[0, 0], [0, 2]]])
def test_simulation_dimensions_and_no_mutation(pca, matrix):
    covariance = np.array(matrix, dtype=float)
    original = covariance.copy()
    if pca:
        samples = simulate_pca(covariance, 100, explained=1, seed=545)
    else:
        samples = simulate_normal(100, covariance, seed=545)
    assert samples.shape == (100, len(matrix))
    assert np.isfinite(samples).all()
    for i in range(len(matrix)):
        if covariance[i, i] == 0:
            np.testing.assert_array_equal(samples[:, i], 0)
    np.testing.assert_array_equal(covariance, original)


def test_returns_reject_duplicate_columns():
    prices = pd.DataFrame([[1, 2, 3], [2, 3, 4]], columns=["Date", "A", "A"])
    with pytest.raises(ValueError, match="unique"):
        return_calculate(prices)


def test_log_returns_with_extreme_price_ratio():
    prices = pd.DataFrame({"Date": ["a", "b"], "A": [1e-200, 1e200]})
    result = return_calculate(prices, method="log")
    assert result["A"].iloc[0] == pytest.approx(400 * np.log(10))
    with pytest.raises(ValueError, match="too large"):
        return_calculate(prices)


def test_returns_preserve_dates_with_nondefault_index():
    dates = pd.date_range("2020-01-01", periods=3)
    prices = pd.DataFrame({"Date": dates, "A": [100, 110, 99]}, index=[8, 2, 5])
    actual = return_calculate(prices)
    pd.testing.assert_series_equal(actual["Date"], pd.Series(dates[1:], name="Date"))
    np.testing.assert_allclose(actual["A"], [0.1, -0.1])


@pytest.mark.parametrize("prices, method", [([[1, 2]], "log"),
    (pd.DataFrame({"Date": [1, 2], "A": [1, 2]}), None)])
def test_returns_invalid_argument_types(prices, method):
    with pytest.raises(ValueError):
        return_calculate(prices, method=method)

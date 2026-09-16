from pathlib import Path

import numpy as np
import pytest

from lib.expo_weighted_corr_and_covar import (
    ew_covariance,
    ew_correlation,
    ew_covariance_combined,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.mark.parametrize("function, expected_file", [
    pytest.param(ew_covariance, "testout_2.1.csv", id="2.1"),
    pytest.param(ew_correlation, "testout_2.2.csv", id="2.2"),
    pytest.param(ew_covariance_combined, "testout_2.3.csv", id="2.3"),
])
def test_workbook_expected_output(function, expected_file):
    data = np.genfromtxt(DATA_DIR / "test2.csv", delimiter=",", skip_header=1)
    expected = np.genfromtxt(DATA_DIR / expected_file, delimiter=",", skip_header=1)
    actual = function(data)
    assert actual.shape == expected.shape == (5, 5)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_weighted_mean_and_observation_order():
    # Weights are 1/7, 2/7, 4/7; the weighted mean is 18/7.
    actual = ew_covariance([[0], [1], [4]], decay=0.5)
    np.testing.assert_allclose(actual, [[138 / 49]])


def test_matching_decay_factors():
    data = [[1, 4], [2, 1], [5, 3]]
    actual = ew_covariance_combined(data, 0.8, 0.8)
    np.testing.assert_allclose(actual, ew_covariance(data, 0.8))


def test_constant_column():
    data = [[0, 1], [0, 3], [0, 5]]
    corr = ew_correlation(data)
    assert np.isnan(corr[0, :]).all()
    assert np.isnan(corr[:, 0]).all()
    assert corr[1, 1] == pytest.approx(1)
    cov = ew_covariance_combined(data)
    np.testing.assert_array_equal(cov[0, :], [0, 0])
    np.testing.assert_array_equal(cov[:, 0], [0, 0])


@pytest.mark.parametrize("decay", [0, 1, -0.5, np.nan])
def test_invalid_decay(decay):
    with pytest.raises(ValueError):
        ew_covariance([[1], [2]], decay)


@pytest.mark.parametrize("data", [[1, 2], [[1]], [[1], [np.nan]]])
def test_invalid_data(data):
    with pytest.raises(ValueError):
        ew_covariance(data)

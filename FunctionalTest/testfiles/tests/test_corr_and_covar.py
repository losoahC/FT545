"""Tests against the input/output mapping in Tests.xlsx, cases 1.1–1.4."""

from pathlib import Path

import numpy as np
import pytest

from lib.corr_and_covar import (
    correlation_missing_data_pairwise,
    correlation_missing_data_skip_missing_rows,
    covariance_missing_data_pairwise,
    covariance_missing_data_skip_missing_rows,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CASES = [
    pytest.param(covariance_missing_data_skip_missing_rows, "testout_1.1.csv", id="1.1"),
    pytest.param(correlation_missing_data_skip_missing_rows, "testout_1.2.csv", id="1.2"),
    pytest.param(covariance_missing_data_pairwise, "testout_1.3.csv", id="1.3"),
    pytest.param(correlation_missing_data_pairwise, "testout_1.4.csv", id="1.4"),
]
FUNCTIONS = [
    covariance_missing_data_skip_missing_rows,
    correlation_missing_data_skip_missing_rows,
    covariance_missing_data_pairwise,
    correlation_missing_data_pairwise,
]


def read_csv(filename):
    """Read numeric CSV data, interpreting empty fields as NaN."""
    return np.genfromtxt(DATA_DIR / filename, delimiter=",", skip_header=1, ndmin=2)


@pytest.mark.parametrize("function, expected_file", CASES)
def test_workbook_expected_output(function, expected_file):
    actual = function(read_csv("test1.csv"))
    expected = read_csv(expected_file)
    assert actual.shape == expected.shape == (5, 5)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_missing_row_policy():
    data = [[1, 2, 0], [3, 4, 1], [9, 8, np.nan]]
    complete = covariance_missing_data_skip_missing_rows(data)
    pairwise = covariance_missing_data_pairwise(data)
    assert complete[0, 0] == pytest.approx(2)
    assert pairwise[0, 0] == pytest.approx(52 / 3)
    assert pairwise[0, 2] == pytest.approx(1)


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("data", [[], [1, 2], [[np.inf]], [[-np.inf]]])
def test_invalid_input(function, data):
    with pytest.raises(ValueError):
        function(data)


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("data", [np.empty((0, 2)), [[1, 2]], [[np.nan, np.nan]]])
def test_insufficient_observations(function, data):
    actual = function(data)
    assert actual.shape == (2, 2)
    assert np.isnan(actual).all()


@pytest.mark.parametrize("function", FUNCTIONS[2:])
def test_pairwise_without_shared_observations(function):
    actual = function([[1, np.nan], [2, np.nan], [np.nan, 3], [np.nan, 4]])
    assert np.isfinite(actual.diagonal()).all()
    assert np.isnan(actual[0, 1]) and np.isnan(actual[1, 0])


@pytest.mark.parametrize("function", [FUNCTIONS[1], FUNCTIONS[3]])
def test_constant_column_correlation(function):
    actual = function([[1, 2], [1, 3], [1, 4]])
    assert np.isnan(actual[0, :]).all()
    assert np.isnan(actual[:, 0]).all()
    assert actual[1, 1] == pytest.approx(1)


@pytest.mark.parametrize("function", FUNCTIONS)
def test_single_column(function):
    actual = function([[1], [3], [np.nan]])
    expected = 1 if function in (FUNCTIONS[1], FUNCTIONS[3]) else 2
    np.testing.assert_allclose(actual, [[expected]])

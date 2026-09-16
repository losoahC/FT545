from pathlib import Path

import numpy as np
import pytest

from lib.fix_nonpsd_corr_covar import near_psd, higham_nearest_psd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.mark.parametrize("function, input_file, expected_file", [
    pytest.param(near_psd, "testout_1.3.csv", "testout_3.1.csv", id="3.1"),
    pytest.param(near_psd, "testout_1.4.csv", "testout_3.2.csv", id="3.2"),
    pytest.param(higham_nearest_psd, "testout_1.3.csv", "testout_3.3.csv", id="3.3"),
    pytest.param(higham_nearest_psd, "testout_1.4.csv", "testout_3.4.csv", id="3.4"),
])
def test_workbook_expected_output(function, input_file, expected_file):
    matrix = np.genfromtxt(DATA_DIR / input_file, delimiter=",", skip_header=1)
    expected = np.genfromtxt(DATA_DIR / expected_file, delimiter=",", skip_header=1)
    actual = function(matrix)
    assert actual.shape == expected.shape == (5, 5)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("function", [near_psd, higham_nearest_psd])
def test_repair_preserves_variances(function):
    matrix = np.array([[4, 3], [3, 1]], dtype=float)
    original = matrix.copy()
    actual = function(matrix)
    np.testing.assert_allclose(actual, actual.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(actual), np.diag(matrix))
    # Higham's stopping tolerance applies before restoring the variances.
    std = np.sqrt(np.diag(actual))
    corr = actual / np.outer(std, std)
    assert np.linalg.eigvalsh(corr)[0] >= -1e-9
    np.testing.assert_array_equal(matrix, original)


@pytest.mark.parametrize("function", [near_psd, higham_nearest_psd])
@pytest.mark.parametrize("matrix", [[[2, 0.5], [0.5, 1]], [[1, 1], [1, 1]], [[4]]])
def test_psd_input_is_unchanged(function, matrix):
    np.testing.assert_allclose(function(matrix), matrix, atol=1e-12)


@pytest.mark.parametrize("function", [near_psd, higham_nearest_psd])
@pytest.mark.parametrize("matrix", [
    [], [[1, 2]], [[1, 2], [0, 1]], [[np.nan]], [[np.inf]], [[0]], [[-1]],
])
def test_invalid_matrix(function, matrix):
    with pytest.raises(ValueError):
        function(matrix)


def test_higham_iteration_limit():
    with pytest.raises(RuntimeError, match="did not converge"):
        higham_nearest_psd([[1, 2], [2, 1]], max_iterations=1)

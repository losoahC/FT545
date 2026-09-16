from pathlib import Path

import numpy as np
import pytest

from lib.chol_psd import chol_psd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_workbook_4_1():
    matrix = np.genfromtxt(DATA_DIR / "testout_3.1.csv", delimiter=",", skip_header=1)
    expected = np.genfromtxt(DATA_DIR / "testout_4.1.csv", delimiter=",", skip_header=1)
    root = chol_psd(matrix)
    np.testing.assert_allclose(root, expected, rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(root @ root.T, matrix, atol=1e-12)


@pytest.mark.parametrize("matrix", [
    [[4, 2], [2, 3]], [[1, 1], [1, 1]], [[0, 0], [0, 1]], [[0]],
])
def test_reconstruction(matrix):
    root = chol_psd(matrix)
    np.testing.assert_array_equal(root, np.tril(root))
    np.testing.assert_allclose(root @ root.T, matrix, atol=1e-12)


@pytest.mark.parametrize("matrix", [
    [[1, 2], [2, 1]], [[0, 1], [1, 0]], [[1, 2], [0, 1]], [[np.nan]], [1, 2],
])
def test_invalid_matrix(matrix):
    with pytest.raises(ValueError):
        chol_psd(matrix)

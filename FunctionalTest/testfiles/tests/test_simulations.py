from pathlib import Path

import numpy as np
import pytest

from lib.fix_nonpsd_corr_covar import near_psd, higham_nearest_psd
from lib.simulations import simulate_normal, simulate_pca

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def read_csv(filename):
    return np.genfromtxt(DATA_DIR / filename, delimiter=",", skip_header=1)


@pytest.mark.parametrize("case, input_file, fix_method", [
    pytest.param("5.1", "test5_1.csv", None, id="5.1"),
    pytest.param("5.2", "test5_2.csv", None, id="5.2"),
    pytest.param("5.3", "test5_3.csv", near_psd, id="5.3"),
    pytest.param("5.4", "test5_3.csv", higham_nearest_psd, id="5.4"),
    pytest.param("5.5", "test5_2.csv", None, id="5.5"),
])
def test_workbook_simulation(case, input_file, fix_method):
    covariance = read_csv(input_file)
    expected = read_csv("testout_" + case + ".csv")
    n = 100000
    if case == "5.5":
        samples = simulate_pca(covariance, n, explained=0.99, seed=545)
        # Construct the theoretical covariance of the retained components.
        values, vectors = np.linalg.eigh(covariance)
        values, vectors = values[::-1], vectors[:, ::-1]
        count = np.searchsorted(np.cumsum(values) / np.sum(values), 0.99) + 1
        target = (vectors[:, :count] * values[:count]) @ vectors[:, :count].T
    else:
        samples = simulate_normal(n, covariance, fix_method=fix_method, seed=545)
        target = covariance if fix_method is None else fix_method(covariance)

    assert samples.shape == (n, 5)
    assert np.isfinite(samples).all()
    actual = np.cov(samples, rowvar=False)
    variances = np.diag(target)
    # Gaussian sample covariance standard errors, including small entries.
    standard_error = np.sqrt((np.outer(variances, variances) + target ** 2) / (n - 1))
    assert np.all(np.abs(actual - target) <= 6 * standard_error)
    assert np.all(np.abs(expected - target) <= 6 * standard_error)
    # Julia and NumPy generate different draws, so allow both sampling errors.
    assert np.all(np.abs(actual - expected) <= 6 * np.sqrt(2) * standard_error)
    assert np.all(np.abs(samples.mean(axis=0)) <= 6 * np.sqrt(variances / n))


def test_reproducible_samples():
    covariance = [[1, 0.5], [0.5, 1]]
    np.testing.assert_array_equal(
        simulate_normal(100, covariance, seed=1),
        simulate_normal(100, covariance, seed=1),
    )
    np.testing.assert_array_equal(
        simulate_pca(covariance, 100, seed=1),
        simulate_pca(covariance, 100, seed=1),
    )


def test_pca_variance_cutoff():
    covariance = np.diag([9, 1])
    truncated = simulate_pca(covariance, 1000, explained=0.9, seed=1)
    full = simulate_pca(covariance, 1000, explained=1, seed=1)
    np.testing.assert_array_equal(truncated[:, 1], 0)
    assert np.var(full[:, 1]) > 0


def test_non_psd_requires_repair():
    with pytest.raises(ValueError):
        simulate_normal(100, [[1, 2], [2, 1]])
    with pytest.raises(ValueError):
        simulate_pca([[1, 2], [2, 1]], 100)


@pytest.mark.parametrize("explained", [0, 1.1, np.nan])
def test_invalid_explained(explained):
    with pytest.raises(ValueError):
        simulate_pca(np.eye(2), 100, explained=explained)


@pytest.mark.parametrize("n", [0, 1, 2.5])
def test_invalid_sample_count(n):
    with pytest.raises(ValueError):
        simulate_normal(n, np.eye(2))
    with pytest.raises(ValueError):
        simulate_pca(np.eye(2), n)

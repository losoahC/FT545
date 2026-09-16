import numpy as np

from lib.chol_psd import chol_psd


# 5.1-5.4 Generate normal samples, optionally repairing the covariance first.
def simulate_normal(n_samples, covariance, fix_method=None, seed=None):
    if not isinstance(n_samples, int) or n_samples < 2:
        raise ValueError("n_samples must be an integer of at least two")
    covariance = np.array(covariance, dtype=float)
    if fix_method is not None:
        covariance = fix_method(covariance)
    root = chol_psd(covariance)
    rng = np.random.default_rng(seed)
    normal = rng.standard_normal((n_samples, len(root)))
    return normal @ root.T


# 5.5 Retain the fewest principal components reaching the requested variance.
def simulate_pca(covariance, n_samples, explained=0.99, seed=None):
    if not isinstance(n_samples, int) or n_samples < 2:
        raise ValueError("n_samples must be an integer of at least two")
    if not 0 < explained <= 1:
        raise ValueError("explained must be in (0, 1]")
    covariance = np.array(covariance, dtype=float)
    if (covariance.ndim != 2 or covariance.size == 0
            or covariance.shape[0] != covariance.shape[1]):
        raise ValueError("covariance must be a nonempty square matrix")
    if not np.isfinite(covariance).all():
        raise ValueError("covariance must contain only finite values")
    if not np.allclose(covariance, covariance.T, rtol=0, atol=1e-12):
        raise ValueError("covariance must be symmetric")

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    limit = 1e-8 * np.max(np.abs(covariance))
    if eigenvalues[0] < -limit:
        raise ValueError("covariance must be positive semidefinite")
    eigenvalues = np.maximum(eigenvalues[::-1], 0)
    eigenvectors = eigenvectors[:, ::-1]
    if np.sum(eigenvalues) == 0:
        return np.zeros((n_samples, len(covariance)))

    cumulative = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    cumulative[-1] = 1
    count = np.searchsorted(cumulative, explained) + 1
    loadings = eigenvectors[:, :count] * np.sqrt(eigenvalues[:count])
    rng = np.random.default_rng(seed)
    normal = rng.standard_normal((n_samples, count))
    return normal @ loadings.T

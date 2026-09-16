import numpy as np


def _to_correlation(matrix):
    matrix = np.array(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.size == 0:
        raise ValueError("matrix must be a nonempty square matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("matrix must contain only finite values")
    if not np.allclose(matrix, matrix.T, rtol=0, atol=1e-12):
        raise ValueError("matrix must be symmetric")
    if np.any(np.diag(matrix) <= 0):
        raise ValueError("matrix must have a positive diagonal")

    # Convert covariance to correlation while saving the original scales.
    std = np.sqrt(np.diag(matrix))
    corr = matrix / np.outer(std, std)
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)
    return corr, std


def _positive_part(matrix):
    # A symmetric matrix is PSD when all its eigenvalues are nonnegative.
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, 0)
    result = (eigenvectors * eigenvalues) @ eigenvectors.T
    return (result + result.T) / 2


# 3.1 and 3.2: Repair covariance or correlation by clipping eigenvalues.
def near_psd(matrix):
    corr, std = _to_correlation(matrix)
    result = _positive_part(corr)

    # Rescale the diagonal to one, then restore the original variances.
    scale = np.sqrt(np.diag(result))
    result = result / np.outer(scale, scale)
    return result * np.outer(std, std)


# 3.3 and 3.4: Higham's alternating projections with Dykstra's correction.
def higham_nearest_psd(matrix, tolerance=1e-9, max_iterations=1000):
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite")
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")

    corr, std = _to_correlation(matrix)
    result = corr.copy()
    correction = np.zeros_like(corr)
    previous_distance = np.inf

    for _ in range(max_iterations):
        adjusted = result - correction
        projected = _positive_part(adjusted)
        correction = projected - adjusted

        # Project back to matrices with a unit diagonal.
        result = projected.copy()
        np.fill_diagonal(result, 1)

        distance = np.sum((result - corr) ** 2)
        min_eigenvalue = np.linalg.eigvalsh(result)[0]
        if (abs(distance - previous_distance) < tolerance
                and min_eigenvalue >= -tolerance):
            return result * np.outer(std, std)
        previous_distance = distance

    raise RuntimeError("Higham's algorithm did not converge")

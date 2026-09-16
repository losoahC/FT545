import numpy as np


# 4.1 Cholesky factorization for positive semidefinite matrices.
def chol_psd(matrix, tolerance=1e-8):
    matrix = np.array(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.size == 0:
        raise ValueError("matrix must be a nonempty square matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("matrix must contain only finite values")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite")
    if not np.allclose(matrix, matrix.T, rtol=0, atol=1e-12):
        raise ValueError("matrix must be symmetric")

    n = len(matrix)
    root = np.zeros_like(matrix)
    scale = np.max(np.abs(matrix))
    limit = tolerance * scale
    if np.linalg.eigvalsh(matrix)[0] < -limit:
        raise ValueError("matrix must be positive semidefinite")

    for j in range(n):
        diagonal = matrix[j, j] - np.sum(root[j, :j] ** 2)
        if diagonal < -limit:
            raise ValueError("matrix must be positive semidefinite")
        # Treat a tiny negative pivot from rounding as zero.
        root[j, j] = np.sqrt(max(diagonal, 0))
        for i in range(j + 1, n):
            residual = matrix[i, j] - np.dot(root[i, :j], root[j, :j])
            if root[j, j] > 0:
                root[i, j] = residual / root[j, j]
            elif abs(residual) > limit:
                raise ValueError("matrix must be positive semidefinite")
    return root

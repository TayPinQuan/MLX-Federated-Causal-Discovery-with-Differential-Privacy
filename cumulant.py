# cumulant.py
"""MLX-native computation of higher-order cumulant tensors."""
from __future__ import annotations

import mlx.core as mx


def local_cumulant_tensor(
    X: mx.array,
    standardize: bool = False,
    max_order: int = 3,
) -> tuple[dict[int, mx.array], int]:
    """
    Compute second- and third-order cumulant tensors.

    Args:
        X: MLX array of shape (n_samples, n_vars).
        standardize: If True, apply marginal standardization. Default False
            because standardization destroys the scale signal that encodes
            causal direction (see FedRCD paper).
        max_order: 2 or 3. If 2, only the covariance is returned.

    Returns:
        cumulants: dict {2: C2, 3: C3} (C3 only if max_order >= 3).
        n_samples: number of local samples.
    """
    n, d = X.shape

    # Center only; keep scale information intact.
    X = X - mx.mean(X, axis=0)

    if standardize:
        X = X / (mx.std(X, axis=0) + 1e-8)

    cumulants: dict[int, mx.array] = {}

    # Second-order cumulant == covariance.
    C2 = mx.matmul(X.T, X) / n
    cumulants[2] = C2

    if max_order >= 3:
        # C3[i, j, k] = E[x_i * x_j * x_k]
        # Computed as (X^T @ (X * x_i))[j, k] / n for each i.
        C3 = mx.zeros((d, d, d))
        for i in range(d):
            Xi = X * X[:, i : i + 1]          # (n, d)
            C3[i] = mx.matmul(X.T, Xi) / n    # (d, d)
        cumulants[3] = C3

    return cumulants, n


def to_numpy_cumulants(cumulants: dict[int, mx.array]) -> dict[int, "np.ndarray"]:
    """Convert MLX cumulant tensors to NumPy for Flower transport."""
    import numpy as np
    return {order: np.array(C) for order, C in cumulants.items()}
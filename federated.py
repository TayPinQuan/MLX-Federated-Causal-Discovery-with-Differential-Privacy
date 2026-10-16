# federated.py
"""Top-level wrapper for federated causal discovery.

Implements single-round federated aggregation directly in NumPy/MLX,
without depending on Flower. This is the reference implementation
that verifies the core claim: sample-weighted cumulant aggregation is
exact in expectation.

For the Flower-based implementation, see federated_flower.py.
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx

from cumulant import local_cumulant_tensor
from dp import add_dp_noise
from hc_lingam import hc_lingam


def _client_fit(
    data: np.ndarray,
    epsilon: float,
    delta: float,
    standardize: bool,
    seed: int,
) -> tuple[dict[int, np.ndarray], int]:
    """Simulate one client's fit: local cumulants + DP noise."""
    X = mx.array(np.asarray(data, dtype=np.float32))
    cumulants, n = local_cumulant_tensor(X, standardize=standardize)
    noisy = add_dp_noise(cumulants, n, epsilon, delta, seed=seed)
    return {order: np.array(C) for order, C in noisy.items()}, n


def run_federated_lingam(
    client_data: list[np.ndarray],
    epsilon: float,
    delta: float = 1e-5,
    standardize: bool = False,
    ridge_alpha: float = 1e-3,
    seed: int = 42,
) -> tuple[np.ndarray, list[int]]:
    """
    Run one federated round and return the global adjacency and order.

    Each client computes local cumulants (with optional DP noise) and
    sends them to the server. The server performs a sample-weighted
    average, then runs HC-LiNGAM.
    """
    # --- Client phase (simulated locally) ---
    client_results = []
    for k, data in enumerate(client_data):
        cumulants, n = _client_fit(
            data, epsilon, delta, standardize, seed=seed + k
        )
        client_results.append((cumulants, n))

    # --- Server phase: sample-weighted aggregation ---
    total_n = sum(n for _, n in client_results)
    C2_global = sum(
        cumulants[2] * (n / total_n)
        for cumulants, n in client_results
    )
    C3_global = sum(
        cumulants[3] * (n / total_n)
        for cumulants, n in client_results
    )

    # --- HC-LiNGAM on aggregated cumulants ---
    n_vars = C2_global.shape[0]
    B, order = hc_lingam(
        mx.array(C2_global), mx.array(C3_global), n_vars, ridge_alpha
    )
    return B, order
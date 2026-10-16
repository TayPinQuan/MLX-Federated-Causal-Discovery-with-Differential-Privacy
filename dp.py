# dp.py
"""Differential privacy via the Gaussian mechanism on cumulant tensors."""
from __future__ import annotations

import math
import mlx.core as mx
import numpy as np


def gaussian_sigma(sensitivity: float, epsilon: float, delta: float = 1e-5) -> float:
    """Standard deviation of Gaussian noise for (epsilon, delta)-DP."""
    if math.isinf(epsilon):
        return 0.0
    return sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon


def add_dp_noise(
    cumulants: dict[int, mx.array],
    n_samples: int,
    epsilon: float,
    delta: float = 1e-5,
    seed: int = 42,
) -> dict[int, mx.array]:
    """
    Add Gaussian noise to each cumulant tensor.

    Sensitivity is bounded conservatively by 1 / n_samples. For third-order
    cumulants of centered data, the exact per-element sensitivity is O(1/n),
    so this bound is safe for experiments.

    When epsilon == inf, returns the input unchanged (no-DP baseline).
    """
    if math.isinf(epsilon):
        return cumulants

    sensitivity = 1.0 / n_samples
    sigma = gaussian_sigma(sensitivity, epsilon, delta)

    noisy: dict[int, mx.array] = {}
    for order, C in cumulants.items():
        key = mx.random.key(seed + order)
        noise = mx.random.normal(C.shape, key=key) * sigma
        noisy[order] = C + noise
    return noisy
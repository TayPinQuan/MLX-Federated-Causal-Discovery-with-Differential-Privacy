# client.py
"""Flower client: local cumulant computation + differential privacy."""
from __future__ import annotations

import numpy as np
import mlx.core as mx
import flwr as fl

from cumulant import local_cumulant_tensor
from dp import add_dp_noise


class FederatedLingamClient(fl.client.NumPyClient):
    def __init__(
        self,
        data: np.ndarray,
        epsilon: float,
        delta: float = 1e-5,
        standardize: bool = False,
        seed: int = 42,
    ):
        self.data = mx.array(np.asarray(data, dtype=np.float32))
        self.epsilon = epsilon
        self.delta = delta
        self.standardize = standardize
        self.seed = seed

    def get_parameters(self, config):
        return []

    def fit(self, parameters, config):
        cumulants, n = local_cumulant_tensor(
            self.data, standardize=self.standardize
        )
        noisy = add_dp_noise(
            cumulants, n, self.epsilon, self.delta, seed=self.seed
        )
        # Return as NumPy arrays for Flower transport.
        return (
            [np.array(noisy[2]), np.array(noisy[3])],
            n,
            {},
        )

    def evaluate(self, parameters, config):
        return 0.0, 0, {}
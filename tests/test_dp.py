# tests/test_dp.py
import math
import numpy as np
import mlx.core as mx
from cumulant import local_cumulant_tensor
from dp import add_dp_noise

def test_no_noise_when_epsilon_inf():
    rng = np.random.default_rng(0)
    X = mx.array(rng.standard_normal((200, 5)).astype(np.float32))
    c, n = local_cumulant_tensor(X)
    noisy = add_dp_noise(c, n, epsilon=float("inf"))
    np.testing.assert_allclose(np.array(noisy[2]), np.array(c[2]))

def test_noise_scale_decreases_with_epsilon():
    """Larger epsilon -> smaller noise -> smaller distance from clean."""
    rng = np.random.default_rng(1)
    X = mx.array(rng.standard_normal((500, 5)).astype(np.float32))
    c, n = local_cumulant_tensor(X)

    d_small_eps = np.linalg.norm(np.array(add_dp_noise(c, n, 0.5)[2]) - np.array(c[2]))
    d_large_eps = np.linalg.norm(np.array(add_dp_noise(c, n, 10.0)[2]) - np.array(c[2]))
    assert d_small_eps > d_large_eps
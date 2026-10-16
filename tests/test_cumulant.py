# tests/test_cumulant.py
import numpy as np
import mlx.core as mx
from cumulant import local_cumulant_tensor

def test_c2_matches_numpy_covariance():
    rng = np.random.default_rng(0)
    X_np = rng.standard_normal((500, 6)).astype(np.float32)
    X_mx = mx.array(X_np)

    cumulants, n = local_cumulant_tensor(X_mx, standardize=False)
    C2_mlx = np.array(cumulants[2])

    X_c = X_np - X_np.mean(axis=0)
    C2_ref = X_c.T @ X_c / 500

    np.testing.assert_allclose(C2_mlx, C2_ref, atol=1e-5)
    assert n == 500

def test_c3_symmetry():
    """C3[i,j,k] should be symmetric under permutation of indices."""
    rng = np.random.default_rng(1)
    X = mx.array(rng.standard_normal((300, 4)).astype(np.float32))
    cumulants, _ = local_cumulant_tensor(X)
    C3 = np.array(cumulants[3])

    for i in range(4):
        for j in range(4):
            for k in range(4):
                assert abs(C3[i,j,k] - C3[j,i,k]) < 1e-4
                assert abs(C3[i,j,k] - C3[i,k,j]) < 1e-4
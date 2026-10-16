import numpy as np
import mlx.core as mx
from cumulant import local_cumulant_tensor
from hc_lingam import hc_lingam_order


def test_order_on_chain():
    """For a chain 0 -> 1 -> 2 with asymmetric noise, order must be [0,1,2]."""
    rng = np.random.default_rng(0)
    n = 5000
    # Asymmetric (exponential, centered) noise
    e0 = rng.exponential(1.0, n) - 1.0
    e1 = rng.exponential(1.0, n) - 1.0
    e2 = rng.exponential(1.0, n) - 1.0
    x0 = e0
    x1 = 0.8 * x0 + 0.3 * e1
    x2 = 0.7 * x1 + 0.3 * e2
    X = mx.array(np.stack([x0, x1, x2], axis=1).astype(np.float32))

    c, _ = local_cumulant_tensor(X)
    order = hc_lingam_order(c[3], 3, C2=c[2])
    assert order == [0, 1, 2], f"Expected [0,1,2], got {order}"
# scripts/verify_inf.py
import numpy as np
from data.sachs import load_sachs_federated, _load_sachs_raw
from cumulant import local_cumulant_tensor
from hc_lingam import hc_lingam, threshold_adjacency
from evaluation import compute_shd, compute_f1, ancestral_accuracy
from federated import run_federated_lingam
import mlx.core as mx

# --- Centralized reference ---
X, true_adj = _load_sachs_raw()
c, n = local_cumulant_tensor(mx.array(X.astype(np.float32)))
B_central, order_central = hc_lingam(c[2], c[3], X.shape[1])
B_central = threshold_adjacency(B_central, tau=0.1)

# --- Federated with epsilon = inf ---
client_data = load_sachs_federated(n_clients=2, random_state=42)
B_fed, order_fed = run_federated_lingam(client_data, epsilon=float("inf"))
B_fed = threshold_adjacency(B_fed, tau=0.1)

# --- Compare ---
print(f"Centralized: SHD={compute_shd(B_central, true_adj)}  "
      f"F1={compute_f1(B_central, true_adj):.3f}  "
      f"AA={ancestral_accuracy(true_adj, order_central):.3f}")
print(f"Federated:   SHD={compute_shd(B_fed, true_adj)}  "
      f"F1={compute_f1(B_fed, true_adj):.3f}  "
      f"AA={ancestral_accuracy(true_adj, order_fed):.3f}")

# Expected: SHD difference <= 2, AA difference <= 0.1
assert abs(compute_shd(B_central, true_adj) - compute_shd(B_fed, true_adj)) <= 2, \
    "Federated aggregation diverges too much from centralized"
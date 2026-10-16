# scripts/verify_federated_synthetic.py
"""Verify federated HC-LiNGAM on synthetic data at epsilon = inf.

Tests that federated aggregation exactly recovers the centralized
cumulants when no DP noise is added.
"""
import numpy as np
import mlx.core as mx

from data.synthetic_lingam import (
    generate_lingam_dag, generate_lingam_data, federated_split,
)
from cumulant import local_cumulant_tensor
from hc_lingam import hc_lingam, threshold_adjacency
from evaluation import compute_shd, compute_f1, ancestral_accuracy, direction_accuracy
from federated import run_federated_lingam


def run_centralized(X):
    c, _ = local_cumulant_tensor(mx.array(X), standardize=False)
    B, order = hc_lingam(c[2], c[3], X.shape[1], ridge_alpha=1e-3)
    return threshold_adjacency(B, tau=0.1), order


def main():
    print("=" * 72)
    print("Federated HC-LiNGAM verification at epsilon = inf")
    print("=" * 72)

    n_vars = 6
    n_samples = 5000
    n_clients = 2

    print(f"\nSetup: {n_vars} vars, {n_samples} samples, "
          f"{n_clients} clients, exponential noise")

    for seed in [0, 1, 2, 3, 4]:
        B_true, _ = generate_lingam_dag(n_vars, 0.3, random_state=seed)
        X = generate_lingam_data(B_true, n_samples,
                                 noise_dist="exponential",
                                 random_state=seed)
        A_true = (B_true != 0).astype(int)

        # Centralized
        B_c, order_c = run_centralized(X)
        aa_c = ancestral_accuracy(A_true, order_c)
        shd_c = compute_shd(B_c, A_true)
        f1_c = compute_f1(B_c, A_true)

        # Federated (no DP)
        client_data = federated_split(X, n_clients, random_state=seed)
        B_f, order_f = run_federated_lingam(
            client_data, epsilon=float("inf"), standardize=False,
        )
        B_f = threshold_adjacency(B_f, tau=0.1)
        aa_f = ancestral_accuracy(A_true, order_f)
        shd_f = compute_shd(B_f, A_true)
        f1_f = compute_f1(B_f, A_true)

        print(f"  seed={seed}: "
              f"central AA={aa_c:.3f} SHD={shd_c} | "
              f"federated AA={aa_f:.3f} SHD={shd_f} | "
              f"ΔAA={abs(aa_c - aa_f):.3f} ΔSHD={abs(shd_c - shd_f)}")


if __name__ == "__main__":
    main()
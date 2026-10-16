# experiments/run_heterogeneity_scan.py
"""Heterogeneity scan: how does federated HC-LiNGAM degrade when
clients share structure but have different coefficients?

Compares federated (epsilon = inf and 0.1) against a pooled-data
centralized baseline.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import mlx.core as mx

from data.synthetic_lingam import (
    generate_lingam_dag, generate_lingam_data,
    generate_heterogeneous_clients,
)
from cumulant import local_cumulant_tensor
from hc_lingam import hc_lingam, threshold_adjacency
from evaluation import (
    compute_shd, compute_f1, ancestral_accuracy, direction_accuracy,
)
from federated import run_federated_lingam


HETEROGENEITIES = [0.0, 0.2, 0.5, 1.0]
EPSILONS = [float("inf"), 0.1]
SEEDS = list(range(10))
N_VARS = 6
N_SAMPLES_PER_CLIENT = 500    # total = 2 * 500 = 1000 (same as before)
N_CLIENTS = 2
TAU = 0.1
RIDGE_ALPHA = 1e-3


def run_centralized_pooled(client_data, A_true):
    """Centralized on pooled data: upper bound on what federation could achieve."""
    X = np.vstack(client_data)
    c, _ = local_cumulant_tensor(mx.array(X.astype(np.float32)), standardize=False)
    B, order = hc_lingam(c[2], c[3], X.shape[1], ridge_alpha=RIDGE_ALPHA)
    B_t = threshold_adjacency(B, tau=TAU)
    return {
        "shd": compute_shd(B_t, A_true),
        "f1": compute_f1(B_t, A_true),
        "aa": ancestral_accuracy(A_true, order),
        "dir_acc": direction_accuracy(B_t, A_true),
    }


def main(out_dir: str = "results"):
    os.makedirs(out_dir, exist_ok=True)
    records = []

    for h in HETEROGENEITIES:
        for seed in SEEDS:
            client_data, A_true, _ = generate_heterogeneous_clients(
                n_vars=N_VARS,
                n_samples_per_client=N_SAMPLES_PER_CLIENT,
                n_clients=N_CLIENTS,
                heterogeneity=h,
                edge_density=0.3,
                noise_dist="exponential",
                random_state=seed + 1000,
            )

            # Centralized on pooled data (upper bound)
            rec = run_centralized_pooled(client_data, A_true)
            rec.update({"heterogeneity": h, "seed": seed, "method": "pooled"})
            records.append(rec)

            # Federated at each epsilon
            for eps in EPSILONS:
                B, order = run_federated_lingam(
                    client_data, epsilon=eps, delta=1e-5,
                    standardize=False, ridge_alpha=RIDGE_ALPHA, seed=seed,
                )
                B_t = threshold_adjacency(B, tau=TAU)
                eps_label = "inf" if np.isinf(eps) else str(eps)
                records.append({
                    "heterogeneity": h,
                    "seed": seed,
                    "method": f"fed_eps_{eps_label}",
                    "shd": compute_shd(B_t, A_true),
                    "f1": compute_f1(B_t, A_true),
                    "aa": ancestral_accuracy(A_true, order),
                    "dir_acc": direction_accuracy(B_t, A_true),
                })

    df = pd.DataFrame(records)
    csv_path = os.path.join(out_dir, "heterogeneity_scan.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved to {csv_path}\n")

    print("=" * 92)
    print(f"{'method':<14} | {'h':>4} | {'SHD':>10} | {'F1':>10} | "
          f"{'AA':>10} | {'DirAcc':>10}")
    print("-" * 92)
    for method in ["pooled", "fed_eps_inf", "fed_eps_0.1"]:
        for h in HETEROGENEITIES:
            sub = df[(df["method"] == method) & (df["heterogeneity"] == h)]
            print(
                f"{method:<14} | {h:>4.1f} | "
                f"{sub['shd'].mean():>4.2f}±{sub['shd'].std():<4.2f} | "
                f"{sub['f1'].mean():>5.3f}±{sub['f1'].std():<4.3f} | "
                f"{sub['aa'].mean():>5.3f}±{sub['aa'].std():<4.3f} | "
                f"{sub['dir_acc'].mean():>5.3f}±{sub['dir_acc'].std():<4.3f}"
            )
        print("-" * 92)


if __name__ == "__main__":
    main()
# experiments/run_client_count_ablation.py
"""Client count ablation: how does the number of clients affect
federated causal discovery, with and without DP?

Theoretical prediction:
  - epsilon = inf: invariant (cumulant additivity)
  - epsilon = 0.1: aggregated noise scales as sqrt(K) / (N * eps),
    so more clients -> more noise -> worse recovery.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from data.synthetic_lingam import (
    generate_lingam_dag, generate_lingam_data, federated_split,
)
from hc_lingam import threshold_adjacency
from evaluation import (
    compute_shd, compute_f1, ancestral_accuracy, direction_accuracy,
)
from federated import run_federated_lingam


CLIENT_COUNTS = [1, 2, 4, 8]
EPSILONS = [float("inf"), 0.1]
SEEDS = list(range(10))
N_VARS = 6
N_SAMPLES = 1000
TAU = 0.1
RIDGE_ALPHA = 1e-3
DELTA = 1e-5


def main(out_dir: str = "results"):
    os.makedirs(out_dir, exist_ok=True)
    records = []

    for seed in SEEDS:
        B_true, _ = generate_lingam_dag(
            N_VARS, 0.3, random_state=seed + 1000
        )
        X = generate_lingam_data(
            B_true, N_SAMPLES, noise_dist="exponential",
            random_state=seed + 1000,
        )
        A_true = (B_true != 0).astype(int)

        for n_clients in CLIENT_COUNTS:
            client_data = federated_split(X, n_clients, random_state=seed)
            for eps in EPSILONS:
                B, order = run_federated_lingam(
                    client_data, epsilon=eps, delta=DELTA,
                    standardize=False, ridge_alpha=RIDGE_ALPHA, seed=seed,
                )
                B_t = threshold_adjacency(B, tau=TAU)
                records.append({
                    "seed": seed,
                    "n_clients": n_clients,
                    "epsilon": eps,
                    "shd": compute_shd(B_t, A_true),
                    "f1": compute_f1(B_t, A_true),
                    "aa": ancestral_accuracy(A_true, order),
                    "dir_acc": direction_accuracy(B_t, A_true),
                })

    df = pd.DataFrame(records)
    csv_path = os.path.join(out_dir, "client_count_ablation.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved to {csv_path}\n")

    print("=" * 88)
    print(f"{'epsilon':>8} | {'K':>3} | {'SHD':>10} | {'F1':>10} | "
          f"{'AA':>10} | {'DirAcc':>10}")
    print("-" * 88)
    for eps in EPSILONS:
        eps_label = "inf" if np.isinf(eps) else f"{eps}"
        for k in CLIENT_COUNTS:
            sub = df[(df["epsilon"] == eps) & (df["n_clients"] == k)]
            print(
                f"{eps_label:>8} | {k:>3} | "
                f"{sub['shd'].mean():>4.2f}±{sub['shd'].std():<4.2f} | "
                f"{sub['f1'].mean():>5.3f}±{sub['f1'].std():<4.3f} | "
                f"{sub['aa'].mean():>5.3f}±{sub['aa'].std():<4.3f} | "
                f"{sub['dir_acc'].mean():>5.3f}±{sub['dir_acc'].std():<4.3f}"
            )
        print("-" * 88)


if __name__ == "__main__":
    main()
# experiments/run_standardization_ablation.py
"""Standardization ablation: test whether marginal standardization
destroys the causal scale signal that cumulant-based LiNGAM relies on.

FedRCD warns that standardizing each variable to unit variance removes the
scale differences that encode causal direction, causing HC-LiNGAM to
degenerate to near-random ordering.
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


EPSILONS = [0.01, 0.1, 1.0, float("inf")]
STANDARDIZE_OPTIONS = [False, True]  # A1: False, A2: True
SEEDS = list(range(10))
N_VARS = 6
N_SAMPLES = 1000
N_CLIENTS = 2
TAU = 0.1


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
        client_data = federated_split(X, N_CLIENTS, random_state=seed)

        for std in STANDARDIZE_OPTIONS:
            for eps in EPSILONS:
                B, order = run_federated_lingam(
                    client_data, epsilon=eps, delta=1e-5,
                    standardize=std, ridge_alpha=1e-3, seed=seed,
                )
                B_t = threshold_adjacency(B, tau=TAU)
                records.append({
                    "seed": seed,
                    "epsilon": eps,
                    "standardize": std,
                    "shd": compute_shd(B_t, A_true),
                    "f1": compute_f1(B_t, A_true),
                    "aa": ancestral_accuracy(A_true, order),
                    "dir_acc": direction_accuracy(B_t, A_true),
                })

    df = pd.DataFrame(records)
    csv_path = os.path.join(out_dir, "standardization_ablation.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved to {csv_path}\n")

    print("=" * 80)
    print(f"{'mode':<10} | {'epsilon':>8} | {'SHD':>10} | {'F1':>10} | "
          f"{'AA':>10} | {'DirAcc':>10}")
    print("-" * 80)
    for std in STANDARDIZE_OPTIONS:
        tag = "A2_std" if std else "A1_raw"
        for eps in EPSILONS:
            sub = df[(df["standardize"] == std) & (df["epsilon"] == eps)]
            eps_label = "inf" if np.isinf(eps) else f"{eps}"
            print(
                f"{tag:<10} | {eps_label:>8} | "
                f"{sub['shd'].mean():>4.2f}±{sub['shd'].std():<4.2f} | "
                f"{sub['f1'].mean():>5.3f}±{sub['f1'].std():<4.3f} | "
                f"{sub['aa'].mean():>5.3f}±{sub['aa'].std():<4.3f} | "
                f"{sub['dir_acc'].mean():>5.3f}±{sub['dir_acc'].std():<4.3f}"
            )
        print("-" * 80)


if __name__ == "__main__":
    main()
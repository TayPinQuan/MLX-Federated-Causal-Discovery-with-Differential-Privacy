# experiments/run_cumulant_order_ablation.py
"""Cumulant order ablation: verify that third-order cumulants are necessary
for causal direction identification.

If C2 alone (covariance) could determine direction, LiNGAM would not need
higher-order statistics. This ablation confirms the theoretical requirement.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import mlx.core as mx

from data.synthetic_lingam import (
    generate_lingam_dag, generate_lingam_data, federated_split,
)
from cumulant import local_cumulant_tensor
from hc_lingam import hc_lingam_adjacency, threshold_adjacency
from evaluation import (
    compute_shd, compute_f1, ancestral_accuracy, direction_accuracy,
)
from federated import run_federated_lingam


SEEDS = list(range(10))
N_VARS = 6
N_SAMPLES = 1000
N_CLIENTS = 2
TAU = 0.1
RIDGE_ALPHA = 1e-3


def _random_order(n_vars, seed):
    rng = np.random.default_rng(seed)
    order = list(range(n_vars))
    rng.shuffle(order)
    return order


def main(out_dir: str = "results"):
    os.makedirs(out_dir, exist_ok=True)
    records = []

    for seed in SEEDS:
        B_true, true_order = generate_lingam_dag(
            N_VARS, 0.3, random_state=seed + 1000
        )
        X = generate_lingam_data(
            B_true, N_SAMPLES, noise_dist="exponential",
            random_state=seed + 1000,
        )
        A_true = (B_true != 0).astype(int)
        client_data = federated_split(X, N_CLIENTS, random_state=seed)

        # O1: C2 only -> random order (C2 is symmetric, no direction info)
        order_o1 = _random_order(N_VARS, seed)
        # Aggregate C2 only
        total_n = 0
        C2_sum = None
        for data in client_data:
            c, n = local_cumulant_tensor(
                mx.array(data.astype(np.float32)), standardize=False
            )
            C2_sum = c[2] * n if C2_sum is None else C2_sum + c[2] * n
            total_n += n
        C2_avg = mx.array(np.array(C2_sum) / total_n)
        B_o1 = hc_lingam_adjacency(C2_avg, order_o1, N_VARS, RIDGE_ALPHA)
        B_o1 = threshold_adjacency(B_o1, tau=TAU)
        records.append({
            "seed": seed,
            "mode": "O1_C2_only",
            "shd": compute_shd(B_o1, A_true),
            "f1": compute_f1(B_o1, A_true),
            "aa": ancestral_accuracy(A_true, order_o1),
            "dir_acc": direction_accuracy(B_o1, A_true),
        })

        # O2: C2 + C3 -> full pipeline
        B_o2, order_o2 = run_federated_lingam(
            client_data, epsilon=float("inf"), delta=1e-5,
            standardize=False, ridge_alpha=RIDGE_ALPHA, seed=seed,
        )
        B_o2 = threshold_adjacency(B_o2, tau=TAU)
        records.append({
            "seed": seed,
            "mode": "O2_C2_C3",
            "shd": compute_shd(B_o2, A_true),
            "f1": compute_f1(B_o2, A_true),
            "aa": ancestral_accuracy(A_true, order_o2),
            "dir_acc": direction_accuracy(B_o2, A_true),
        })

    df = pd.DataFrame(records)
    csv_path = os.path.join(out_dir, "cumulant_order_ablation.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved to {csv_path}\n")

    print("=" * 80)
    print(f"{'mode':<12} | {'SHD':>10} | {'F1':>10} | {'AA':>10} | {'DirAcc':>10}")
    print("-" * 80)
    for mode in ["O1_C2_only", "O2_C2_C3"]:
        sub = df[df["mode"] == mode]
        print(
            f"{mode:<12} | "
            f"{sub['shd'].mean():>4.2f}±{sub['shd'].std():<4.2f} | "
            f"{sub['f1'].mean():>5.3f}±{sub['f1'].std():<4.3f} | "
            f"{sub['aa'].mean():>5.3f}±{sub['aa'].std():<4.3f} | "
            f"{sub['dir_acc'].mean():>5.3f}±{sub['dir_acc'].std():<4.3f}"
        )


if __name__ == "__main__":
    main()
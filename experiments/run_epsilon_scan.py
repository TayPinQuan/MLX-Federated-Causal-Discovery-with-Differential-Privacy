# experiments/run_epsilon_scan.py
"""Privacy-accuracy trade-off for federated HC-LiNGAM.

Sweeps the privacy budget epsilon across a logarithmic grid and
records graph recovery quality. The primary output is a threshold
epsilon_min below which causal direction recovery degrades to chance.
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


# --- Experiment configuration ---
EPSILONS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0, float("inf")]
SEEDS = list(range(10))
N_VARS = 6
N_SAMPLES = 1000       
N_CLIENTS = 2
NOISE_DIST = "exponential"
EDGE_DENSITY = 0.3
TAU = 0.1
RIDGE_ALPHA = 1e-3
DELTA = 1e-5


def run_one(seed: int, epsilon: float) -> dict:
    B_true, _ = generate_lingam_dag(
        N_VARS, EDGE_DENSITY, random_state=seed
    )
    X = generate_lingam_data(
        B_true, N_SAMPLES, noise_dist=NOISE_DIST, random_state=seed
    )
    A_true = (B_true != 0).astype(int)

    client_data = federated_split(X, N_CLIENTS, random_state=seed)
    B_est, order_est = run_federated_lingam(
        client_data,
        epsilon=epsilon,
        delta=DELTA,
        standardize=False,
        ridge_alpha=RIDGE_ALPHA,
        seed=seed,
    )
    B_t = threshold_adjacency(B_est, tau=TAU)

    return {
        "seed": seed,
        "epsilon": epsilon,
        "shd": compute_shd(B_t, A_true),
        "f1": compute_f1(B_t, A_true),
        "aa": ancestral_accuracy(A_true, order_est),
        "dir_acc": direction_accuracy(B_t, A_true),
    }


def main(out_dir: str = "results"):
    os.makedirs(out_dir, exist_ok=True)
    records = []
    total = len(EPSILONS) * len(SEEDS)
    done = 0

    for epsilon in EPSILONS:
        eps_label = "inf" if np.isinf(epsilon) else f"{epsilon}"
        for seed in SEEDS:
            rec = run_one(seed, epsilon)
            records.append(rec)
            done += 1
            print(
                f"[{done:>3}/{total}] eps={eps_label:<5} seed={seed} "
                f"| SHD={rec['shd']:>2} F1={rec['f1']:.3f} "
                f"AA={rec['aa']:.3f} DirAcc={rec['dir_acc']:.3f}"
            )

    df = pd.DataFrame(records)
    csv_path = os.path.join(out_dir, "epsilon_scan.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved raw results to {csv_path}")

    # Summary table
    print("\n" + "=" * 78)
    print(f"{'epsilon':>8} | {'SHD':>10} | {'F1':>10} | {'AA':>10} | {'DirAcc':>10}")
    print("-" * 78)
    for epsilon in EPSILONS:
        sub = df[df["epsilon"] == epsilon]
        eps_label = "inf" if np.isinf(epsilon) else f"{epsilon}"
        print(
            f"{eps_label:>8} | "
            f"{sub['shd'].mean():>4.2f}±{sub['shd'].std():<4.2f} | "
            f"{sub['f1'].mean():>5.3f}±{sub['f1'].std():<4.3f} | "
            f"{sub['aa'].mean():>5.3f}±{sub['aa'].std():<4.3f} | "
            f"{sub['dir_acc'].mean():>5.3f}±{sub['dir_acc'].std():<4.3f}"
        )


if __name__ == "__main__":
    main()
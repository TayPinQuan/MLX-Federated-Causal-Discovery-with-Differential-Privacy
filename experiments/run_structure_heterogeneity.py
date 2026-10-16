# experiments/run_structure_heterogeneity.py
"""Structure heterogeneity scan.

Clients share a base DAG but have client-specific edges. We measure how
well federated aggregation recovers:
  - shared edges S_shared (present in ALL clients)
  - specific edges S_specific (present in SOME but not all)
  - union edges S_union = S_shared ∪ S_specific
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from data.synthetic_lingam import generate_structurally_heterogeneous_clients
from hc_lingam import threshold_adjacency
from evaluation import ancestral_accuracy, compute_f1
from federated import run_federated_lingam


PERTURB_PROBS = [0.0, 0.2, 0.5, 0.8]
EPSILONS = [float("inf"), 0.1]
SEEDS = list(range(10))
N_VARS = 6
N_SAMPLES_PER_CLIENT = 500
N_CLIENTS = 2
TAU = 0.1
RIDGE_ALPHA = 1e-3


def recall_on_subset(B_est: np.ndarray, target_adj: np.ndarray) -> float:
    """Fraction of target edges that appear in B_est."""
    est = (np.abs(B_est) > 0).astype(int)
    target = (target_adj > 0).astype(int)
    np.fill_diagonal(est, 0)
    np.fill_diagonal(target, 0)
    n_target = int(target.sum())
    if n_target == 0:
        return np.nan
    recovered = int(((est == 1) & (target == 1)).sum())
    return recovered / n_target


def main(out_dir: str = "results"):
    os.makedirs(out_dir, exist_ok=True)
    records = []

    for p in PERTURB_PROBS:
        for seed in SEEDS:
            client_data, meta = generate_structurally_heterogeneous_clients(
                n_vars=N_VARS,
                n_samples_per_client=N_SAMPLES_PER_CLIENT,
                n_clients=N_CLIENTS,
                perturb_prob=p,
                edge_density=0.3,
                noise_dist="exponential",
                random_state=seed + 1000,
            )
            shared = meta["shared"]
            specific = meta["specific"]
            union = meta["union"]

            for eps in EPSILONS:
                B, order = run_federated_lingam(
                    client_data, epsilon=eps, delta=1e-5,
                    standardize=False, ridge_alpha=RIDGE_ALPHA, seed=seed,
                )
                B_t = threshold_adjacency(B, tau=TAU)
                eps_label = "inf" if np.isinf(eps) else str(eps)

                records.append({
                    "perturb_prob": p,
                    "seed": seed,
                    "epsilon": eps_label,
                    "n_shared": int(shared.sum()),
                    "n_specific": int(specific.sum()),
                    "n_union": int(union.sum()),
                    "n_est": int((np.abs(B_t) > 0).sum()),
                    "recall_shared": recall_on_subset(B_t, shared),
                    "recall_specific": recall_on_subset(B_t, specific),
                    "recall_union": recall_on_subset(B_t, union),
                    "f1_union": compute_f1(B_t, union),
                    "aa_union": ancestral_accuracy(union, order),
                })

    df = pd.DataFrame(records)
    csv_path = os.path.join(out_dir, "structure_heterogeneity.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved to {csv_path}\n")

    print("=" * 100)
    print(f"{'eps':>6} | {'p':>4} | {'#shared':>7} | {'#spec':>6} | {'#est':>5} | "
          f"{'R_shared':>8} | {'R_specific':>10} | {'F1_union':>8} | {'AA':>6}")
    print("-" * 100)
    for eps_label in ["inf", "0.1"]:
        for p in PERTURB_PROBS:
            sub = df[(df["epsilon"] == eps_label) & (df["perturb_prob"] == p)]
            print(
                f"{eps_label:>6} | {p:>4.1f} | "
                f"{sub['n_shared'].mean():>7.1f} | "
                f"{sub['n_specific'].mean():>6.1f} | "
                f"{sub['n_est'].mean():>5.1f} | "
                f"{sub['recall_shared'].mean():>8.3f} | "
                f"{sub['recall_specific'].mean():>10.3f} | "
                f"{sub['f1_union'].mean():>8.3f} | "
                f"{sub['aa_union'].mean():>6.3f}"
            )
        print("-" * 100)


if __name__ == "__main__":
    main()
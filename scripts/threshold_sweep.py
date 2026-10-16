# scripts/threshold_sweep.py
"""Sweep the edge threshold tau to check whether DP-induced
false positives can be filtered out."""
import numpy as np
import mlx.core as mx

from data.synthetic_lingam import (
    generate_lingam_dag, generate_lingam_data, federated_split,
)
from hc_lingam import threshold_adjacency
from evaluation import compute_shd, compute_f1
from federated import run_federated_lingam

SEEDS = list(range(10))
TAUS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]


def main():
    results = {("inf", t): [] for t in TAUS}
    results.update({("0.1", t): [] for t in TAUS})

    for seed in SEEDS:
        B_true, _ = generate_lingam_dag(6, 0.3, random_state=seed + 1000)
        X = generate_lingam_data(B_true, 1000, noise_dist="exponential",
                                  random_state=seed + 1000)
        A_true = (B_true != 0).astype(int)
        client_data = federated_split(X, 2, random_state=seed)

        for eps, tag in [(float("inf"), "inf"), (0.1, "0.1")]:
            B, _ = run_federated_lingam(
                client_data, epsilon=eps, delta=1e-5,
                standardize=False, ridge_alpha=1e-3, seed=seed,
            )
            for tau in TAUS:
                B_t = threshold_adjacency(B, tau=tau)
                results[(tag, tau)].append({
                    "shd": compute_shd(B_t, A_true),
                    "f1": compute_f1(B_t, A_true),
                    "n_edges": int((np.abs(B_t) > 0).sum()),
                })

    print(f"{'eps':<6} | {'tau':>5} | {'SHD':>10} | {'F1':>10} | {'edges':>8}")
    print("-" * 55)
    for tag in ["inf", "0.1"]:
        for tau in TAUS:
            recs = results[(tag, tau)]
            shd = np.mean([r["shd"] for r in recs])
            f1 = np.mean([r["f1"] for r in recs])
            ne = np.mean([r["n_edges"] for r in recs])
            print(f"{tag:<6} | {tau:>5.2f} | {shd:>4.2f}±{np.std([r['shd'] for r in recs]):<4.2f} | "
                  f"{f1:>5.3f}±{np.std([r['f1'] for r in recs]):<4.3f} | {ne:>8.1f}")
        print("-" * 55)


if __name__ == "__main__":
    main()
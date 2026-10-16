# scripts/threshold_sweep_structure.py
"""Threshold sweep for structure heterogeneity.

Check whether the over-densification under structure divergence can be
mitigated by raising the edge threshold tau.
"""
import numpy as np
import mlx.core as mx

from data.synthetic_lingam import generate_structurally_heterogeneous_clients
from hc_lingam import threshold_adjacency
from evaluation import compute_f1
from federated import run_federated_lingam


PERTURB_PROBS = [0.2, 0.5, 0.8]
TAUS = [0.1, 0.2, 0.3, 0.5, 0.8]
SEEDS = list(range(10))


def main():
    results = {}
    for p in PERTURB_PROBS:
        for tau in TAUS:
            results[(p, tau)] = []

    for seed in SEEDS:
        for p in PERTURB_PROBS:
            client_data, meta = generate_structurally_heterogeneous_clients(
                n_vars=6, n_samples_per_client=500, n_clients=2,
                perturb_prob=p, edge_density=0.3,
                noise_dist="exponential", random_state=seed + 1000,
            )
            union = meta["union"]
            B, _ = run_federated_lingam(
                client_data, epsilon=float("inf"), delta=1e-5,
                standardize=False, ridge_alpha=1e-3, seed=seed,
            )
            for tau in TAUS:
                B_t = threshold_adjacency(B, tau=tau)
                results[(p, tau)].append({
                    "f1": compute_f1(B_t, union),
                    "n_est": int((np.abs(B_t) > 0).sum()),
                })

    print(f"{'p':>5} | {'tau':>5} | {'F1':>10} | {'#est':>6}")
    print("-" * 40)
    for p in PERTURB_PROBS:
        for tau in TAUS:
            recs = results[(p, tau)]
            f1 = np.mean([r["f1"] for r in recs])
            ne = np.mean([r["n_est"] for r in recs])
            print(f"{p:>5.1f} | {tau:>5.2f} | {f1:>5.3f}±{np.std([r['f1'] for r in recs]):<4.3f} | {ne:>6.1f}")
        print("-" * 40)


if __name__ == "__main__":
    main()
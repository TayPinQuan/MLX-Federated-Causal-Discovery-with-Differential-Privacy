# scripts/verify_sachs_centralized.py
"""Verify HC-LiNGAM on centralized Sachs data."""
import numpy as np
import mlx.core as mx

from data.sachs import load_sachs, SACHS_COLUMNS, SACHS_EDGES
from cumulant import local_cumulant_tensor
from hc_lingam import hc_lingam, hc_lingam_order, threshold_adjacency
from evaluation import compute_shd, compute_f1, ancestral_accuracy, direction_accuracy


def main():
    X, true_adj = load_sachs()
    print(f"Sachs: {X.shape[0]} samples, {X.shape[1]} variables")
    print(f"Ground truth: {len(SACHS_EDGES)} edges\n")

    # Compute cumulants (no standardization)
    cumulants, n = local_cumulant_tensor(mx.array(X), standardize=False)
    C2, C3 = cumulants[2], cumulants[3]

    # Run HC-LiNGAM
    B, order = hc_lingam(C2, C3, X.shape[1], ridge_alpha=1e-3)

    # Evaluate at several thresholds
    print(f"{'tau':>6} | {'SHD':>4} | {'F1':>6} | {'DirAcc':>7} | {'#edges':>6}")
    print("-" * 45)
    for tau in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]:
        A = threshold_adjacency(B, tau=tau)
        shd = compute_shd(A, true_adj)
        f1 = compute_f1(A, true_adj)
        da = direction_accuracy(A, true_adj)
        n_edges = int((np.abs(A) > 0).sum())
        print(f"{tau:>6.2f} | {shd:>4} | {f1:>6.3f} | {da:>7.3f} | {n_edges:>6}")

    # Ancestral Accuracy (order only, threshold-independent)
    aa = ancestral_accuracy(true_adj, order)
    print(f"\nEstimated causal order: {order}")
    print(f"Order names: {[SACHS_COLUMNS[i] for i in order]}")
    print(f"Ancestral Accuracy: {aa:.3f}")

    # Print estimated adjacency at tau=0.1
    A = threshold_adjacency(B, tau=0.1)
    print(f"\nEstimated edges at tau=0.1:")
    for i in range(11):
        for j in range(11):
            if A[i, j] != 0:
                print(f"  {SACHS_COLUMNS[i]:>4} -> {SACHS_COLUMNS[j]:<4} "
                      f"(coef={A[i,j]:+.3f})")


if __name__ == "__main__":
    main()
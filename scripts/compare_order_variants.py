# scripts/compare_order_variants.py
"""Compare several causal-order heuristics on Sachs."""
import numpy as np
import mlx.core as mx

from data.sachs import load_sachs, SACHS_COLUMNS
from cumulant import local_cumulant_tensor
from hc_lingam import hc_lingam_adjacency, threshold_adjacency
from evaluation import compute_shd, compute_f1, ancestral_accuracy, direction_accuracy


def order_v1_normalized(C2_np, C3_np, n_vars):
    """Normalized self-skew: |C3[i,j,j]| / |C3[j,j,j]|."""
    eps = 1e-8
    self_skew = np.array([abs(C3_np[i, i, i]) for i in range(n_vars)])
    ev = np.zeros((n_vars, n_vars))
    for i in range(n_vars):
        for j in range(n_vars):
            if i != j:
                ev[j, i] = abs(C3_np[i, j, j]) / (self_skew[j] + eps)
    return _greedy_order(ev, n_vars)


def order_v2_raw(C2_np, C3_np, n_vars):
    """Raw asymmetry: |C3[i,j,j]|."""
    ev = np.zeros((n_vars, n_vars))
    for i in range(n_vars):
        for j in range(n_vars):
            if i != j:
                ev[j, i] = abs(C3_np[i, j, j])
    return _greedy_order(ev, n_vars)


def order_v3_residual_skew(C2_np, C3_np, n_vars):
    """Residual skew direction score."""
    ev = np.zeros((n_vars, n_vars))
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                continue
            s = _pair_score(C2_np, C3_np, i, j)
            ev[j, i] = max(0.0, s)
    return _greedy_order(ev, n_vars)


def order_v4_hybrid(C2_np, C3_np, n_vars):
    """Residual skew + absolute asymmetry (combined)."""
    ev = np.zeros((n_vars, n_vars))
    eps = 1e-8
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                continue
            s_res = max(0.0, _pair_score(C2_np, C3_np, i, j))
            s_asym = abs(C3_np[i, j, j]) / (abs(C3_np[j, j, j]) + eps)
            ev[j, i] = s_res + 0.1 * s_asym
    return _greedy_order(ev, n_vars)


def _pair_score(C2_np, C3_np, i, j):
    var_i, var_j = C2_np[i, i], C2_np[j, j]
    cov_ij = C2_np[i, j]

    b_ij = cov_ij / (var_j + 1e-8)
    E_r1 = (C3_np[i, i, i] - 3 * b_ij * C3_np[i, i, j]
            + 3 * b_ij ** 2 * C3_np[i, j, j] - b_ij ** 3 * C3_np[j, j, j])
    var_r1 = max(var_i - b_ij ** 2 * var_j, 1e-10)
    skew_r1 = abs(E_r1) / (var_r1 ** 1.5)

    b_ji = cov_ij / (var_i + 1e-8)
    E_r2 = (C3_np[j, j, j] - 3 * b_ji * C3_np[i, j, j]
            + 3 * b_ji ** 2 * C3_np[i, i, j] - b_ji ** 3 * C3_np[i, i, i])
    var_r2 = max(var_j - b_ji ** 2 * var_i, 1e-10)
    skew_r2 = abs(E_r2) / (var_r2 ** 1.5)

    return skew_r2 - skew_r1


def _greedy_order(ev, n_vars):
    order = []
    remaining = set(range(n_vars))
    while remaining:
        if len(remaining) == 1:
            order.append(remaining.pop())
            break
        incoming = {i: sum(ev[j, i] for j in remaining if j != i)
                    for i in remaining}
        src = min(remaining, key=lambda i: incoming[i])
        order.append(src)
        remaining.remove(src)
    return order


def evaluate(order, B, true_adj):
    B_t = threshold_adjacency(B, tau=0.1)
    return {
        "AA": ancestral_accuracy(true_adj, order),
        "SHD": compute_shd(B_t, true_adj),
        "F1": compute_f1(B_t, true_adj),
        "DirAcc": direction_accuracy(B_t, true_adj),
    }


def main():
    X, true_adj = load_sachs()
    n_vars = X.shape[1]
    cumulants, _ = local_cumulant_tensor(mx.array(X), standardize=False)
    C2_np = np.array(cumulants[2])
    C3_np = np.array(cumulants[3])

    variants = {
        "v1_normalized": order_v1_normalized,
        "v2_raw": order_v2_raw,
        "v3_residual_skew": order_v3_residual_skew,
        "v4_hybrid": order_v4_hybrid,
    }

    print(f"{'variant':<20} | {'AA':>5} | {'SHD':>4} | {'F1':>6} | {'DirAcc':>6}")
    print("-" * 55)

    for name, fn in variants.items():
        order = fn(C2_np, C3_np, n_vars)
        B = hc_lingam_adjacency(mx.array(C2_np), order, n_vars, ridge_alpha=1e-3)
        m = evaluate(order, B, true_adj)
        print(f"{name:<20} | {m['AA']:>5.3f} | {m['SHD']:>4} | "
              f"{m['F1']:>6.3f} | {m['DirAcc']:>6.3f}")
        print(f"  order: {[SACHS_COLUMNS[i] for i in order]}")


if __name__ == "__main__":
    main()
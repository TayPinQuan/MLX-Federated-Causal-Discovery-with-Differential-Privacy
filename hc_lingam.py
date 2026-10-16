# hc_lingam.py — 完整替换
"""HC-LiNGAM: causal order and adjacency from higher-order cumulants."""
from __future__ import annotations

import numpy as np
import mlx.core as mx


def _pair_direction_score(C2_np: np.ndarray, C3_np: np.ndarray,
                          i: int, j: int) -> float:
    """
    Positive score -> j is a parent of i (j -> i).
    Negative score -> i is a parent of j (i -> j).

    Uses consistency between 2nd- and 3rd-order regression coefficients.
    If j -> i, then x_i = b*x_j + e_i with e_i independent of x_j, so:
        C2[i,j] = b * C2[j,j]
        C3[i,j,j] = b * C3[j,j,j]
    Both equations imply b = C2[i,j]/C2[j,j] = C3[i,j,j]/C3[j,j,j].
    Cross-multiplying avoids division:
        D1 = |C3[i,j,j] * C2[j,j] - C2[i,j] * C3[j,j,j]|
    D1 is small when j -> i. Symmetrically:
        D2 = |C3[i,i,j] * C2[i,i] - C2[i,j] * C3[i,i,i]|
    D2 is small when i -> j. Score = D2 - D1.
    """
    D1 = abs(C3_np[i, j, j] * C2_np[j, j] - C2_np[i, j] * C3_np[j, j, j])
    D2 = abs(C3_np[i, i, j] * C2_np[i, i] - C2_np[i, j] * C3_np[i, i, i])
    return D2 - D1


def hc_lingam_order(
    C3: mx.array,
    n_vars: int,
    C2: mx.array | None = None,
) -> list[int]:
    """
    Determine causal order using the LiNGAM independence criterion.

    Requires C2 for the regression coefficients. If C2 is None, falls back
    to the raw asymmetry heuristic (not recommended).
    """
    if C2 is None:
        C3_np = np.array(C3)
        evidence = np.zeros((n_vars, n_vars))
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:
                    evidence[j, i] = abs(C3_np[i, j, j])
    else:
        C2_np = np.array(C2)
        C3_np = np.array(C3)
        evidence = np.zeros((n_vars, n_vars))
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    continue
                s = _pair_direction_score(C2_np, C3_np, i, j)
                # Keep only positive evidence: j -> i
                evidence[j, i] = max(0.0, s)

    order: list[int] = []
    remaining = set(range(n_vars))
    while remaining:
        if len(remaining) == 1:
            order.append(remaining.pop())
            break
        incoming = {
            i: sum(evidence[j, i] for j in remaining if j != i)
            for i in remaining
        }
        source = min(remaining, key=lambda i: incoming[i])
        order.append(source)
        remaining.remove(source)

    return order


def hc_lingam_adjacency(
    C2: mx.array,
    order: list[int],
    n_vars: int,
    ridge_alpha: float = 1e-3,
) -> np.ndarray:
    """Estimate adjacency given causal order via Ridge regression on C2."""
    C2_np = np.array(C2)
    B = np.zeros((n_vars, n_vars))

    for idx, target in enumerate(order):
        if idx == 0:
            continue
        preds = order[:idx]
        X_pred = C2_np[np.ix_(preds, preds)]
        y_cov = C2_np[np.ix_(preds, [target])].flatten()

        A = X_pred + ridge_alpha * np.eye(len(preds))
        coef = np.linalg.solve(A, y_cov)

        for k, pred in enumerate(preds):
            B[pred, target] = coef[k]

    return B


def hc_lingam(
    C2: mx.array,
    C3: mx.array,
    n_vars: int,
    ridge_alpha: float = 1e-3,
) -> tuple[np.ndarray, list[int]]:
    """Full pipeline: order + adjacency."""
    order = hc_lingam_order(C3, n_vars, C2=C2)
    B = hc_lingam_adjacency(C2, order, n_vars, ridge_alpha)
    return B, order


def threshold_adjacency(B: np.ndarray, tau: float = 0.1) -> np.ndarray:
    A = B.copy()
    A[np.abs(A) < tau] = 0.0
    return A
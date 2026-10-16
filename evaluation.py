# evaluation.py
"""Evaluation metrics for federated causal discovery."""
from __future__ import annotations

import numpy as np


def compute_shd(est_adj: np.ndarray, true_adj: np.ndarray) -> int:
    """Structural Hamming Distance (undirected, ignoring self-loops)."""
    est = (np.abs(est_adj) > 0).astype(int)
    true = (true_adj > 0).astype(int)
    np.fill_diagonal(est, 0)
    np.fill_diagonal(true, 0)
    # Symmetrize for undirected SHD
    est_sym = ((est + est.T) > 0).astype(int)
    true_sym = ((true + true.T) > 0).astype(int)
    return int(np.sum(est_sym != true_sym) // 2)


def compute_f1(est_adj: np.ndarray, true_adj: np.ndarray) -> float:
    """Pairwise F1 of edge recovery."""
    est = (np.abs(est_adj) > 0).astype(int)
    true = (true_adj > 0).astype(int)
    np.fill_diagonal(est, 0)
    np.fill_diagonal(true, 0)

    tp = int(np.sum((est == 1) & (true == 1)))
    fp = int(np.sum((est == 1) & (true == 0)))
    fn = int(np.sum((est == 0) & (true == 1)))

    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# evaluation.py — 替换 ancestral_accuracy
def ancestral_accuracy(true_adj: np.ndarray, est_order: list[int]) -> float:
    """
    Fraction of ancestor pairs (i, j) where i is a true ancestor of j,
    and the estimated order places i before j.

    Only counts pairs with a true ancestor relation. Unrelated pairs are
    ignored because any relative order is a valid topological ordering.
    """
    n = true_adj.shape[0]
    ancestors = _transitive_closure(true_adj)
    pos = {v: i for i, v in enumerate(est_order)}

    correct = 0
    total = 0
    for i in range(n):
        for j in range(n):
            if i == j or ancestors[i, j] != 1:
                continue
            total += 1
            if pos[i] < pos[j]:
                correct += 1
    return correct / total if total > 0 else 1.0


def direction_accuracy(est_adj: np.ndarray, true_adj: np.ndarray) -> float:
    """Among correctly recovered edges, proportion with correct orientation."""
    est = (np.abs(est_adj) > 0).astype(int)
    true = (true_adj > 0).astype(int)
    np.fill_diagonal(est, 0)
    np.fill_diagonal(true, 0)

    correct = 0
    total = 0
    for i in range(true.shape[0]):
        for j in range(true.shape[1]):
            if true[i, j] == 1 and est[i, j] == 1:
                correct += 1
                total += 1
            elif true[i, j] == 1 and est[j, i] == 1:
                total += 1  # reversed
    return correct / total if total > 0 else 0.0


def _transitive_closure(adj: np.ndarray) -> np.ndarray:
    n = adj.shape[0]
    closure = (adj > 0).astype(bool)
    for k in range(n):
        for i in range(n):
            if closure[i, k]:
                closure[i] |= closure[k]
    return closure.astype(int)
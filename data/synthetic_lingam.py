# data/synthetic_lingam.py
"""Synthetic LiNGAM data generator.

Generates data satisfying all LiNGAM assumptions:
- Linear mechanisms
- Non-Gaussian noise (uniform, exponential, or laplace)
- Acyclic structure
- No latent confounders
"""
from __future__ import annotations

import numpy as np


def generate_lingam_dag(
    n_vars: int,
    edge_density: float = 0.3,
    weight_range: tuple[float, float] = (0.5, 1.5),
    random_state: int = 42,
) -> tuple[np.ndarray, list[int]]:
    """
    Generate a random DAG in topological order.

    Node indices are topologically sorted by construction: for any edge
    i -> j, we require i < j. This is the standard convention for
    generating random DAGs.

    Returns:
        B: (n_vars, n_vars) weighted adjacency; B[i, j] = weight of i -> j.
        true_order: topological order (which is just [0, 1, ..., n_vars-1]).
    """
    rng = np.random.default_rng(random_state)
    B = np.zeros((n_vars, n_vars))

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if rng.random() < edge_density:
                sign = rng.choice([-1, 1])
                magnitude = rng.uniform(*weight_range)
                B[i, j] = sign * magnitude

    true_order = list(range(n_vars))
    return B, true_order


def generate_lingam_data(
    B: np.ndarray,
    n_samples: int,
    noise_dist: str = "exponential",  # changed default
    noise_scale: float = 1.0,
    random_state: int = 42,
) -> np.ndarray:
    """
    Sample data from a linear non-Gaussian SEM.

    IMPORTANT: Only asymmetric (nonzero-skew) noise distributions are
    compatible with third-order cumulant methods. Uniform and Laplace are
    non-Gaussian but symmetric, so they carry no direction information in
    C3. Use 'exponential', 'gamma', or 'chi2'.
    """
    rng = np.random.default_rng(random_state)
    n_vars = B.shape[0]

    if noise_dist == "exponential":
        E = rng.exponential(noise_scale, size=(n_samples, n_vars))
        E = E - E.mean(axis=0)
    elif noise_dist == "gamma":
        E = rng.gamma(shape=2.0, scale=noise_scale, size=(n_samples, n_vars))
        E = E - E.mean(axis=0)
    elif noise_dist == "chi2":
        E = rng.chisquare(df=3, size=(n_samples, n_vars))
        E = E - E.mean(axis=0)
    elif noise_dist == "uniform_mix":
        # Mixture of two uniforms (Shimizu 2006 style, asymmetric)
        w = rng.random(size=(n_samples, n_vars))
        E1 = rng.uniform(-noise_scale, noise_scale, size=(n_samples, n_vars))
        E2 = rng.uniform(-noise_scale / 3, noise_scale / 3, size=(n_samples, n_vars))
        E = np.where(w < 0.5, E1, E2)
        E = E - E.mean(axis=0)
    else:
        raise ValueError(
            f"Unknown or invalid noise distribution: {noise_dist}. "
            f"Use 'exponential', 'gamma', 'chi2', or 'uniform_mix'."
        )

    I = np.eye(n_vars)
    M = np.linalg.inv(I - B.T)
    X = E @ M.T
    return X.astype(np.float32)


def generate_drifting_lingam_data(
    n_samples: int = 5000,
    n_vars: int = 8,
    drift_at: int = 2500,
    drift_type: str = "both",
    edge_density: float = 0.3,
    noise_dist: str = "uniform",
    random_state: int = 42,
) -> tuple[np.ndarray, dict]:
    """
    Generate synthetic LiNGAM data with a structural drift.

    Returns:
        X: (n_samples, n_vars) concatenated pre/post drift data.
        meta: dict with B_pre, B_post, true_order, drift_at.
    """
    B_pre, order = generate_lingam_dag(n_vars, edge_density, random_state=random_state)
    B_post = B_pre.copy()

    rng = np.random.default_rng(random_state + 1)
    if drift_type in ("structure", "both"):
        # Remove a few edges and add a few new ones
        nonzeros = np.argwhere(B_pre != 0)
        if len(nonzeros) > 2:
            for idx in rng.choice(len(nonzeros), size=2, replace=False):
                B_post[tuple(nonzeros[idx])] = 0.0
        # Add 2 new edges (keeping acyclicity by only adding i -> j with i < j)
        for _ in range(2):
            i, j = sorted(rng.choice(n_vars, size=2, replace=False))
            B_post[i, j] = rng.uniform(0.5, 1.5) * rng.choice([-1, 1])

    if drift_type in ("coefficients", "both"):
        # Perturb existing edge weights
        nz = np.argwhere(B_post != 0)
        for idx in rng.choice(len(nz), size=min(3, len(nz)), replace=False):
            i, j = nz[idx]
            B_post[i, j] *= rng.uniform(0.3, 0.7) * rng.choice([-1, 1])

    X_pre = generate_lingam_data(B_pre, drift_at, noise_dist=noise_dist,
                                  random_state=random_state)
    X_post = generate_lingam_data(B_post, n_samples - drift_at, noise_dist=noise_dist,
                                   random_state=random_state + 100)
    X = np.vstack([X_pre, X_post])

    meta = {
        "B_pre": B_pre,
        "B_post": B_post,
        "true_order": order,
        "drift_at": drift_at,
        "n_vars": n_vars,
    }
    return X, meta

def federated_split(
    X: np.ndarray,
    n_clients: int = 2,
    random_state: int = 42,
) -> list[np.ndarray]:
    """
    Horizontally split data across clients.

    Horizontal split preserves the joint distribution on each client
    (each client sees all variables, a subset of samples), which is the
    setting Federated LiNGAM is designed for.
    """
    rng = np.random.default_rng(random_state)
    idx = rng.permutation(len(X))
    splits = np.array_split(idx, n_clients)
    return [X[s] for s in splits]

def generate_heterogeneous_clients(
    n_vars: int,
    n_samples_per_client: int,
    n_clients: int,
    heterogeneity: float,
    edge_density: float = 0.3,
    noise_dist: str = "exponential",
    random_state: int = 42,
) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray]]:
    """
    Generate K clients sharing the same causal structure but with
    heterogeneous coefficients.

    Args:
        heterogeneity: scale of per-edge coefficient perturbation.
            0.0 -> i.i.d. clients (identical SEM).
            1.0 -> each edge's coefficient independently scaled by
                   factor ~ N(1, 0.5) per client.

    Returns:
        client_data: list of K data arrays.
        B_global: shared edge support as binary adjacency.
        B_per_client: list of K per-client weighted adjacency matrices.
    """
    rng = np.random.default_rng(random_state)

    # Shared structure (base DAG)
    B_base, _ = generate_lingam_dag(
        n_vars, edge_density, random_state=random_state
    )

    B_per_client = []
    client_data = []
    for k in range(n_clients):
        B_k = B_base.copy()
        nz = np.argwhere(B_k != 0)
        for i, j in nz:
            # Multiplicative perturbation with magnitude scaled by heterogeneity
            factor = 1.0 + heterogeneity * rng.normal(0, 0.5)
            B_k[i, j] *= factor
        B_per_client.append(B_k)

        X_k = generate_lingam_data(
            B_k, n_samples_per_client,
            noise_dist=noise_dist,
            random_state=random_state + k + 100,
        )
        client_data.append(X_k)

    # Ground-truth structure is the shared support
    B_global = (np.abs(B_base) > 0).astype(int)

    return client_data, B_global, B_per_client

def generate_structurally_heterogeneous_clients(
    n_vars: int,
    n_samples_per_client: int,
    n_clients: int,
    perturb_prob: float,
    edge_density: float = 0.3,
    noise_dist: str = "exponential",
    random_state: int = 42,
) -> tuple[list[np.ndarray], dict]:
    """
    Generate K clients sharing a base DAG but with client-specific edges.

    Each edge in the base DAG is dropped per-client with probability
    `perturb_prob`; additional edges are added per-client with the same
    probability. Node ordering is preserved to maintain acyclicity.

    Args:
        perturb_prob: probability that each base edge is dropped in a given
            client, and probability that a new edge is added.
            0.0 -> all clients identical (i.i.d. structure).
            1.0 -> each edge is essentially random.

    Returns:
        client_data: list of K data arrays.
        meta: dict containing:
            - "shared": binary matrix of edges present in ALL clients
            - "specific": binary matrix of edges present in SOME but not all
            - "union": binary matrix of edges present in ANY client
            - "B_per_client": list of K weighted adjacency matrices
    """
    rng = np.random.default_rng(random_state)

    B_base, _ = generate_lingam_dag(
        n_vars, edge_density, random_state=random_state
    )
    base_edges = set(map(tuple, np.argwhere(B_base != 0)))

    B_per_client = []
    client_data = []
    edge_presence = np.zeros((n_clients, n_vars, n_vars), dtype=int)

    for k in range(n_clients):
        B_k = np.zeros_like(B_base)

        # Retain or drop each base edge
        for (i, j) in base_edges:
            if rng.random() > perturb_prob:
                B_k[i, j] = B_base[i, j] * (1.0 + 0.2 * rng.normal())

        # Add client-specific edges (i < j to preserve acyclicity)
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if (i, j) in base_edges:
                    continue
                if rng.random() < perturb_prob * 0.3:
                    sign = rng.choice([-1, 1])
                    B_k[i, j] = sign * rng.uniform(0.5, 1.5)

        B_per_client.append(B_k)
        edge_presence[k] = (np.abs(B_k) > 0).astype(int)

        X_k = generate_lingam_data(
            B_k, n_samples_per_client,
            noise_dist=noise_dist,
            random_state=random_state + k + 100,
        )
        client_data.append(X_k)

    # Shared = present in all clients
    shared = np.all(edge_presence > 0, axis=0).astype(int)
    # Union = present in any client
    union = np.any(edge_presence > 0, axis=0).astype(int)
    # Specific = in union but not shared
    specific = (union - shared).clip(min=0)

    return client_data, {
        "shared": shared,
        "specific": specific,
        "union": union,
        "B_per_client": B_per_client,
    }
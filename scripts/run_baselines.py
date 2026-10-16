# scripts/run_baselines.py
"""Baseline comparison for federated HC-LiNGAM.

Baselines:
  B1: Centralized DirectLiNGAM (lingam package) - upper bound
  B2: Local HC-LiNGAM + majority edge voting
  B3: Federated HC-LiNGAM, no DP (epsilon = inf)
  B4: Federated HC-LiNGAM + DP (epsilon = 0.1)
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import mlx.core as mx

from lingam import DirectLiNGAM

from data.synthetic_lingam import (
    generate_lingam_dag, generate_lingam_data, federated_split,
)
from cumulant import local_cumulant_tensor
from hc_lingam import hc_lingam, threshold_adjacency
from evaluation import (
    compute_shd, compute_f1, ancestral_accuracy, direction_accuracy,
)
from federated import run_federated_lingam


# --- Config (matches epsilon_scan) ---
SEEDS = list(range(10))
N_VARS = 6
N_SAMPLES = 1000
N_CLIENTS = 2
NOISE_DIST = "exponential"
EDGE_DENSITY = 0.3
TAU = 0.1
RIDGE_ALPHA = 1e-3
DELTA = 1e-5
EPS_DP = 0.1  # epsilon for B4


def _make_data(seed):
    B_true, _ = generate_lingam_dag(
        N_VARS, EDGE_DENSITY, random_state=seed + 1000
    )
    X = generate_lingam_data(
        B_true, N_SAMPLES, noise_dist=NOISE_DIST, random_state=seed + 1000
    )
    A_true = (B_true != 0).astype(int)
    return X, A_true


def _eval(name, B_est, order_est, A_true):
    B_t = threshold_adjacency(B_est, tau=TAU) if B_est.dtype.kind == "f" else B_est
    return {
        "method": name,
        "shd": compute_shd(B_t, A_true),
        "f1": compute_f1(B_t, A_true),
        "aa": ancestral_accuracy(A_true, order_est) if order_est else np.nan,
        "dir_acc": direction_accuracy(B_t, A_true),
    }


# --- B1: Centralized DirectLiNGAM (lingam package) ---
def run_b1(X, A_true):
    model = DirectLiNGAM()
    model.fit(X)
    # lingam convention: B[i,j] != 0 means j -> i; transpose to our convention.
    B = model.adjacency_matrix_.T
    order = list(model.causal_order_)
    return _eval("B1_centralized_DirectLiNGAM", B, order, A_true)


# --- B2: Local HC-LiNGAM + edge voting ---
def run_b2(client_data, A_true):
    n_vars = client_data[0].shape[1]
    votes = np.zeros((n_vars, n_vars), dtype=int)
    for data in client_data:
        c, _ = local_cumulant_tensor(
            mx.array(data.astype(np.float32)), standardize=False
        )
        B_k, _ = hc_lingam(c[2], c[3], n_vars, ridge_alpha=RIDGE_ALPHA)
        B_k = threshold_adjacency(B_k, tau=TAU)
        votes += (np.abs(B_k) > 0).astype(int)
    # Majority vote
    B_vote = (votes > len(client_data) / 2).astype(float)
    # Order: no global order from voting; use AA = NaN
    return _eval("B2_local_voting", B_vote, None, A_true)


# --- B3: Federated no DP ---
def run_b3(client_data, A_true):
    B, order = run_federated_lingam(
        client_data, epsilon=float("inf"), delta=DELTA,
        standardize=False, ridge_alpha=RIDGE_ALPHA, seed=0,
    )
    return _eval("B3_federated_no_dp", B, order, A_true)


# --- B4: Federated + DP ---
def run_b4(client_data, A_true, seed):
    B, order = run_federated_lingam(
        client_data, epsilon=EPS_DP, delta=DELTA,
        standardize=False, ridge_alpha=RIDGE_ALPHA, seed=seed,
    )
    return _eval("B4_federated_dp", B, order, A_true)


def main(out_dir: str = "results"):
    os.makedirs(out_dir, exist_ok=True)
    records = []

    for seed in SEEDS:
        X, A_true = _make_data(seed)
        client_data = federated_split(X, N_CLIENTS, random_state=seed)

        for runner in [run_b1, run_b2, run_b3, run_b4]:
            if runner is run_b1:
                rec = runner(X, A_true)
            elif runner is run_b4:
                rec = runner(client_data, A_true, seed)
            else:
                rec = runner(client_data, A_true)
            rec["seed"] = seed
            records.append(rec)

    df = pd.DataFrame(records)
    csv_path = os.path.join(out_dir, "baselines.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved to {csv_path}\n")

    # Summary
    print("=" * 88)
    print(f"{'method':<32} | {'SHD':>10} | {'F1':>10} | {'AA':>10} | {'DirAcc':>10}")
    print("-" * 88)
    for method in df["method"].unique():
        sub = df[df["method"] == method]
        aa = sub["aa"].mean()
        aa_str = f"{aa:.3f}±{sub['aa'].std():.3f}" if not np.isnan(aa) else "  N/A    "
        print(
            f"{method:<32} | "
            f"{sub['shd'].mean():>4.2f}±{sub['shd'].std():<4.2f} | "
            f"{sub['f1'].mean():>5.3f}±{sub['f1'].std():<4.3f} | "
            f"{aa_str:>10} | "
            f"{sub['dir_acc'].mean():>5.3f}±{sub['dir_acc'].std():<4.3f}"
        )


if __name__ == "__main__":
    main()
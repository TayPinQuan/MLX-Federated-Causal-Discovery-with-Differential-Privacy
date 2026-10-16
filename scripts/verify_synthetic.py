# scripts/verify_synthetic.py
"""Verify HC-LiNGAM on synthetic LiNGAM data with known ground truth.

Uses ONLY asymmetric (nonzero-skew) noise distributions, because
third-order cumulant methods are blind to symmetric noise.
"""
import numpy as np
import mlx.core as mx

from data.synthetic_lingam import generate_lingam_dag, generate_lingam_data
from cumulant import local_cumulant_tensor
from hc_lingam import hc_lingam, threshold_adjacency
from evaluation import compute_shd, compute_f1, ancestral_accuracy, direction_accuracy


def run_once(n_vars, n_samples, edge_density, noise_dist, seed):
    B_true, true_order = generate_lingam_dag(
        n_vars, edge_density, random_state=seed
    )
    X = generate_lingam_data(
        B_true, n_samples, noise_dist=noise_dist, random_state=seed
    )

    cumulants, _ = local_cumulant_tensor(mx.array(X), standardize=False)
    B_est, order_est = hc_lingam(
        cumulants[2], cumulants[3], n_vars, ridge_alpha=1e-3
    )

    B_t = threshold_adjacency(B_est, tau=0.1)
    A_true = (B_true != 0).astype(int)

    return {
        "AA": ancestral_accuracy(A_true, order_est),
        "SHD": compute_shd(B_t, A_true),
        "F1": compute_f1(B_t, A_true),
        "DirAcc": direction_accuracy(B_t, A_true),
        "order_true": true_order,
        "order_est": order_est,
    }


def main():
    print("=" * 72)
    print("Synthetic LiNGAM verification (asymmetric noise only)")
    print("=" * 72)

    # Scenario 1: small clean problem
    print("\nScenario 1: 6 vars, 5000 samples, exponential noise, edge_density=0.3")
    aa_list = []
    for seed in [0, 1, 2, 3, 4]:
        m = run_once(6, 5000, 0.3, "exponential", seed)
        aa_list.append(m["AA"])
        print(f"  seed={seed}: AA={m['AA']:.3f}  SHD={m['SHD']:>2}  "
              f"F1={m['F1']:.3f}  DirAcc={m['DirAcc']:.3f}")
    print(f"  mean AA = {np.mean(aa_list):.3f}")

    # Scenario 2: more variables
    print("\nScenario 2: 10 vars, 5000 samples, exponential noise, edge_density=0.3")
    aa_list = []
    for seed in [0, 1, 2, 3, 4]:
        m = run_once(10, 5000, 0.3, "exponential", seed)
        aa_list.append(m["AA"])
        print(f"  seed={seed}: AA={m['AA']:.3f}  SHD={m['SHD']:>2}  "
              f"F1={m['F1']:.3f}  DirAcc={m['DirAcc']:.3f}")
    print(f"  mean AA = {np.mean(aa_list):.3f}")

    # Scenario 3: various asymmetric noise distributions
    print("\nScenario 3: 8 vars, 5000 samples, various ASYMMETRIC noise")
    for dist in ["exponential", "gamma", "chi2", "uniform_mix"]:
        results = [run_once(8, 5000, 0.3, dist, s) for s in range(5)]
        aa = np.mean([r["AA"] for r in results])
        da = np.mean([r["DirAcc"] for r in results])
        print(f"  {dist:<12}: AA={aa:.3f}  DirAcc={da:.3f}")

    # Scenario 4: sample size sensitivity
    print("\nScenario 4: 8 vars, exponential noise, varying sample size")
    for n in [500, 1000, 2000, 5000, 10000]:
        results = [run_once(8, n, 0.3, "exponential", s) for s in range(5)]
        aa = np.mean([r["AA"] for r in results])
        da = np.mean([r["DirAcc"] for r in results])
        print(f"  n={n:<6}: AA={aa:.3f}  DirAcc={da:.3f}")


if __name__ == "__main__":
    main()
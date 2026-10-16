# scripts/plot_ablations.py
"""Generate ablation summary figures."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_standardization():
    df = pd.read_csv("results/standardization_ablation.csv")
    df["eps_plot"] = df["epsilon"].replace(np.inf, 1e3)
    fig, ax = plt.subplots(figsize=(6, 4))
    for std, label, color in [(False, "Raw (default)", "C0"),
                              (True, "Standardized", "C3")]:
        sub = df[df["standardize"] == std].groupby("eps_plot").agg(
            aa_mean=("aa", "mean"), aa_std=("aa", "std"),
        ).reset_index()
        ax.errorbar(sub["eps_plot"], sub["aa_mean"], yerr=sub["aa_std"],
                    marker="o", label=label, color=color, capsize=3)
    ax.axhline(0.5, ls=":", color="red", alpha=0.5, label="chance")
    ax.set_xscale("log")
    ax.set_xlabel("Privacy budget ε")
    ax.set_ylabel("Ancestral Accuracy")
    ax.set_title("Standardization destroys DP robustness")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/fig_standardization.png", dpi=150)
    print("Saved results/fig_standardization.png")


def plot_client_count():
    df = pd.read_csv("results/client_count_ablation.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    for eps, label, color in [(float("inf"), "ε = ∞", "C0"),
                              (0.1, "ε = 0.1", "C3")]:
        sub = df[df["epsilon"] == eps].groupby("n_clients").agg(
            shd_mean=("shd", "mean"), shd_std=("shd", "std"),
        ).reset_index()
        ax.errorbar(sub["n_clients"], sub["shd_mean"], yerr=sub["shd_std"],
                    marker="o", label=label, color=color, capsize=3)
    ax.set_xlabel("Number of clients K")
    ax.set_ylabel("SHD")
    ax.set_title("Fragmentation cost under DP")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/fig_client_count.png", dpi=150)
    print("Saved results/fig_client_count.png")

def plot_structure_heterogeneity():
    df = pd.read_csv("results/structure_heterogeneity.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for eps_label, ls in [("inf", "-"), ("0.1", "--")]:
        sub = df[df["epsilon"] == eps_label]
        for col, label, color in [
            ("recall_shared", f"R_shared (ε={eps_label})", "C0"),
            ("recall_specific", f"R_specific (ε={eps_label})", "C3"),
        ]:
            agg = sub.groupby("perturb_prob").agg(
                m=(col, "mean"), s=(col, "std"),
            ).reset_index()
            ax.errorbar(agg["perturb_prob"], agg["m"], yerr=agg["s"],
                        marker="o", ls=ls, label=label, color=color,
                        capsize=3)
    ax.set_xlabel("perturb_prob (client structure divergence)")
    ax.set_ylabel("Recall")
    ax.set_title("Federation favors shared structure over specific")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/fig_structure_heterogeneity.png", dpi=150)
    print("Saved results/fig_structure_heterogeneity.png")

if __name__ == "__main__":
    plot_standardization()
    plot_client_count()
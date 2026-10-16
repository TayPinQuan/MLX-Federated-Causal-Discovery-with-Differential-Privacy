# visualize.py
"""Privacy-accuracy trade-off plots."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_privacy_accuracy(
    csv_path: str = "results/epsilon_scan.csv",
    save_path: str = "results/privacy_accuracy.png",
):
    df = pd.read_csv(csv_path)
    # Replace inf with a large finite number for plotting
    df["eps_plot"] = df["epsilon"].replace(np.inf, 1e3)

    summary = df.groupby("eps_plot").agg(
        shd_mean=("shd", "mean"), shd_std=("shd", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        aa_mean=("aa", "mean"), aa_std=("aa", "std"),
        dir_mean=("dir_acc", "mean"), dir_std=("dir_acc", "std"),
    ).reset_index()

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    configs = [
        ("shd_mean", "shd_std", "SHD", "lower is better"),
        ("f1_mean", "f1_std", "F1", "higher is better"),
        ("aa_mean", "aa_std", "Ancestral Accuracy", "higher is better"),
        ("dir_mean", "dir_std", "Direction Accuracy", "higher is better"),
    ]

    for ax, (mean_col, std_col, label, sub) in zip(axes, configs):
        ax.errorbar(
            summary["eps_plot"], summary[mean_col],
            yerr=summary[std_col], marker="o", capsize=4,
        )
        ax.set_xscale("log")
        ax.set_xlabel("Privacy budget ε (log scale)")
        ax.set_ylabel(label)
        ax.set_title(f"{label}\n({sub})")
        ax.grid(alpha=0.3)

        # Mark chance level for AA / DirAcc
        if label in ("Ancestral Accuracy", "Direction Accuracy"):
            ax.axhline(0.5, linestyle=":", color="red", alpha=0.5,
                       label="chance")
            ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved {save_path}")


if __name__ == "__main__":
    plot_privacy_accuracy()
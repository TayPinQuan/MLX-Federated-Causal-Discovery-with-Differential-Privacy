# data/sachs.py
"""Sachs protein signaling dataset: loading and federated partitioning."""
from __future__ import annotations

import os
import urllib.request
import numpy as np
import pandas as pd

SACHS_COLUMNS = [
    "Raf", "Mek", "Plcg", "PIP2", "PIP3",
    "Erk", "Akt", "PKA", "PKC", "P38", "Jnk",
]

SACHS_EDGES = [
    ("PKC", "Raf"), ("PKC", "Mek"), ("PKC", "Jnk"),
    ("PKC", "P38"), ("PKC", "PKA"),
    ("PKA", "Raf"), ("PKA", "Mek"), ("PKA", "Erk"),
    ("PKA", "Akt"), ("PKA", "Jnk"), ("PKA", "P38"),
    ("Raf", "Mek"), ("Mek", "Erk"), ("Erk", "Akt"),
    ("Plcg", "PIP2"), ("Plcg", "PIP3"), ("PIP3", "PIP2"),
]

SACHS_URL = (
    "https://raw.githubusercontent.com/IowaBiostat/data-sets/"
    "main/Sachs2005/Sachs2005.txt"
)

# Candidate local cache paths (checked in order)
_LOCAL_CANDIDATES = [
    "data/cache/Sachs2005.txt",
    "data/cache/sachs.txt",
    "data/cache/sachs.csv",
]


def _find_local(cache_dir: str = "data/cache") -> str | None:
    """Look for a pre-downloaded Sachs file. Returns path or None."""
    # Check project-relative candidates
    for rel in _LOCAL_CANDIDATES:
        if os.path.exists(rel):
            return rel
    # Check files in cache_dir generically
    if os.path.isdir(cache_dir):
        for name in os.listdir(cache_dir):
            low = name.lower()
            if "sachs" in low and low.endswith((".txt", ".csv")):
                return os.path.join(cache_dir, name)
    return None


def _download_sachs(cache_dir: str = "data/cache") -> str:
    """Return a path to Sachs data, preferring local cache."""
    local = _find_local(cache_dir)
    if local is not None:
        print(f"Using cached Sachs: {local}")
        return local

    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "Sachs2005.txt")
    print(f"No local cache found. Downloading to {path}...")
    try:
        urllib.request.urlretrieve(SACHS_URL, path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download Sachs from {SACHS_URL}. "
            f"Please download manually and place it in {cache_dir}/. "
            f"Original error: {e}"
        )
    return path


def _read_sachs(path: str) -> pd.DataFrame:
    """Read Sachs from either .txt (tab-separated) or .csv."""
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        # IowaBiostat / bnlearn versions are tab-separated
        try:
            df = pd.read_csv(path, sep="\t")
        except Exception:
            df = pd.read_csv(path)

    # Normalize column names
    if all(c in df.columns for c in SACHS_COLUMNS):
        return df[SACHS_COLUMNS]

    # Case-insensitive match
    col_map = {c.lower(): c for c in df.columns}
    matched = [col_map.get(c.lower()) for c in SACHS_COLUMNS]
    if all(m is not None for m in matched):
        df = df[matched]
        df.columns = SACHS_COLUMNS
        return df

    # Fallback: take first 11 numeric columns
    num_df = df.select_dtypes(include=[np.number])
    df = num_df.iloc[:, :11]
    df.columns = SACHS_COLUMNS
    return df


def load_sachs() -> tuple[np.ndarray, np.ndarray]:
    """
    Load Sachs data and ground-truth adjacency.

    Returns:
        X: (n_samples, 11) float32 array, columns in SACHS_COLUMNS order.
        true_adj: (11, 11) int array; true_adj[i, j] = 1 means i -> j.
    """
    path = _download_sachs()
    df = _read_sachs(path)
    X = df.values.astype(np.float32)

    idx = {name: i for i, name in enumerate(SACHS_COLUMNS)}
    true_adj = np.zeros((11, 11), dtype=int)
    for src, dst in SACHS_EDGES:
        true_adj[idx[src], idx[dst]] = 1

    return X, true_adj


def load_sachs_federated(
    n_clients: int = 2,
    random_state: int = 42,
    scheme: str = "horizontal",
) -> list[np.ndarray]:
    """Split Sachs across clients."""
    X, _ = load_sachs()
    rng = np.random.default_rng(random_state)

    if scheme == "horizontal":
        indices = rng.permutation(len(X))
        splits = np.array_split(indices, n_clients)
        return [X[s] for s in splits]
    elif scheme == "vertical":
        cols = rng.permutation(X.shape[1])
        splits = np.array_split(cols, n_clients)
        return [X[:, s] for s in splits]
    else:
        raise ValueError(f"Unknown scheme: {scheme}")
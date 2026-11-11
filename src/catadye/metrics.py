"""Continual-learning metrics from a loss matrix, plus paired bootstrap CIs.

R[i][j] = eval loss on domain j after finishing training on domain i (i = 0 is the untrained model
at index -1 convention: we store `R0[j]` separately). Lower is better.
"""
from __future__ import annotations

import numpy as np


def forgetting(R: np.ndarray) -> float:
    """Mean loss increase on earlier domains between the end of their own training and the end.

    R is (T, T). For domain j < T-1: R[T-1, j] - R[j, j]. Positive = forgetting.
    """
    T = R.shape[0]
    return float(np.mean([R[T - 1, j] - R[j, j] for j in range(T - 1)]))


def backward_transfer(R: np.ndarray) -> float:
    """Negative of forgetting in loss units (higher = better), mirroring the usual BWT sign."""
    return -forgetting(R)


def forward_transfer(R: np.ndarray, R0: np.ndarray) -> float:
    """Mean gain on domain j right before training on it vs the untrained model: R0[j] - R[j-1, j]."""
    T = R.shape[0]
    return float(np.mean([R0[j] - R[j - 1, j] for j in range(1, T)]))


def final_avg(R: np.ndarray) -> float:
    return float(np.mean(R[-1]))


def new_domain_loss(R: np.ndarray) -> float:
    """Loss on the domain just trained, averaged over domains (plasticity)."""
    return float(np.mean(np.diag(R)))


def paired_bootstrap_ci(a, b, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    """CI for mean(a - b) with seed-paired samples. Returns (mean, lo, hi)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    return float(d.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def holm(pvals):
    """Holm-adjusted p-values, same order as input."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj

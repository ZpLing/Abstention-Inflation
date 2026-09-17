"""P1 — Counterfactual Abstention-Inflation Direction (AID).

PLAN_v2 §四 小实验 1D, P1.

Given matched (S1, S2) prompt-pair activations on Abstention Inflation samples (samples where
S2 outputs Unknown), compute per-layer diff-of-means:

    AID(ℓ) = mean[h_ℓ(x_S2)] − mean[h_ℓ(x_S1)]

The best layer ℓ* is the one where AID(ℓ) best linearly separates "S1 commit"
vs "S2 abstain" activations on a held-out split (logistic AUC).

Inputs: two arrays of shape (n_ai, n_layers, hidden) — one for each setting.
Outputs: AID matrix (n_layers, hidden), per-layer AUC vector, ℓ*, ê_AID*.

Sanity baselines (PLAN_v2 §1D P1): random direction AUC + cosine similarity
to a harm-refusal direction. The harm-refusal direction is out-of-scope for
this minimal first cut — we record `random_baseline_auc` and leave the
refusal-direction comparison as a TODO surfaced in the summary so the user
remembers to add it before paper submission.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class P1Result:
    aid: np.ndarray                  # (L, H) — per-layer AID
    per_layer_auc: np.ndarray        # (L,)    — held-out logistic AUC
    best_layer: int                  # ℓ* (index into per_layer_auc)
    aid_unit_at_best: np.ndarray     # (H,)    — ê_AID at ℓ*
    random_baseline_auc: float       # mean AUC of random unit directions at ℓ*


def compute_aid(s1_hs: np.ndarray, s2_hs: np.ndarray) -> np.ndarray:
    """Per-layer diff-of-means.

    s1_hs, s2_hs: (n_ai, L, H). Returns (L, H). Caller is responsible for
    aligning samples — entry i in s1_hs and s2_hs must be the *same* Abs Rate
    sample under the two prompt settings.
    """
    if s1_hs.shape != s2_hs.shape:
        raise ValueError(
            f"[P1] shape mismatch: S1={s1_hs.shape} S2={s2_hs.shape}"
        )
    return s2_hs.mean(axis=0) - s1_hs.mean(axis=0)


def per_layer_auc(s1_hs: np.ndarray, s2_hs: np.ndarray,
                   *, test_frac: float = 0.3, seed: int = 0) -> np.ndarray:
    """Per-layer held-out logistic AUC of the S1-vs-S2 binary task.

    Returns (L,). One LogisticRegression per layer. Held-out split is shared
    across layers so the AUCs are directly comparable.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    n, L, H = s1_hs.shape
    if n < 4:
        # Not enough data for a stable split; return zeros and let runner warn.
        return np.zeros(L, dtype=np.float32)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = max(1, int(round(test_frac * n)))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    aucs = np.zeros(L, dtype=np.float32)
    for ell in range(L):
        x = np.concatenate([s1_hs[:, ell, :], s2_hs[:, ell, :]], axis=0)
        y = np.concatenate([np.zeros(n), np.ones(n)], axis=0)
        # Map indices into the doubled array: S1 first n rows, S2 next n.
        train_full = np.concatenate([train_idx, train_idx + n])
        test_full = np.concatenate([test_idx, test_idx + n])
        clf = LogisticRegression(
            max_iter=2000, solver="liblinear", C=1.0
        )
        try:
            clf.fit(x[train_full], y[train_full])
            scores = clf.decision_function(x[test_full])
            aucs[ell] = float(roc_auc_score(y[test_full], scores))
        except Exception:
            aucs[ell] = 0.5  # degenerate column / class imbalance
    return aucs


def random_baseline_auc(s1_hs: np.ndarray, s2_hs: np.ndarray, layer: int,
                         *, n_directions: int = 16, seed: int = 0) -> float:
    """Mean held-out AUC of `n_directions` random unit directions at one layer.

    Sanity check from PLAN_v2 §1D P1: AID's AUC should be much higher than
    a random direction's AUC at the same layer.
    """
    from sklearn.metrics import roc_auc_score

    n = s1_hs.shape[0]
    H = s1_hs.shape[2]
    if n < 4:
        return 0.5

    rng = np.random.default_rng(seed + 1)
    x_s1 = s1_hs[:, layer, :]
    x_s2 = s2_hs[:, layer, :]
    aucs = []
    for _ in range(n_directions):
        v = rng.standard_normal(H)
        v /= (np.linalg.norm(v) + 1e-8)
        s1_proj = x_s1 @ v
        s2_proj = x_s2 @ v
        scores = np.concatenate([s1_proj, s2_proj])
        labels = np.concatenate([np.zeros(n), np.ones(n)])
        try:
            aucs.append(roc_auc_score(labels, scores))
        except Exception:
            aucs.append(0.5)
    return float(np.mean(aucs))


def fit_p1(s1_hs: np.ndarray, s2_hs: np.ndarray) -> P1Result:
    """Full P1 pipeline: compute AID, per-layer AUC, pick ℓ*, baseline."""
    aid = compute_aid(s1_hs, s2_hs)                 # (L, H)
    aucs = per_layer_auc(s1_hs, s2_hs)              # (L,)
    best = int(np.argmax(aucs))
    aid_at_best = aid[best]
    ehat = aid_at_best / (np.linalg.norm(aid_at_best) + 1e-8)
    rand_auc = random_baseline_auc(s1_hs, s2_hs, best)
    return P1Result(
        aid=aid, per_layer_auc=aucs, best_layer=best,
        aid_unit_at_best=ehat, random_baseline_auc=rand_auc,
    )

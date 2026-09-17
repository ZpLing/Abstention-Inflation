"""P3 — 3-way representation-level attribution of Abstention Inflation samples.

PLAN_v2 §四 小实验 1D, P3.

Train a 3-class softmax classifier at the best layer ℓ* (from P1) on three
*independent* ground-truth pools — no exclusion-rule definitions, every class
has its own pure training set:

    γ-pool = S1-correct samples' hidden state    (knows + dares to say)
    β-pool = S1-incorrect samples' hidden state  ("thought I knew but I don't")
    α-pool = FLD genuinely-Unknown samples that S2 outputs Unknown
            (model correctly recognizes objective unanswerability)

Apply to Abstention Inflation samples → per-sample (p_γ, p_β, p_α). argmax = bucket; max-prob =
confidence. PLAN_v2 §1D P3 explicitly notes this is a partition (covers 100%
of Abs Rate) and that boundary cases get a confidence < 0.6 marker.

Inputs:
    gamma_hs   (n_γ, H)
    beta_hs    (n_β, H)
    alpha_hs   (n_α, H)
    ai_hs     (n_ai, H)        — pre-sliced at ℓ*

Outputs (P3Result):
    train_acc        sanity: 3-class fit on its own training set
    ai_labels       length n_ai, each in {"gamma","beta","alpha"}
    ai_probs        (n_ai, 3) — column order is (gamma, beta, alpha)
    ai_confidences  length n_ai, max prob per sample
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


_CLASSES = ("gamma", "beta", "alpha")


@dataclass
class P3Result:
    layer: int
    train_acc: float
    ai_labels: List[str]
    ai_probs: np.ndarray             # (n_ai, 3) in order (γ, β, α)
    ai_confidences: List[float]
    bucket_counts: dict               # {"gamma": int, "beta": int, "alpha": int}
    high_conf_share: float            # fraction with confidence > 0.6
    pool_sizes: dict                  # {"gamma": n, "beta": n, "alpha": n}


def fit_p3(gamma_hs: np.ndarray, beta_hs: np.ndarray, alpha_hs: np.ndarray,
            ai_hs: np.ndarray, layer: int) -> P3Result:
    from sklearn.linear_model import LogisticRegression

    pool_sizes = {
        "gamma": int(gamma_hs.shape[0]),
        "beta":  int(beta_hs.shape[0]),
        "alpha": int(alpha_hs.shape[0]),
    }
    # Need at least 2 samples per class for a 3-way fit to be meaningful.
    if min(pool_sizes.values()) < 2 or ai_hs.shape[0] == 0:
        return P3Result(
            layer=layer, train_acc=0.0, ai_labels=[],
            ai_probs=np.zeros((0, 3), dtype=np.float32),
            ai_confidences=[], bucket_counts={c: 0 for c in _CLASSES},
            high_conf_share=0.0, pool_sizes=pool_sizes,
        )

    X = np.concatenate([gamma_hs, beta_hs, alpha_hs], axis=0)
    y = np.concatenate([
        np.full(gamma_hs.shape[0], 0, dtype=np.int64),  # γ → 0
        np.full(beta_hs.shape[0],  1, dtype=np.int64),  # β → 1
        np.full(alpha_hs.shape[0], 2, dtype=np.int64),  # α → 2
    ])

    clf = LogisticRegression(
        max_iter=3000, solver="lbfgs", C=1.0, multi_class="multinomial"
    )
    clf.fit(X, y)
    train_acc = float((clf.predict(X) == y).mean())

    # Predict on Abs Rate.
    probs = clf.predict_proba(ai_hs)                   # (n_ai, K)
    # sklearn orders columns by `clf.classes_`. Reorder to (γ, β, α).
    col_order = [list(clf.classes_).index(i) for i in (0, 1, 2)]
    probs = probs[:, col_order]
    label_idx = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    ai_labels = [_CLASSES[i] for i in label_idx]
    counts = {c: int(np.sum(label_idx == i)) for i, c in enumerate(_CLASSES)}
    high_conf = float((confidences > 0.6).mean()) if len(confidences) else 0.0

    return P3Result(
        layer=layer,
        train_acc=train_acc,
        ai_labels=ai_labels,
        ai_probs=probs.astype(np.float32),
        ai_confidences=[float(c) for c in confidences],
        bucket_counts=counts,
        high_conf_share=high_conf,
        pool_sizes=pool_sizes,
    )

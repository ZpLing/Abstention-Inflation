"""P2 — "Knows but won't say" linear letter-decodability probe.

PLAN_v2 §四 小实验 1D, P2.

Train a per-layer 4-class softmax (sklearn LogisticRegression, multinomial)
on S1-correct samples (letter is known and was committed to). Evaluate on
Abstention Inflation sample S2 hidden states; per-sample probe confidence

    c_letter(x) = max_letter softmax_prob

is the headline. High c_letter on an Abstention Inflation sample = "internal state already
locked onto a letter, output layer suppressed it" → γ candidate.

Inputs:
    train_hs       (n_train, L, H)  — S1 forward pass on S1-correct samples
    train_letters  (n_train,)       — letter strings ("A".."D")
    ai_s2_hs      (n_ai, L, H)    — S2 forward pass on Abstention Inflation samples
    layer          int              — which layer to report
    ai_gold       (n_ai,)         — gold letter for each Abstention Inflation sample

Outputs (P2Result):
    train_acc / train_f1 (sanity: probe should fit on its own training set)
    test_topk_acc        (does max-prob letter on Abs Rate == gold? interesting,
                          but the literature interprets this as "linearly
                          decodable knowledge" not "model uses it" — see
                          PLAN_v2 §四 1D limitation note 2.)
    ai_confidence       per-sample c_letter
    ai_pred_letter      per-sample argmax letter
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


_LETTERS = ["A", "B", "C", "D"]


@dataclass
class P2Result:
    layer: int
    train_acc: float
    test_acc_on_ai: float                  # how often probe's argmax = gold letter
    ai_pred_letters: List[str]             # length n_ai
    ai_confidences: List[float]            # length n_ai


def _letters_to_idx(letters: List[str]) -> np.ndarray:
    return np.array([_LETTERS.index(c) for c in letters], dtype=np.int64)


def fit_p2(train_hs: np.ndarray, train_letters: List[str],
            ai_s2_hs: np.ndarray, ai_gold_letters: List[str],
            layer: int) -> P2Result:
    from sklearn.linear_model import LogisticRegression

    if train_hs.shape[0] != len(train_letters):
        raise ValueError("[P2] train_hs / train_letters length mismatch")
    if ai_s2_hs.shape[0] != len(ai_gold_letters):
        raise ValueError("[P2] ai_s2_hs / ai_gold_letters length mismatch")
    if train_hs.shape[0] < 4:
        # Probe is degenerate without enough data; surface zeros + skip.
        return P2Result(
            layer=layer, train_acc=0.0, test_acc_on_ai=0.0,
            ai_pred_letters=["UNK"] * ai_s2_hs.shape[0],
            ai_confidences=[0.0] * ai_s2_hs.shape[0],
        )

    Xtr = train_hs[:, layer, :]
    ytr = _letters_to_idx(train_letters)
    Xte = ai_s2_hs[:, layer, :]
    yte = _letters_to_idx(ai_gold_letters) if ai_gold_letters else None

    clf = LogisticRegression(
        max_iter=3000, solver="lbfgs", C=1.0, multi_class="multinomial"
    )
    clf.fit(Xtr, ytr)
    train_pred = clf.predict(Xtr)
    train_acc = float((train_pred == ytr).mean())

    proba = clf.predict_proba(Xte)              # (n_ai, 4)
    pred_idx = proba.argmax(axis=1)
    confidences = proba.max(axis=1)
    pred_letters = [_LETTERS[i] for i in pred_idx]

    if yte is not None and len(yte) > 0:
        test_acc = float((pred_idx == yte).mean())
    else:
        test_acc = 0.0

    return P2Result(
        layer=layer,
        train_acc=train_acc,
        test_acc_on_ai=test_acc,
        ai_pred_letters=pred_letters,
        ai_confidences=[float(c) for c in confidences],
    )

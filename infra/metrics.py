"""Label-level metrics shared by every setting (paper Sec. 3.5).

        label_acc(preds, golds)             — *Acc* (paper Eq. 1), exact-match
                                              accuracy
        abs_rate(preds)                     — *Abs Rate* (paper Eq. 2), the
                                              fraction answered "Unknown"
        label_macro_f1(preds, golds, classes)
                                            — macro-F1 over the dataset's
                                              label space (incl. UNKNOWN)

Whether the model's own reasoning reached a conclusion its final answer then
withheld is S7's question; it is read off the stored raw text by the NLI probe
in experiments/C3_later_layer_override/S7_reasoning_traces_evaluation/.

S6 (self-diagnosis) lives in experiments/C2_deny_yet_capable/S6_self_diagnosis/ and
uses its own metrics — by design it is NOT a label-prediction task on the
original question.

Predictions are letter strings produced by
Evaluator.parse_mcq_tiered / parse_judge_tiered:
    "A" | "B" | "C" | "D"   — concrete option choice (or POS/NEG for Judge)
    "UNKNOWN"               — abstain
    "UNPARSEABLE"           — extract-match failed
    None                    — same as UNPARSEABLE

`answer_idx` is the ground-truth option index (0..3 for MCQ; 0=POS / 1=NEG
for Judge). Metrics here are computed on `answerable` samples only
(answer_idx >= 0); the runner is responsible for filtering.
"""

from typing import List, Sequence

_BAD = {"UNKNOWN", "UNPARSEABLE", None}


def is_correct(letter: str, answer_idx: int) -> bool:
    """Letter-string correctness check, abstaining counts as wrong on answerable."""
    if letter in _BAD:
        return False
    return ord(letter) - ord("A") == answer_idx


def label_acc(preds: Sequence[str], answer_idxs: Sequence[int]) -> float:
    n = len(preds)
    if n == 0:
        return 0.0
    return sum(is_correct(p, ai) for p, ai in zip(preds, answer_idxs)) / n


def label_macro_f1(
    preds: Sequence[str], answer_idxs: Sequence[int], classes: Sequence[str]
) -> float:
    """Macro-averaged F1 over the dataset's label space.

    `classes` is the full label space INCLUDING the abstain class (e.g.
    ["A", "B", "C", "D", "UNKNOWN"] for MCQ, ["A", "B", "UNKNOWN"] for Judge).
    Gold abstain samples should not appear in `answer_idxs` (answer_idx >= 0
    has already filtered them); but UNKNOWN may still appear in `preds` and
    is treated as its own class for F1 purposes.
    """
    if not preds:
        return 0.0
    # Normalize golds into the same string label space as preds.
    gold_letters = []
    for ai in answer_idxs:
        if ai < 0:
            gold_letters.append("UNKNOWN")
        else:
            gold_letters.append(chr(ord("A") + ai))

    f1s = []
    for c in classes:
        tp = sum(1 for p, g in zip(preds, gold_letters) if p == c and g == c)
        fp = sum(1 for p, g in zip(preds, gold_letters) if p == c and g != c)
        fn = sum(1 for p, g in zip(preds, gold_letters) if p != c and g == c)
        if tp == 0 and (fp == 0 or fn == 0):
            f1s.append(0.0)
            continue
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * precision * recall / (precision + recall))
    return sum(f1s) / len(f1s)


# Class-list helpers (callers should use these to avoid magic strings).


def mcq_classes(with_unknown: bool = True) -> List[str]:
    base = ["A", "B", "C", "D"]
    return base + (["UNKNOWN"] if with_unknown else [])


def judge_classes(with_unknown: bool = True) -> List[str]:
    base = ["A", "B"]
    return base + (["UNKNOWN"] if with_unknown else [])


def accuracy(preds, answer_idxs):
    return label_acc(preds, answer_idxs)


# ============================================================
# S9 — Unknown-labeled subset (the mirror image of Abs Rate)
#
# Computed on genuinely-Unknown samples only (``answer_idx == -1``, filtered
# by the runner). Here, choosing "Unknown" is the *correct* behaviour, so
# ``correct_abstention_rate`` equals accuracy on that subset, and the paper
# contrasts it with Abs Rate on the answerable subset.
# ============================================================


def correct_abstention_rate(preds: Sequence[str]) -> float:
    """Fraction of predictions that abstain, on Unknown-labeled samples.

    The higher, the better the model recognises that no determinable answer
    exists. The paper plots it against Abs Rate on answerable items to show
    the bias is directional rather than indiscriminate.
    """
    n = len(preds)
    if n == 0:
        return 0.0
    return sum(1 for p in preds if p == "UNKNOWN") / n


def forced_commitment_rate(preds: Sequence[str]) -> float:
    """S1 only: fraction of predictions that commit to a concrete label.

    Since S1 hides the abstain option, every committed answer on a genuinely
    unanswerable question is a forced wrong commitment — the cost the model
    pays when "Unknown" is removed. UNPARSEABLE is excluded (the model leaked
    an abstention despite the S1 phrasing).
    """
    n = len(preds)
    if n == 0:
        return 0.0
    return sum(1 for p in preds if p in ("A", "B")) / n


def commit_rate(preds: Sequence[str], letter: str) -> float:
    """Per-letter rate (used to spot a True/False bias under S1)."""
    n = len(preds)
    if n == 0:
        return 0.0
    return sum(1 for p in preds if p == letter) / n


def abs_rate(preds: Sequence[str]) -> float:
    """*Abs Rate* (paper Eq. 2) — fraction of items answered with "Unknown".

    Reported on the answerable subset, where every abstention is by definition
    an Abstention Inflation event. Under the S4 Word Content Ablation the
    abstain slot carries a different word; the parser maps that slot onto the
    same ``UNKNOWN`` token, so this function measures the changed word too.
    """
    n = len(preds)
    if n == 0:
        return 0.0
    return sum(1 for p in preds if p == "UNKNOWN") / n


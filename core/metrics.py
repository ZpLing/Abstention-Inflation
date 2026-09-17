"""Unified S1/S2 metrics — paper §3.5 Metrics.

Two evaluation layers; LABEL is the same across all datasets, but TRACE splits
into two families based on whether the gold trace is *discrete* or *prose*.

    LABEL layer (all datasets)
        label_acc(preds, golds)             — exact-match accuracy
        label_macro_f1(preds, golds, classes)
                                            — macro-F1 over the dataset's
                                              label space (incl. UNKNOWN)

    TRACE layer
        Family A — HARD F1 (discrete set arithmetic)
            Used for FLD (atoms = fact_i / int_i citations) and FEVER
            (atoms = normalized evidence-sentence keys). Pred and gold are
            Set[str]; F1 = 2*|P∩G| / (|P|+|G|).

            trace_set_f1(pred_atoms, gold_atoms)        → (P, R, F1)
            mean_trace_set_f1(pred_atoms_list, gold_atoms_list)

        Family B — SOFT F1 (BERTScore over free text)
            Used for ARC (gold = WorldTree fact strings) and MedQA
            (gold = MedReason free-form CoT). Pred and gold are bare strings;
            F1 is BERTScore-F1 between the two.

            trace_bertscore_f1(pred_text, gold_text)    → float
            mean_trace_bertscore_f1(pred_texts, gold_texts)

The split is deliberate (paper §3 Evaluation): exact-set arithmetic gives
strict, reproducible F1 on datasets whose gold trace is genuinely enumerable;
BERTScore captures semantic equivalence on datasets where models will
paraphrase rather than echo gold tokens.

S1–S4 main experiment reports (Acc_L, F1_L, F1_T) per (setting, dataset),
where F1_T is hard-set F1 on FLD/FEVER and BERTScore-F1 on ARC/MedQA. The
column is the same name in the summary JSON but the underlying computation
differs by family — readers must consult `trace_family` for interpretation.

S6 (self-diagnosis) lives in experiments/C2_introspective_gap/S6_self_diagnosis/ and
uses its own metrics — by design it is NOT a label-prediction task on the
original question.

Predictions for the LABEL layer are letter strings produced by
Evaluator.parse_mcq_tiered / parse_judge_tiered:
    "A" | "B" | "C" | "D"   — concrete option choice (or POS/NEG for Judge)
    "UNKNOWN"               — abstain
    "UNPARSEABLE"           — extract-match failed
    None                    — same as UNPARSEABLE

`answer_idx` is the ground-truth option index (0..3 for MCQ; 0=POS / 1=NEG
for Judge). Metrics here are computed on `answerable` samples only
(answer_idx >= 0); the runner is responsible for filtering.
"""
from typing import List, Set, Sequence, Tuple, Optional


_BAD = {"UNKNOWN", "UNPARSEABLE", None}


# ============================================================
# LABEL layer
# ============================================================

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


def label_macro_f1(preds: Sequence[str], answer_idxs: Sequence[int],
                    classes: Sequence[str]) -> float:
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


# ============================================================
# TRACE layer — Family A: HARD F1 (discrete set arithmetic)
#   Used for FLD (atoms = fact_i / int_i refs) and FEVER (atoms =
#   normalized evidence-sentence keys).
# ============================================================

def _safe_set(s) -> Set[str]:
    if s is None:
        return set()
    if isinstance(s, set):
        return s
    return set(s)


def trace_set_f1(pred_atoms, gold_atoms) -> Tuple[float, float, float]:
    """Set Precision / Recall / F1 over atomic units. Returns (P, R, F1).

    Convention:
        empty pred + empty gold        → (1.0, 1.0, 1.0)
        empty pred but non-empty gold  → (0, 0, 0)
        non-empty pred but empty gold  → (0, 0, 0)
    """
    p = _safe_set(pred_atoms)
    g = _safe_set(gold_atoms)
    if not p and not g:
        return 1.0, 1.0, 1.0
    if not p or not g:
        return 0.0, 0.0, 0.0
    tp = len(p & g)
    if tp == 0:
        return 0.0, 0.0, 0.0
    precision = tp / len(p)
    recall = tp / len(g)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def mean_trace_set_f1(pred_atoms_list, gold_atoms_list) -> float:
    """Per-sample set-F1, then macro-average across samples."""
    if not pred_atoms_list:
        return 0.0
    return sum(trace_set_f1(p, g)[2] for p, g in zip(pred_atoms_list, gold_atoms_list)) \
           / len(pred_atoms_list)


# Substring-containment variant — used for FEVER, where atoms are *content
# strings* of evidence sentences rather than discrete IDs. A model's quoted
# evidence usually appears verbatim inside its CoT but with a prefix
# ("The evidence states: ...") that breaks exact set equality. Bidirectional
# substring matching credits the model whenever its sentence contains the
# gold sentence (or vice-versa) as a substring after normalization.

def trace_substring_set_f1(pred_atoms, gold_atoms) -> Tuple[float, float, float]:
    """Set-F1 with bidirectional substring containment.

    For Recall: each gold atom is "covered" if it is a substring of any pred
    atom OR vice versa. Precision is symmetric over pred atoms.
    """
    p = _safe_set(pred_atoms)
    g = _safe_set(gold_atoms)
    if not p and not g:
        return 1.0, 1.0, 1.0
    if not p or not g:
        return 0.0, 0.0, 0.0

    def _match(a, others):
        return any((a in o) or (o in a) for o in others)

    tp_recall = sum(1 for x in g if _match(x, p))
    tp_prec = sum(1 for x in p if _match(x, g))
    if tp_recall == 0 and tp_prec == 0:
        return 0.0, 0.0, 0.0
    recall = tp_recall / len(g)
    precision = tp_prec / len(p)
    if precision + recall == 0:
        return 0.0, 0.0, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def mean_trace_substring_set_f1(pred_atoms_list, gold_atoms_list) -> float:
    if not pred_atoms_list:
        return 0.0
    return sum(trace_substring_set_f1(p, g)[2]
                for p, g in zip(pred_atoms_list, gold_atoms_list)) \
           / len(pred_atoms_list)


def hard_trace_f1_dispatcher(source: str):
    """Pick the right hard-family F1 function for a dataset source name.

    FLD  → exact set-F1 (atoms are discrete fact_i / int_i IDs).
    FEVER → substring set-F1 (atoms are normalized evidence-sentence strings).
    """
    s = (source or "").upper()
    if s.startswith("FEVER"):
        return mean_trace_substring_set_f1
    return mean_trace_set_f1


# ============================================================
# TRACE layer — Family B: SOFT F1 (BERTScore over free text)
#   Used for ARC (gold = WorldTree fact strings concatenated) and MedQA
#   (gold = MedReason CoT prose).
#
# BERTScore is loaded lazily — `bert_score` is a heavy dependency (torch +
# transformers). When the package is unavailable or both inputs are empty
# the metric returns 0.0; when only one side is empty we return 0.0 by
# convention to match the hard-F1 boundary handling.
# ============================================================

_BERTSCORE_MODEL: Optional[str] = "roberta-large"  # default; rescale_with_baseline=True


def _resolve_device() -> Optional[str]:
    """Pick the best torch device available.

    Apple-silicon MacBook → "mps", CUDA host → "cuda", otherwise "cpu".
    Returns None when torch isn't importable so the caller can let bert_score
    fall through to its own default detection.
    """
    try:
        import torch
    except ImportError:
        return None
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _bertscore_pair(cands: Sequence[str], refs: Sequence[str]) -> List[float]:
    """Batch BERTScore-F1 between aligned (cand, ref) pairs.

    Returns an empty list if `bert_score` is not installed (callers should
    treat this as "soft F1 unavailable" rather than 0.0).
    """
    try:
        from bert_score import score as bs_score
    except ImportError:
        return []
    if not cands:
        return []
    # bert_score runs in a single batch; rescale_with_baseline maps random
    # baseline to ~0 instead of ~0.85 (much more interpretable).
    P, R, F = bs_score(
        cands, refs,
        model_type=_BERTSCORE_MODEL,
        lang="en",
        rescale_with_baseline=True,
        verbose=False,
        device=_resolve_device(),
    )
    return [float(x) for x in F.tolist()]


def trace_bertscore_f1(pred_text: str, gold_text: str) -> float:
    """Single-pair BERTScore-F1. Returns 0.0 if either side is empty or
    bert_score is not installed."""
    if not pred_text or not gold_text:
        return 0.0
    out = _bertscore_pair([pred_text], [gold_text])
    return out[0] if out else 0.0


def mean_trace_bertscore_f1(pred_texts: Sequence[str],
                             gold_texts: Sequence[str]) -> float:
    """Macro-averaged BERTScore-F1 over aligned (pred, gold) text pairs.

    Empty pred OR empty gold → 0.0 for that sample.
    Returns 0.0 (and logs a warning at the import site) if `bert_score` is
    not installed.
    """
    if not pred_texts:
        return 0.0
    # Filter out pairs where either side is empty before calling BERTScore;
    # they contribute 0.0 to the mean by convention.
    pairs = [
        (i, p, g)
        for i, (p, g) in enumerate(zip(pred_texts, gold_texts))
        if p and g
    ]
    if not pairs:
        return 0.0
    cands = [p for _, p, _ in pairs]
    refs  = [g for _, _, g in pairs]
    f1s = _bertscore_pair(cands, refs)
    if not f1s:
        return 0.0
    # Sum non-empty F1s; empty-pair contributions are 0.
    return sum(f1s) / len(pred_texts)


# ============================================================
# Trace family resolution: which F1 to use for each dataset.
# ============================================================

HARD_F1_FAMILIES = {"FLD", "FEVER"}
SOFT_F1_FAMILIES = {"ARC", "MedQA", "FOLIO"}


def trace_family(source: str) -> str:
    """Return "hard" / "soft" / "none" for a dataset source name.

    "hard" → discrete set-F1 (FLD, FEVER)
    "soft" → BERTScore-F1   (ARC, MedQA, FOLIO)
    "none" → no trace metric available for this dataset
    """
    if not source:
        return "none"
    s = source.upper()
    if s.startswith("FLD") or s.startswith("FEVER"):
        return "hard"
    if s.startswith("ARC") or s.startswith("MEDQA") or s.startswith("FOLIO"):
        return "soft"
    return "none"


# ============================================================
# Backward-compat shims (kept for now; older code may still import).
# `mean_trace_jaccard` is no longer reported in summaries — set-F1 alone
# carries the discrete-trace signal.
# ============================================================

def trace_f1(pred_atoms, gold_atoms) -> Tuple[float, float, float]:
    return trace_set_f1(pred_atoms, gold_atoms)


def mean_trace_f1(pred_atoms_list, gold_atoms_list) -> float:
    return mean_trace_set_f1(pred_atoms_list, gold_atoms_list)


def accuracy(preds, answer_idxs):
    return label_acc(preds, answer_idxs)


# ============================================================
# S9 — truly-Unknown subset (the mirror image of Abs Rate)
#
# Computed on genuinely-Unknown samples only (``answer_idx == -1``, filtered
# by the runner). Here, choosing "Unknown" is the *correct* behaviour, so
# ``correct_abstention_rate`` equals accuracy on that subset. Figure 7 (right)
# contrasts it with Abs Rate on the answerable subset.
# ============================================================

def correct_abstention_rate(preds: Sequence[str]) -> float:
    """Fraction of predictions that abstain, on truly-Unknown samples.

    The higher, the better the model recognises that no determinable answer
    exists. Paper Figure 7 (right) plots this against Abs Rate on answerable
    items to show the bias is directional rather than indiscriminate.
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


#: pre-rename alias for :func:`correct_abstention_rate`.
car = correct_abstention_rate


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


#: pre-rename alias (Abs Rate = Abstention Inflation Rate) for :func:`abs_rate`.
abs_rate = abs_rate

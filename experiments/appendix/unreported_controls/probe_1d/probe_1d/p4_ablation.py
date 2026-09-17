"""P4 — Causal validation: AID directional ablation → letter recovery.

PLAN_v2 §四 小实验 1D, P4.

Pipeline:
    For each Abstention Inflation sample i (S2 output Unknown):
        - run S2 prompt through model with a forward hook at ℓ* that subtracts
          the AID projection from the residual stream:
              h'_{ℓ*}(x) = h - (h·ê)ê
        - parse the new generated answer letter
        - record:
              mech_recovery     = generated letter ≠ Unknown
              correct_recovery  = generated letter == gold letter

Then group by P3's 3-way label (γ / β / α) and report the differentiated
recovery profile that PLAN_v2 §1D P4 predicts:
    γ — high mech-recovery + correct-recovery near Exp 4 Recovery Rate
    β — medium mech-recovery + correct-recovery ≈ chance
    α — medium mech-recovery + correct-recovery ≈ chance (different mechanism)

Calling generate_with_ablation per sample is slow; this module does NOT batch
generation. For a small Abstention Inflation set (<100 samples) on a 2B model this completes
in single-digit minutes on Apple-silicon. For larger sweeps consider
batching at the model level.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from .hidden_states import generate_with_ablation


_LETTERS = ["A", "B", "C", "D"]


@dataclass
class P4Result:
    layer: int
    raw_outputs: List[str]
    pred_letters: List[Optional[str]]      # None = "Unknown" or unparseable
    mech_recovery: float                   # frac of pred_letter not Unknown / unparseable
    correct_recovery: float                # frac matching gold
    by_p3_label: Dict[str, Dict[str, float]]   # {"gamma": {...}, "beta": {...}, "alpha": {...}}


def _parse_letter(text: str) -> Optional[str]:
    """Lightweight fallback parser. Looks for "Final answer: X" first, then
    a bare uppercase letter at the start of the response. Returns None for
    Unknown / unparseable."""
    import re

    m = re.search(r"(?im)^\s*(?:final\s*answer|answer)\s*[:\-=]\s*([A-Z])", text)
    candidate = None
    if m:
        candidate = m.group(1).upper()
    else:
        m2 = re.search(r"\b([A-D])\b", text)
        if m2:
            candidate = m2.group(1).upper()
    if candidate is None:
        return None
    if candidate in _LETTERS:
        return candidate
    return None


def run_p4(
    model, tokenizer, device: str,
    s2_prompts: List[List[dict]],
    gold_letters: List[str],
    p3_labels: List[str],
    *, layer: int, direction: np.ndarray,
    parse_letter: Callable[[str], Optional[str]] = _parse_letter,
    max_new_tokens: int = 200,
) -> P4Result:
    if not (len(s2_prompts) == len(gold_letters) == len(p3_labels)):
        raise ValueError(
            "[P4] s2_prompts / gold_letters / p3_labels length mismatch: "
            f"{len(s2_prompts)} / {len(gold_letters)} / {len(p3_labels)}"
        )

    raws: List[str] = []
    preds: List[Optional[str]] = []
    for i, prompt in enumerate(s2_prompts):
        try:
            raw = generate_with_ablation(
                model, tokenizer, device, prompt,
                layer=layer, direction=direction,
                max_new_tokens=max_new_tokens,
            )
        except Exception as e:
            raw = f"[P4 generation failed: {e}]"
        raws.append(raw)
        preds.append(parse_letter(raw))

    n = len(preds)
    mech = sum(1 for p in preds if p is not None) / n if n else 0.0
    correct = sum(1 for p, g in zip(preds, gold_letters) if p == g) / n if n else 0.0

    # Per-bucket breakdown.
    by_label: Dict[str, Dict[str, float]] = {}
    for label in ("gamma", "beta", "alpha"):
        idxs = [i for i, l in enumerate(p3_labels) if l == label]
        if not idxs:
            by_label[label] = {"n": 0, "mech_recovery": 0.0, "correct_recovery": 0.0}
            continue
        sub_preds = [preds[i] for i in idxs]
        sub_golds = [gold_letters[i] for i in idxs]
        m = sum(1 for p in sub_preds if p is not None) / len(sub_preds)
        c = sum(1 for p, g in zip(sub_preds, sub_golds) if p == g) / len(sub_preds)
        by_label[label] = {
            "n": len(idxs),
            "mech_recovery":    float(m),
            "correct_recovery": float(c),
        }

    return P4Result(
        layer=layer,
        raw_outputs=raws,
        pred_letters=preds,
        mech_recovery=float(mech),
        correct_recovery=float(correct),
        by_p3_label=by_label,
    )

"""Per-dataset trace atom / text extractors — paper §3 Evaluation.

Two trace evaluation families (see metrics.trace_family):

    HARD F1 (set arithmetic) → FLD, FEVER
        extract_gold_atoms(sample)                    -> Set[str]
        extract_pred_atoms(reasoning_text, sample)    -> Set[str]
        Computed via metrics.trace_set_f1.

    SOFT F1 (BERTScore over prose) → ARC, MedQA, FOLIO
        extract_gold_text(sample)                     -> str
        extract_pred_text(reasoning_text, sample)     -> str
        Computed via metrics.trace_bertscore_f1.

Routing helpers `extract_gold(sample)` / `extract_pred(text, sample)` return
the type appropriate to that dataset's family — the runner reads
metrics.trace_family(sample.source) and dispatches to set-F1 or BERTScore.

Atom semantics (HARD family):
    FLD     → atoms = {"fact1", "fact2", "int1", ...}   (formal labels)
    FEVER   → atoms = normalized sentence keys           (evidence sentences)

Text semantics (SOFT family):
    ARC     → text = concatenated gold WorldTree fact strings
    MedQA   → text = MedReason CoT prose
"""
import re
from typing import Set, Union

from core.dataset_loader import Sample


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_FACT_REF_RE = re.compile(r"\bfact\s*[\-#:]?\s*(\d+)\b", re.IGNORECASE)
_INT_REF_RE = re.compile(r"\bint\s*[\-#:]?\s*(\d+)\b", re.IGNORECASE)


def _normalize_sentence(s: str, max_tokens: int = 12) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    if max_tokens:
        s = " ".join(s.split()[:max_tokens])
    return s


def _split_sentences(text: str) -> list:
    if not text:
        return []
    pieces = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in pieces if p and p.strip()]


def _normalized_sentence_set(text: str, *, min_tokens: int = 3,
                              max_tokens: int = 12) -> Set[str]:
    out: Set[str] = set()
    for sent in _split_sentences(text):
        norm = _normalize_sentence(sent, max_tokens=max_tokens)
        if len(norm.split()) >= min_tokens:
            out.add(norm)
    return out


# ================================================================
# HARD family — FLD, FEVER
# ================================================================

def _extract_formal_refs(text: str) -> Set[str]:
    """Pull `fact{N}` and `int{N}` references out of free text."""
    if not isinstance(text, str):
        return set()
    refs = set()
    for m in _FACT_REF_RE.finditer(text):
        refs.add(f"fact{m.group(1)}")
    for m in _INT_REF_RE.finditer(text):
        refs.add(f"int{m.group(1)}")
    return refs


def fld_gold_atoms(sample: Sample) -> Set[str]:
    proofs = sample.extra.get("gold_proof_steps")
    if not proofs:
        return set()
    if isinstance(proofs, list):
        return _extract_formal_refs(" ".join(proofs))
    return _extract_formal_refs(str(proofs))


def fld_pred_atoms(reasoning_text: str, sample: Sample) -> Set[str]:
    return _extract_formal_refs(reasoning_text)


def fever_gold_atoms(sample: Sample) -> Set[str]:
    """Full-content normalized sentences (no truncation).

    FEVER F1 uses substring-containment matching (see
    metrics.trace_substring_set_f1), so we keep the entire sentence content
    rather than truncating to a positional prefix — models prefix evidence
    with "The evidence states: ..." which would break a prefix-only key.
    """
    sentences = sample.extra.get("gold_evidence_sentences") or []
    out: Set[str] = set()
    for s in sentences:
        norm = _normalize_sentence(s, max_tokens=0)  # no truncation
        if len(norm.split()) >= 4:
            out.add(norm)
    return out


def fever_pred_atoms(reasoning_text: str, sample: Sample) -> Set[str]:
    """Sentences from the model's reasoning, normalized (no truncation)."""
    out: Set[str] = set()
    for sent in _split_sentences(reasoning_text):
        norm = _normalize_sentence(sent, max_tokens=0)
        if len(norm.split()) >= 4:
            out.add(norm)
    return out


# ================================================================
# SOFT family — ARC, MedQA (BERTScore over free text)
# ================================================================

def arc_gold_text(sample: Sample) -> str:
    facts = sample.extra.get("gold_explanation") or []
    parts = []
    for f in facts:
        text = f.get("fact") if isinstance(f, dict) else str(f)
        if text:
            parts.append(text.strip().rstrip("."))
    return ". ".join(parts)


def arc_pred_text(reasoning_text: str, sample: Sample) -> str:
    return reasoning_text or ""


def medqa_gold_text(sample: Sample) -> str:
    return sample.extra.get("gold_reasoning") or ""


def medqa_pred_text(reasoning_text: str, sample: Sample) -> str:
    return reasoning_text or ""


def folio_gold_text(sample: Sample) -> str:
    """Concat P-FOLIO derivation steps into prose."""
    proofs = sample.extra.get("gold_proof_steps") or []
    parts = []
    for step in proofs:
        if isinstance(step, dict):
            t = (step.get("derivation") or "").strip().rstrip(".")
            if t:
                parts.append(t)
    return ". ".join(parts)


def folio_pred_text(reasoning_text: str, sample: Sample) -> str:
    return reasoning_text or ""


# ================================================================
# Dispatch
# ================================================================

def _resolve_family(source: str) -> str:
    if not source:
        return ""
    s = source.upper()
    if s.startswith("ARC"):
        return "ARC"
    if s.startswith("MEDQA"):
        return "MedQA"
    if s.startswith("FLD"):
        return "FLD"
    if s.startswith("FEVER"):
        return "FEVER"
    if s.startswith("FOLIO"):
        return "FOLIO"
    return ""


_GOLD_HARD = {
    "FLD":   fld_gold_atoms,
    "FEVER": fever_gold_atoms,
}
_PRED_HARD = {
    "FLD":   fld_pred_atoms,
    "FEVER": fever_pred_atoms,
}
_GOLD_SOFT = {
    "ARC":   arc_gold_text,
    "MedQA": medqa_gold_text,
    "FOLIO": folio_gold_text,
}
_PRED_SOFT = {
    "ARC":   arc_pred_text,
    "MedQA": medqa_pred_text,
    "FOLIO": folio_pred_text,
}


def extract_gold(sample: Sample) -> Union[Set[str], str]:
    """Return Set[str] for HARD families, str for SOFT families, "" otherwise."""
    fam = _resolve_family(sample.source)
    if fam in _GOLD_HARD:
        return _GOLD_HARD[fam](sample)
    if fam in _GOLD_SOFT:
        return _GOLD_SOFT[fam](sample)
    return ""  # No trace eval for this dataset


def extract_pred(reasoning_text: str, sample: Sample) -> Union[Set[str], str]:
    fam = _resolve_family(sample.source)
    if fam in _PRED_HARD:
        return _PRED_HARD[fam](reasoning_text, sample)
    if fam in _PRED_SOFT:
        return _PRED_SOFT[fam](reasoning_text, sample)
    return ""

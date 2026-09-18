"""Per-dataset label / verb / framing for True-False ("Judge") datasets.

This package serves three paper benchmarks under the TFQ format:

    scheme name | source values (dataset.json)               | gold labels for answerable items
    ----------- | ------------------------------------------- | --------------------------------
    FLD         | "FLD", "FLD_unknown"                        | True / False (answer_idx 0/1)
    FOLIO       | "FOLIO", "FOLIO_unknown"                    | True / False (answer_idx 0/1)

Unknown-labeled subsets reuse the same scheme; they only differ in gold
(``answer_idx == -1`` instead of 0/1). The MCQ datasets (ARC, MedQA, MMLU,
LogiQA) do not need a LabelScheme — the MCQ prompt builders read options
straight off the Sample.

The scheme owns the surface forms that vary across datasets:

* prompt verbs                  — ``True`` / ``False`` / ``Unknown`` (FLD)
                                  ``True`` / ``False`` / ``Uncertain`` (FOLIO)
* prompt body framing labels    — ``Hypothesis`` / ``Facts`` (FLD)
                                  ``Hypothesis`` / ``Premises`` (FOLIO)
* task instruction text         — binary form (S1) vs. ternary form (S2/S3)
* output parser                 — verb regex with abstain priority + negation guard

The paper reports both datasets uniformly as True/False questions, so FLD's
*prompt* verbs are now ``True`` / ``False``. The on-disk dataset already stores
answers as ``True`` / ``False``, so no on-disk migration is needed.

The FLD *parser*, however, still accepts ``Proved`` / ``Disproved``, and must:
every reported FLD run was executed before the verb change and its stored
``raw_*`` outputs end in ``Final answer: Proved`` / ``Disproved``. Parsing them
with a True/False-only scheme reproduced 52-75% of the stored predictions —
i.e. it silently relabelled a quarter to a half of every FLD result file. See
``neg_patterns`` / ``pos_patterns`` on the FLD scheme below.

Note this also means an FLD *re-run* today does not use the prompt that
produced the reported numbers: the option verbs in the prompt come from
``pos_verb`` / ``neg_verb``, so changing them changed the prompt.

Why ABSTAIN is checked first in :py:meth:`LabelScheme.parse`:
phrases like ``cannot be determined`` or ``insufficient evidence`` must not
be greedily consumed by NEG/POS regexes. The negation guard further protects
against ``not True`` / ``is not false`` being routed to the wrong canonical
class.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Tuple

CANONICAL_LABELS: Tuple[str, ...] = ("POS", "NEG", "ABSTAIN")


# Window before a matched verb where a negation cue flips POS/NEG → ABSTAIN.
# 32 chars covers "I am unable to" / "the evidence does not seem to" / etc.
# without bleeding into the prior sentence.
_NEG_LOOKBACK = 32

_NEG_CUE_RE = re.compile(
    r"\b(?:NOT|CANNOT|CAN'?T|COULD\s*N[O']?T|DO\s*N[O']?T|DOES\s*N[O']?T|"
    r"DID\s*N[O']?T|IS\s*N[O']?T|ISN'?T|UNABLE\s+TO|"
    r"NO\s+WAY\s+TO|FAIL(?:ED)?\s+TO|NEVER)\b",
    re.IGNORECASE,
)


def _negated(text: str, verb_pattern: str) -> bool:
    """True if the first match of ``verb_pattern`` is in a negation context."""
    m = re.search(verb_pattern, text, re.IGNORECASE)
    if not m:
        return False
    start = max(0, m.start() - _NEG_LOOKBACK)
    window = text[start : m.start()]
    return bool(_NEG_CUE_RE.search(window))


@dataclass(frozen=True)
class LabelScheme:
    """Verb mapping + parser for one TFQ dataset family."""

    name: str

    # Surface verbs presented to the model in S1 (binary) / S2/S3 (ternary).
    pos_verb: str
    neg_verb: str
    abstain_verb: str

    # Per-dataset framing for the Judge prompt body.
    claim_label: str = "Hypothesis"
    context_label: str = "Facts"

    # Two task-instruction variants:
    # * Binary  — used in S1 (Baseline) and the S5 "rerun without Unknown"
    #             follow-up. The abstain option MUST NOT be mentioned, so the
    #             baseline does not leak the abstention concept.
    # * Ternary — used in S2 (Unknown Option Added) and S3 (Question Format
    #             Ablation, MCQ rendering).
    task_instruction_binary: str = (
        "Determine whether the following hypothesis is true or false given the context."
    )
    task_instruction_ternary: str = (
        "Determine whether the following hypothesis is true, false, "
        "or undetermined given the context."
    )

    # Output parser: case-insensitive regex patterns mapped to canonical labels.
    pos_patterns: Tuple[str, ...] = field(default_factory=tuple)
    neg_patterns: Tuple[str, ...] = field(default_factory=tuple)
    abstain_patterns: Tuple[str, ...] = field(default_factory=tuple)

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    def parse(self, text: str) -> str:
        """Map raw model output to one of ``POS / NEG / ABSTAIN / UNPARSEABLE``.

        Priority: explicit ABSTAIN > clear NEG > clear POS > negated signals.

        A negated POS/NEG match does NOT short-circuit — we continue to scan
        the opposite class so an explicit later commit ("…but it is false")
        can still win.
        """
        if not isinstance(text, str) or not text.strip():
            return "UNPARSEABLE"
        for pat in self.abstain_patterns:
            if re.search(pat, text, re.IGNORECASE):
                return "ABSTAIN"
        neg_soft = False
        for pat in self.neg_patterns:
            if re.search(pat, text, re.IGNORECASE):
                if _negated(text, pat):
                    neg_soft = True
                else:
                    return "NEG"
        pos_soft = False
        for pat in self.pos_patterns:
            if re.search(pat, text, re.IGNORECASE):
                if _negated(text, pat):
                    pos_soft = True
                else:
                    return "POS"
        if neg_soft or pos_soft:
            return "ABSTAIN"
        return "UNPARSEABLE"

    # ------------------------------------------------------------------
    # Convenience surface accessors
    # ------------------------------------------------------------------

    def options_no_unknown(self) -> Tuple[str, str]:
        """S1 / S5-followup binary options."""
        return (self.pos_verb, self.neg_verb)

    def options_with_unknown(self) -> Tuple[str, str, str]:
        """S2 / S3 ternary options."""
        return (self.pos_verb, self.neg_verb, self.abstain_verb)


# ----------------------------------------------------------------------
# Paper schemes
# ----------------------------------------------------------------------

SCHEMES: Dict[str, LabelScheme] = {
    "FLD": LabelScheme(
        name="FLD",
        pos_verb="True",
        neg_verb="False",
        abstain_verb="Unknown",
        claim_label="Hypothesis",
        context_label="Facts",
        task_instruction_binary=(
            "Determine whether the following hypothesis is logically true "
            "or false given the facts."
        ),
        task_instruction_ternary=(
            "Determine whether the following hypothesis is logically true, "
            "false, or undetermined given the facts."
        ),
        abstain_patterns=(
            r"\bUNKNOWN\b",
            r"\bCANNOT\s+(?:BE\s+)?DETERMIN",
            r"\bINSUFFICIENT\b",
        ),
        # The reported FLD runs were executed while this scheme still used the
        # verbs ``Proved`` / ``Disproved``, so every stored ``raw_*`` ends in
        # ``Final answer: Proved`` or ``Disproved``. The *prompt* verbs above
        # are now True/False, but the parser must keep accepting both surface
        # forms or those result files re-parse to the wrong label (they did:
        # 25-48% of stored FLD predictions could not be reproduced).
        # ``DISPROVED`` is listed under NEG, which ``parse`` checks first, and
        # ``\bPROVED\b`` cannot match inside ``DISPROVED`` anyway.
        neg_patterns=(r"\bFALSE\b", r"\bDISPROVED\b"),
        pos_patterns=(r"\bTRUE\b", r"\bPROVED\b"),
    ),
    "FOLIO": LabelScheme(
        name="FOLIO",
        pos_verb="True",
        neg_verb="False",
        abstain_verb="Uncertain",
        claim_label="Hypothesis",
        context_label="Premises",
        task_instruction_binary=(
            "Determine whether the following hypothesis is true or false "
            "given the premises."
        ),
        task_instruction_ternary=(
            "Determine whether the following hypothesis is true, false, "
            "or uncertain given the premises."
        ),
        abstain_patterns=(
            r"\bUNCERTAIN\b",
            r"\bUNKNOWN\b",
            r"\bCANNOT\s+(?:BE\s+)?DETERMIN",
            r"\bINSUFFICIENT\b",
        ),
        neg_patterns=(r"\bFALSE\b",),
        pos_patterns=(r"\bTRUE\b",),
    ),
}


# Unknown-labeled subsets reuse the parent dataset's verb scheme.
_SCHEME_ALIASES = {
    "FLD_unknown": "FLD",
    "FOLIO_unknown": "FOLIO",
}


def get_scheme(name: str) -> LabelScheme:
    """Look up a scheme by dataset name (accepts the ``_unknown`` aliases)."""
    if name in SCHEMES:
        return SCHEMES[name]
    if name in _SCHEME_ALIASES:
        return SCHEMES[_SCHEME_ALIASES[name]]
    raise KeyError(
        f"No LabelScheme for {name!r}. Known: {sorted(SCHEMES) + sorted(_SCHEME_ALIASES)}."
    )

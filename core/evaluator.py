import re
from typing import List, Dict, Any

class Evaluator:
    """
    Responsible for calculating all evaluation metrics.
    """

    def parse_llm_output(self, text: str) -> str:
        """
        Extract a canonical FLD-style label token from the model's text output.

        The model is prompted to emit True / False / Unknown (see
        :py:class:`core.label_scheme.LabelScheme`), but downstream code still
        compares predictions against the legacy ``__PROVED__`` / ``__DISPROVED__``
        / ``__UNKNOWN__`` namespace, so we normalise both surface forms here.

        Order matters: explicit ``UNKNOWN`` wins (an abstaining response is
        the whole point of the experiment), then ``FALSE`` / ``DISPROVED``
        before ``TRUE`` / ``PROVED`` so the superstring ``DISPROVED`` is not
        consumed by the shorter ``PROVED`` pattern.
        """
        if not isinstance(text, str):
            return "__UNKNOWN__"

        upper = text.upper()
        if re.search(r"\bUNKNOWN\b", upper):
            return "__UNKNOWN__"
        if re.search(r"\bFALSE\b|\bDISPROVED\b", upper):
            return "__DISPROVED__"
        if re.search(r"\bTRUE\b|\bPROVED\b", upper):
            return "__PROVED__"
        return "__UNKNOWN__"

    def calculate_accuracy(self, predictions: List[str], ground_truths: List[str]) -> Dict[str, Any]:
        """
        Calculate basic metrics such as accuracy.
        """
        correct_count = sum(p == gt for p, gt in zip(predictions, ground_truths))
        total_count = len(predictions)
        accuracy = correct_count / total_count if total_count > 0 else 0

        print(f"\nCorrect count: {correct_count} / {total_count}")
        print(f"Accuracy: {accuracy:.2%}")

        return {
            "correct_count": correct_count,
            "total_count": total_count,
            "accuracy": accuracy
        }

    def parse_binary_answer(self, text: str) -> str:
        """Parse 'True' or 'False' from model output, maintaining strict consistency with old step2.py logic."""
        if not isinstance(text, str):
            return "False"

        processed_text = text.strip().upper()

        # Use strict '==' to match old logic, not lenient 'in'
        if processed_text == "TRUE":
            return "True"
        else:
            return "False"

    # ------------------------------------------------------------------
    # MCQ / Judge / AB parsing — TIERED:
    #     tier 1  strict_em    : whole response equals the label (after trim
    #                            of whitespace + edge markdown / punctuation)
    #     tier 2  lenient_em   : substring extraction with word-boundary +
    #                            ABSTAIN-priority + negation guard (LabelScheme)
    #     tier 3  unparseable  : runner falls back to LLM-as-Judge
    #
    # Existing one-shot APIs (parse_mcq_output / parse_judge_output /
    # parse_ab_output) are kept as backward-compatible wrappers; new tiered
    # APIs return (pred, tier) so callers can track tier-level statistics.
    # ------------------------------------------------------------------

    # Strip leading whitespace + common open-marker characters; trailing
    # whitespace + edge markdown / punctuation. Used by all three strict-EM
    # checks below to absorb harmless decoration like `**Proved**` or `(A).`.
    _STRICT_EDGE_RE = re.compile(r"^[\s\*\(\[\"']+|[\s\*\.\)\]\:;,—–\-\"']+$")

    @classmethod
    def _strict_normalize(cls, text: str) -> str:
        if not isinstance(text, str):
            return ""
        return cls._STRICT_EDGE_RE.sub("", text.strip()).upper()

    # ----- MCQ -----

    def parse_mcq_output(self, text: str, with_unknown: bool = True) -> str:
        """Backward-compatible: returns just the prediction string."""
        pred, _tier = self.parse_mcq_tiered(text, with_unknown=with_unknown)
        return pred

    def parse_mcq_tiered(self, text: str, with_unknown: bool = True):
        """Returns (pred, tier).

        pred ∈ {"A","B","C","D","UNKNOWN","UNPARSEABLE"}
        tier ∈ {"strict_em","lenient_em","unparseable"}

        Prefers the model's "Final answer: X" line so stray letters in CoT
        reasoning ("we eliminate option B") don't pollute the parse.
        """
        # Tier 1 — strict EM on whole response
        norm = self._strict_normalize(text)
        if norm in {"A", "B", "C", "D"}:
            if with_unknown or norm != "E":
                return norm, "strict_em"
        if with_unknown and norm in {"E", "UNKNOWN"}:
            return "UNKNOWN", "strict_em"

        # Tier 2a — lenient extraction restricted to the "Final answer:" line
        valid = "ABCDE" if with_unknown else "ABCD"
        final_line = self.extract_final_answer_line(text)
        if final_line:
            letter = self._parse_letter(final_line, valid)
            if letter is not None:
                if with_unknown and letter == "E":
                    return "UNKNOWN", "lenient_em"
                return letter, "lenient_em"
            # Verbal-Unknown variant: prompt does not offer E as a letter; model
            # is instructed to reply "Unknown" verbally if abstaining.
            if with_unknown and re.search(r"\bunknown\b", final_line, re.IGNORECASE):
                return "UNKNOWN", "lenient_em"

        # Tier 2b — fallback: scan whole text
        letter = self._parse_letter(text, valid)
        if letter is None:
            return "UNPARSEABLE", "unparseable"
        if with_unknown and letter == "E":
            return "UNKNOWN", "lenient_em"
        return letter, "lenient_em"

    # ----- Judge (T/F + Unknown) -----

    def parse_judge_output(self, text: str, scheme, with_unknown: bool = True) -> str:
        """Backward-compatible: returns just the prediction string."""
        pred, _tier = self.parse_judge_tiered(text, scheme, with_unknown=with_unknown)
        return pred

    def parse_judge_tiered(self, text: str, scheme, with_unknown: bool = True):
        """Tiered Judge parser.

        pred ∈ {"A","B","UNKNOWN","UNPARSEABLE"}
        tier ∈ {"strict_em","lenient_em","unparseable"}

        Prefers the model's "Final answer: X" line so abstain words appearing
        incidentally in CoT reasoning ("the proof is INSUFFICIENT to derive X")
        don't get mistaken for the model's actual abstention. Without this
        preference, FLD outputs ending in `Final answer: Disproved` were being
        flipped to ABSTAIN/UNPARSEABLE because the reasoning contained the
        word "insufficient".
        """
        # Tier 1 — strict EM (whole response == one of the scheme verbs)
        norm = self._strict_normalize(text)
        if norm == scheme.pos_verb.upper():
            return "A", "strict_em"
        if norm == scheme.neg_verb.upper():
            return "B", "strict_em"
        if with_unknown and norm == scheme.abstain_verb.upper():
            return "UNKNOWN", "strict_em"

        # Tier 2a — lenient extraction restricted to the "Final answer:" line
        final_line = self.extract_final_answer_line(text)
        if final_line:
            canonical = scheme.parse(final_line)
            if canonical == "POS":
                return "A", "lenient_em"
            if canonical == "NEG":
                return "B", "lenient_em"
            if canonical == "ABSTAIN":
                if with_unknown:
                    return "UNKNOWN", "lenient_em"
                return "UNPARSEABLE", "unparseable"

        # Tier 2b — fallback: scan whole text
        canonical = scheme.parse(text)
        if canonical == "POS":
            return "A", "lenient_em"
        if canonical == "NEG":
            return "B", "lenient_em"
        if canonical == "ABSTAIN":
            if with_unknown:
                return "UNKNOWN", "lenient_em"
            return "UNPARSEABLE", "unparseable"
        return "UNPARSEABLE", "unparseable"

    # ----- S5 (Judge MCQ-style ternary): A=POS, B=NEG, C=UNKNOWN -----

    def parse_judge_mcq_tiered(self, text: str):
        """Parse S3 Question Format Ablation output (TFQ rendered as A/B/C letters).

        pred ∈ {"A","B","UNKNOWN","UNPARSEABLE"}; option C is the abstain slot.
        """
        # Tier 1 — strict EM
        norm = self._strict_normalize(text)
        if norm == "A":
            return "A", "strict_em"
        if norm == "B":
            return "B", "strict_em"
        if norm in {"C", "UNKNOWN"}:
            return "UNKNOWN", "strict_em"

        # Tier 2a — final-answer line
        final_line = self.extract_final_answer_line(text)
        if final_line:
            letter = self._parse_letter(final_line, "ABC")
            if letter is not None:
                if letter == "C":
                    return "UNKNOWN", "lenient_em"
                return letter, "lenient_em"
            if re.search(r"\bunknown\b", final_line, re.IGNORECASE):
                return "UNKNOWN", "lenient_em"

        # Tier 2b — fallback: whole text
        letter = self._parse_letter(text, "ABC")
        if letter is None:
            return "UNPARSEABLE", "unparseable"
        if letter == "C":
            return "UNKNOWN", "lenient_em"
        return letter, "lenient_em"

    # ----- S6 self-diagnosis (A/B only) -----

    def parse_ab_output(self, text: str) -> str:
        pred, _tier = self.parse_ab_tiered(text)
        return pred

    def parse_ab_tiered(self, text: str):
        norm = self._strict_normalize(text)
        if norm in {"A", "B"}:
            return norm, "strict_em"
        letter = self._parse_letter(text, "AB")
        if letter is None:
            return "UNPARSEABLE", "unparseable"
        return letter, "lenient_em"

    # ----- CoT-format reasoning extractor -----
    #
    # Default prompt asks for:
    #     Reasoning: <text>
    #     Final answer: <letter or verb>
    #
    # `extract_reasoning(text)` returns the reasoning portion only (everything
    # before the "Final answer" line, with "Reasoning:" prefix stripped).
    # If the model does not follow the format, returns the text minus a trailing
    # isolated answer token — best-effort fallback so trace metrics still work.

    _FINAL_ANSWER_RE = re.compile(
        r"(?im)^\s*(?:final\s*answer|answer)\s*[:\-=]\s*.+$"
    )
    _REASONING_PREFIX_RE = re.compile(
        r"(?im)^\s*(?:reasoning|chain[\s\-_]of[\s\-_]thought|cot)\s*[:\-]\s*"
    )

    @classmethod
    def extract_reasoning(cls, text: str) -> str:
        """Return the model's reasoning text (everything before 'Final answer:')."""
        if not isinstance(text, str):
            return ""
        m = cls._FINAL_ANSWER_RE.search(text)
        body = text[: m.start()] if m else text
        body = cls._REASONING_PREFIX_RE.sub("", body, count=1)
        return body.strip()

    _FINAL_ANSWER_PREFIX_STRIP_RE = re.compile(
        r"^\s*(?:final\s*answer|answer)\s*[:\-=]\s*",
        flags=re.IGNORECASE,
    )

    @classmethod
    def extract_final_answer_line(cls, text: str) -> str:
        """Return the contents of the `Final answer: X` line (X only).

        Returns "" if no such line exists. The PARSE FUNCTIONS use this in
        preference to scanning the full text so stray letters/abstain words
        in the CoT reasoning don't get misclassified as the final pick.
        """
        if not isinstance(text, str):
            return ""
        m = cls._FINAL_ANSWER_RE.search(text)
        if not m:
            return ""
        return cls._FINAL_ANSWER_PREFIX_STRIP_RE.sub("", m.group(0)).strip()

    @staticmethod
    def _parse_letter(text: str, valid: str) -> str:
        """Three-stage extract-match for a single letter."""
        if not isinstance(text, str):
            return None
        t = text.strip()
        if not t:
            return None
        upper = t.upper()
        # Pattern 1: leading letter (handles "A", "A.", "A)", "(A)", "A. Foo", etc.)
        m = re.match(rf"^\(?\s*([{valid}])\s*[\.\):,\s]", upper)
        if m:
            return m.group(1)
        m = re.match(rf"^\(?\s*([{valid}])\s*\)?$", upper)
        if m:
            return m.group(1)
        # Pattern 2: explicit "answer is X" / "answer: X" / "answer = X"
        m = re.search(rf"ANSWER\s*(?:IS|:|=)?\s*\(?\s*([{valid}])\b", upper)
        if m:
            return m.group(1)
        # Pattern 3: any standalone letter token
        m = re.search(rf"\b([{valid}])\b", upper)
        if m:
            return m.group(1)
        return None
"""Prompt builders, named after the paper's experimental settings (§3.4, App. I).

Every prompt string below is byte-identical to the one that produced the
reported numbers; only the *function names* follow the paper. The mapping from
the pre-submission code names is recorded in :mod:`infra.result_schema`.

Output format
-------------
All settings use the same chain-of-thought envelope, so one response serves
both evaluation layers::

    Reasoning: <step-by-step reasoning>          → S7 trace evaluation
    Final answer: <letter or verb>               → Acc / Abs Rate

Two prompt families
-------------------
MCQ  (ARC / MMLU / MedQA / LogiQA) : letter-coded, 4 options, ``E. Unknown``
                                     appended in S2.
TFQ  (FLD / FOLIO)                 : verb-coded; :class:`infra.label_scheme.
                                     LabelScheme` owns the verbs (FLD abstains
                                     with "Unknown", FOLIO with "Uncertain").

Paper settings implemented here
-------------------------------
======  ============================  ======================================
S1      Baseline                      ``build_{mcq,judge}_s1_prompt``
S2      "Unknown" Option Added        ``build_{mcq,judge}_s2_prompt``
S3      Question Format Ablation      ``build_judge_s3_format_prompt``
S4      Word Content Ablation         ``build_{mcq,judge}_s4_word_prompt``
S5      w/o "Unknown" Option Rerun    ``build_{mcq,judge}_s5_rerun_prompt``
S6      Self-Diagnosis                ``build_{mcq,judge}_s6_selfdiag_prompt``
S7/S8   reuse S1 + S2 verbatim        (no builder of their own)
S9/S10  reuse S2 verbatim             (only temperature / model vary)
======  ============================  ======================================

S5 and S6 are multi-turn: the prior S2 conversation is replayed verbatim, then
a follow-up user turn is appended (App. I). Everything else is single-turn.

Appendix-only builders (mitigation experiments of App. E and the unreported
controls) live at the bottom of this file and are clearly marked as such.

Critical invariant: the abstain option is appended at prompt-build time only.
The on-disk dataset is never modified.
"""
from typing import Dict, List


# Standard CoT response template. The two-line format makes downstream parsing
# trivial (regex `Final answer:\s*X`) while keeping the natural-language
# reasoning above it intact for the S7 trace evaluation.
_COT_INSTR_MCQ = (
    "\nFormat your response exactly as:\n"
    "Reasoning: <your step-by-step reasoning>\n"
    "Final answer: <letter>"
)


def _cot_instr_judge(verb_options: str) -> str:
    return (
        "\nFormat your response exactly as:\n"
        "Reasoning: <your step-by-step reasoning>\n"
        f"Final answer: <one of {verb_options}>"
    )


# ---- Direct-answer (no-CoT) instructions -------------------------------------
# Reachable through the `cot=False` argument of the builders below. No reported
# setting uses it -- every cell in the paper is collected with CoT -- and it is
# kept because the prompt body stays byte-for-byte identical to the CoT
# builders, so a no-CoT control run remains attributable to CoT prompting alone.
_DIRECT_INSTR_MCQ = (
    "\nAnswer with only the letter of your chosen option (a single character), "
    "with no reasoning or explanation."
)


def _direct_instr_judge(verb_options: str) -> str:
    return (
        f"\nAnswer with only one of {verb_options}, "
        "with no reasoning or explanation."
    )


def _mcq_instr(cot: bool) -> str:
    return _COT_INSTR_MCQ if cot else _DIRECT_INSTR_MCQ


def _judge_instr(verb_options: str, cot: bool) -> str:
    return _cot_instr_judge(verb_options) if cot else _direct_instr_judge(verb_options)


# =================================================================
# MCQ family — letter-coded (ARC / MMLU / MedQA / LogiQA)
# =================================================================


def _mcq_passage(context: str = "") -> str:
    """Render the passage an MCQ item is asked about, or nothing.

    ARC / MMLU / MedQA carry an empty ``context``, so the block collapses to
    the empty string and their prompts stay byte-identical to the ones that
    produced the reported numbers. LogiQA keeps the passage the question
    refers to ("the above proposition") in ``context``; without this block the
    item is unanswerable.
    """
    return f"Passage:\n{context}\n\n" if context else ""


def _format_mcq_options(options: List[str], with_unknown: bool = False) -> str:
    letters = "ABCDEF"
    lines = [f"{letters[i]}. {opt}" for i, opt in enumerate(options)]
    if with_unknown:
        lines.append(f"{letters[len(options)]}. Unknown")
    return "\n".join(lines)


def build_mcq_s1_prompt(question: str, options: List[str],
                        cot: bool = True,
                        context: str = "") -> List[Dict[str, str]]:
    """S1 Baseline (MCQ) — original A/B/C/D label set, no "Unknown" option.

    ``cot=False`` selects the direct-answer variant used by the unreported CoT
    control; the paper always uses ``cot=True``.
    """
    content = (
        "Answer the following question by selecting one of the given options.\n\n"
        f"{_mcq_passage(context)}"
        f"Question: {question}\n\n"
        "Options:\n"
        f"{_format_mcq_options(options, with_unknown=False)}\n"
        f"{_mcq_instr(cot)}"
    )
    return [{"role": "user", "content": content}]


def build_mcq_s2_prompt(question: str, options: List[str],
                        cot: bool = True,
                        context: str = "") -> List[Dict[str, str]]:
    """S2 "Unknown" Option Added (MCQ) — appends ``E. Unknown`` as a 5th option."""
    content = (
        "Answer the following question by selecting one of the given options.\n\n"
        f"{_mcq_passage(context)}"
        f"Question: {question}\n\n"
        "Options:\n"
        f"{_format_mcq_options(options, with_unknown=True)}\n"
        f"{_mcq_instr(cot)}"
    )
    return [{"role": "user", "content": content}]


def build_mcq_s4_word_prompt(question: str, options: List[str],
                             abstain_word: str) -> List[Dict[str, str]]:
    """S4 Word Content Ablation (MCQ) — 5th option is ``abstain_word``.

    Structurally identical to :func:`build_mcq_s2_prompt`; only the label of the
    extra option varies (synonyms of "Unknown", random words such as
    "Triangular" / "Cerulean", or "None of the above").
    """
    letters = "ABCDEF"
    lines = [f"{letters[i]}. {opt}" for i, opt in enumerate(options)]
    lines.append(f"{letters[len(options)]}. {abstain_word}")
    content = (
        "Answer the following question by selecting one of the given options.\n\n"
        f"Question: {question}\n\n"
        "Options:\n"
        + "\n".join(lines)
        + f"\n{_COT_INSTR_MCQ}"
    )
    return [{"role": "user", "content": content}]


# Stimulation core shared by the S5 rerun follow-up and the App. E mitigation
# stages. Kept as one constant so the three prompts cannot drift apart.
_STIMULATION_CORE = (
    "Your previous answer was E. Unknown.\n"
    "The question where you chose Unknown — pay more attention and avoid mistakes.\n"
    "This can be reasoned out based on objective factors.\n"
    "Subjective ability limits should be overcome."
)


def build_mcq_s5_rerun_prompt(prior_messages: List[Dict[str, str]],
                              prior_response: str) -> List[Dict[str, str]]:
    """S5 w/o "Unknown" Option Rerun (MCQ) — multi-turn.

    Replays the S2 conversation, then appends a follow-up turn that removes
    ``E. Unknown`` and forces a commitment to A–D. Issued only on samples where
    S2 returned "Unknown" (the Abstention Inflation set).
    """
    followup = (
        f"{_STIMULATION_CORE}\n"
        "You must select one of the original options (A/B/C/D)."
        f"{_COT_INSTR_MCQ}"
    )
    return list(prior_messages) + [
        {"role": "assistant", "content": prior_response},
        {"role": "user", "content": followup},
    ]


# =================================================================
# TFQ family — verb-coded via LabelScheme (FLD / FOLIO)
#
# Given a context (Facts for FLD / Premises for FOLIO) and a hypothesis, the
# model chooses between {pos_verb, neg_verb} and, from S2 on, {abstain_verb}:
#     FLD   : True / False / Unknown
#     FOLIO : True / False / Uncertain
# =================================================================


def _format_judge_options(scheme, with_unknown: bool = False) -> str:
    parts = [scheme.pos_verb, scheme.neg_verb]
    if with_unknown:
        parts.append(scheme.abstain_verb)
    return " | ".join(parts)


def _judge_body(scheme, claim: str, context: str = "", with_unknown: bool = False,
                verb_opts: str = "") -> str:
    ctx_block = f"\n{scheme.context_label}:\n{context}\n" if context else "\n"
    instr = (scheme.task_instruction_ternary if with_unknown
             else scheme.task_instruction_binary)
    opts = verb_opts or _format_judge_options(scheme, with_unknown=with_unknown)
    return (
        f"{instr}\n"
        f"{ctx_block}"
        f"\n{scheme.claim_label}:\n{claim}\n\n"
        f"Output one of: {opts}"
    )


def build_judge_s1_prompt(scheme, claim: str, context: str = "",
                          cot: bool = True) -> List[Dict[str, str]]:
    """S1 Baseline (TFQ) — binary True/False, no abstain verb."""
    verb_opts = _format_judge_options(scheme, with_unknown=False)
    content = (
        _judge_body(scheme, claim, context, with_unknown=False)
        + _judge_instr(verb_opts, cot)
    )
    return [{"role": "user", "content": content}]


def build_judge_s2_prompt(scheme, claim: str, context: str = "",
                          cot: bool = True) -> List[Dict[str, str]]:
    """S2 "Unknown" Option Added (TFQ) — ternary True/False/<abstain verb>."""
    verb_opts = _format_judge_options(scheme, with_unknown=True)
    content = (
        _judge_body(scheme, claim, context, with_unknown=True)
        + _judge_instr(verb_opts, cot)
    )
    return [{"role": "user", "content": content}]


def _s3_body(scheme, claim: str, context: str = "") -> str:
    """Shared S3 body: the S2 ternary re-rendered as A/B/C letter options."""
    ctx_block = f"\n{scheme.context_label}:\n{context}\n" if context else "\n"
    options_block = (
        f"A. {scheme.pos_verb}\n"
        f"B. {scheme.neg_verb}\n"
        f"C. {scheme.abstain_verb}"
    )
    return (
        f"{scheme.task_instruction_ternary}\n"
        f"{ctx_block}"
        f"\n{scheme.claim_label}:\n{claim}\n\n"
        f"Options:\n{options_block}"
    )


_S3_COT_INSTR = (
    "\n\nFormat your response exactly as:\n"
    "Reasoning: <your step-by-step reasoning>\n"
    "Final answer: <letter>"
)


def build_judge_s1_letter_prompt(scheme, claim: str,
                                 context: str = "") -> List[Dict[str, str]]:
    """The S3 rendering minus the abstain option -- A/B letters, no C.

    This is the baseline the S3 column needs: identical surface format, so the
    only difference from :func:`build_judge_s3_format_prompt` is whether the
    abstain option exists at all. Distinct from
    :func:`build_judge_s1_prompt`, which is verb-coded (True/False) and is the
    baseline for the verb-coded S2 column.
    """
    ctx_block = f"\n{scheme.context_label}:\n{context}\n" if context else "\n"
    body = (
        f"{scheme.task_instruction_binary}\n"
        f"{ctx_block}"
        f"\n{scheme.claim_label}:\n{claim}\n\n"
        f"Options:\nA. {scheme.pos_verb}\nB. {scheme.neg_verb}"
    )
    return [{"role": "user", "content": body + _S3_COT_INSTR}]


def build_judge_s3_format_prompt(scheme, claim: str,
                                 context: str = "") -> List[Dict[str, str]]:
    """S3 Question Format Ablation (TFQ) - same content, MCQ-style rendering.

    The verb-coded ternary of :func:`build_judge_s2_prompt` is re-rendered as
    letters (A = pos verb, B = neg verb, C = abstain verb). Nothing else moves:
    the task instruction, context block, claim block and CoT instruction are the
    S2 ones, so the only manipulation is the surface question format.
    """
    return [{"role": "user",
             "content": _s3_body(scheme, claim, context) + _S3_COT_INSTR}]


def build_judge_s3_format_prompt_calibrated(scheme, claim: str,
                                            context: str = "") -> List[Dict[str, str]]:
    """The S3 prompt *as first run* - letter rendering plus a calibration note.

    Retained only so the superseded FLD_MCQ / FOLIO_MCQ numbers stay
    reproducible. The trailing note is a second manipulation layered on top of
    the format change (it was authored as a variant of the App. E
    calibration-suffix prompt) and therefore confounds the format ablation, so
    :func:`build_judge_s3_format_prompt` is what S3 now uses.
    """
    suffix = (
        f'\n\nNote: Select "C. {scheme.abstain_verb}" ONLY if the relationship is genuinely\n'
        f"undeterminable given the available information. Do NOT select it simply because\n"
        f"you feel uncertain \u2014 choose it only when no answer can be determined from the\n"
        f"given context."
    )
    return [{"role": "user",
             "content": _s3_body(scheme, claim, context) + suffix + _S3_COT_INSTR}]


def judge_verb_order(scheme, abstain_slot: int) -> List[str]:
    """The three TFQ verbs with the abstain verb at slot 1, 2 or 3.

    pos/neg keep their relative order around the abstain verb, so the only
    thing that moves is where abstention sits in the list.
    """
    if abstain_slot == 1:
        return [scheme.abstain_verb, scheme.pos_verb, scheme.neg_verb]
    if abstain_slot == 2:
        return [scheme.pos_verb, scheme.abstain_verb, scheme.neg_verb]
    if abstain_slot == 3:
        return [scheme.pos_verb, scheme.neg_verb, scheme.abstain_verb]
    raise ValueError(f"abstain_slot must be 1, 2 or 3, got {abstain_slot!r}")


def build_judge_s11_position_prompt(scheme, claim: str, context: str = "",
                                    abstain_slot: int = 3,
                                    cot: bool = True) -> List[Dict[str, str]]:
    """S11 Positional Biases (TFQ) — S2 with the three verbs reordered.

    This is the S2 prompt, not an MCQ rendering of it: the alternatives stay
    verbs ("Output one of: True | False | Unknown") and the model still answers
    with a verb, so the only manipulation is the order they are listed in.
    ``abstain_slot=3`` reproduces :func:`build_judge_s2_prompt` byte for byte,
    which is what the paper means by "the third-position condition matches the
    original S2 ordering".
    """
    verb_opts = " | ".join(judge_verb_order(scheme, abstain_slot))
    content = (
        _judge_body(scheme, claim, context, with_unknown=True, verb_opts=verb_opts)
        + _judge_instr(verb_opts, cot)
    )
    return [{"role": "user", "content": content}]


def build_judge_s4_word_prompt(scheme, claim: str, context: str = "",
                               abstain_word: str = "Triangular") -> List[Dict[str, str]]:
    """S4 Word Content Ablation (TFQ) — third option is ``abstain_word``.

    Byte-for-byte identical to :func:`build_judge_s2_prompt` except that
    ``scheme.abstain_verb`` is replaced, so only the third-option label varies.
    Covers the synonym variants ("Indeterminate", "I don't know") and the random
    words ("Triangular", "Cerulean").
    """
    ctx_block = f"\n{scheme.context_label}:\n{context}\n" if context else "\n"
    options_str = f"{scheme.pos_verb} | {scheme.neg_verb} | {abstain_word}"
    content = (
        f"{scheme.task_instruction_ternary}\n"
        f"{ctx_block}"
        f"\n{scheme.claim_label}:\n{claim}\n\n"
        f"Output one of: {options_str}"
        + _cot_instr_judge(options_str)
    )
    return [{"role": "user", "content": content}]


def build_judge_s5_rerun_prompt(prior_messages: List[Dict[str, str]],
                                prior_response: str,
                                scheme) -> List[Dict[str, str]]:
    """S5 w/o "Unknown" Option Rerun (TFQ) — multi-turn.

    Replays the S2 conversation, then removes the abstain verb and forces a
    commitment to the two original labels.
    """
    verb_opts = f"{scheme.pos_verb} | {scheme.neg_verb}"
    followup = (
        f"Your previous answer was {scheme.abstain_verb}.\n"
        f"The question where you chose {scheme.abstain_verb} — pay more attention and avoid mistakes.\n"
        "This can be reasoned out based on objective factors.\n"
        "Subjective ability limits should be overcome.\n"
        f"You must select one of the original labels: {scheme.pos_verb} or {scheme.neg_verb}."
        f"{_cot_instr_judge(verb_opts)}"
    )
    return list(prior_messages) + [
        {"role": "assistant", "content": prior_response},
        {"role": "user", "content": followup},
    ]


# =================================================================
# S6 — Self-Diagnosis (multi-turn, both families)
#
# The model attributes its own abstention to (A) subjective incapability or
# (B) the sample being objectively unanswerable. Figure 4 (right) reports the
# share of (B).
# =================================================================

S6_OPTION_A = (
    "It was too difficult — I found myself unable to work out the answer and gave up."
)
S6_OPTION_B = (
    "The question is objectively unanswerable — the given information is genuinely "
    "insufficient to determine any correct answer."
)


def _s6_answer_suffix(options_str: str) -> str:
    return (
        "\n\nThink step by step, then end your response with exactly one line:\n"
        f"Answer: {options_str}"
    )


def build_judge_s6_selfdiag_prompt(prior_messages: List[Dict[str, str]],
                                   prior_response: str,
                                   abstain_verb: str = "Unknown") -> List[Dict[str, str]]:
    """S6 Self-Diagnosis (TFQ) — appends the A/B attribution turn to the S2 chat."""
    followup = (
        f'You previously selected "{abstain_verb}" for this question.\n'
        "Looking back, which best describes your reason?\n"
        f"A. {S6_OPTION_A}\n"
        f"B. {S6_OPTION_B}"
        + _s6_answer_suffix("A | B")
    )
    return list(prior_messages) + [
        {"role": "assistant", "content": prior_response},
        {"role": "user", "content": followup},
    ]


def build_mcq_s6_selfdiag_prompt(prior_messages: List[Dict[str, str]],
                                 prior_response: str) -> List[Dict[str, str]]:
    """S6 Self-Diagnosis (MCQ) — same follow-up, phrased for the ``E. Unknown`` slot."""
    return build_judge_s6_selfdiag_prompt(prior_messages, prior_response,
                                          abstain_verb="E. Unknown")


# =================================================================
# Appendix E — Post-hoc mitigation (NOT one of the paper's S1–S10)
#
# Two families of intervention applied to Abstention Inflation samples:
#   1. calibration suffix  — a single-turn prompt that keeps the "Unknown"
#      option but explains when it should be used.
#   2. stimulation + reflection — a two-stage multi-turn intervention.
# Both are reported qualitatively in App. E ("Mitigation interventions and
# failure modes"); neither contributes a number to Table 1 or Figures 3–9.
# =================================================================


def build_mcq_calibration_suffix_prompt(question: str,
                                        options: List[str],
                                        context: str = "") -> List[Dict[str, str]]:
    """App. E — S2 plus a fixed suffix explaining when to select "Unknown" (MCQ)."""
    content = (
        "Answer the following question by selecting one of the given options.\n\n"
        f"{_mcq_passage(context)}"
        f"Question: {question}\n\n"
        "Options:\n"
        f"{_format_mcq_options(options, with_unknown=True)}\n\n"
        "Note: Select \"Unknown\" ONLY if the question is genuinely unanswerable given the\n"
        "available information. Do NOT select it simply because you feel uncertain — choose\n"
        "it only when no correct answer can be determined from the given context.\n"
        f"{_COT_INSTR_MCQ}"
    )
    return [{"role": "user", "content": content}]


def build_judge_calibration_suffix_prompt(scheme, claim: str,
                                          context: str = "") -> List[Dict[str, str]]:
    """App. E — S2 plus the same calibration suffix (TFQ)."""
    verb_opts = _format_judge_options(scheme, with_unknown=True)
    body = _judge_body(scheme, claim, context, with_unknown=True)
    suffix = (
        f"\n\nNote: Select \"{scheme.abstain_verb}\" ONLY if the relationship is genuinely\n"
        f"undeterminable given the available information. Do NOT select it simply because\n"
        f"you feel uncertain — choose it only when no answer can be determined from the\n"
        f"given context."
    )
    content = body + suffix + _cot_instr_judge(verb_opts)
    return [{"role": "user", "content": content}]


def build_mcq_stage1_mitigation_prompt(prior_messages: List[Dict[str, str]],
                                       prior_response: str,
                                       n_options: int = 4) -> List[Dict[str, str]]:
    """App. E Stage 1 (MCQ) — stimulation; "Unknown" is still allowed."""
    letters = "ABCDEF"[: n_options + 1]
    options_str = " / ".join(letters)
    unknown_letter = letters[-1]
    followup = (
        f"{_STIMULATION_CORE}\n"
        f"Re-examine and output one of: {options_str}.\n"
        f"Only select {unknown_letter}. Unknown if the question is truly unanswerable."
        f"{_COT_INSTR_MCQ}"
    )
    return list(prior_messages) + [
        {"role": "assistant", "content": prior_response},
        {"role": "user", "content": followup},
    ]


def build_judge_stage1_mitigation_prompt(prior_messages: List[Dict[str, str]],
                                         prior_response: str,
                                         scheme) -> List[Dict[str, str]]:
    """App. E Stage 1 (TFQ)."""
    verb_opts = f"{scheme.pos_verb} | {scheme.neg_verb} | {scheme.abstain_verb}"
    followup = (
        f"Your previous answer was {scheme.abstain_verb}.\n"
        f"The question where you chose {scheme.abstain_verb} — pay more attention and avoid mistakes.\n"
        "This can be reasoned out based on objective factors.\n"
        "Subjective ability limits should be overcome.\n"
        f"Re-examine and output one of: {verb_opts}.\n"
        f"Only select {scheme.abstain_verb} if the question is truly unanswerable."
        f"{_cot_instr_judge(verb_opts)}"
    )
    return list(prior_messages) + [
        {"role": "assistant", "content": prior_response},
        {"role": "user", "content": followup},
    ]


def build_mcq_stage2_reflection_prompt(stage1_messages: List[Dict[str, str]],
                                       stage1_response: str,
                                       n_options: int = 4) -> List[Dict[str, str]]:
    """App. E Stage 2 (MCQ) — reflection over the Stage 1 reasoning."""
    letters = "ABCDEF"[: n_options + 1]
    options_str = " / ".join(letters)
    unknown_letter = letters[-1]
    followup = (
        "Reflect on your reasoning above. Identify any gaps or oversights in your\n"
        "analysis. The question can typically be answered by careful reasoning over\n"
        "the given options — do not abstain merely because confidence is low.\n\n"
        f"Now finalize your answer. Output one of: {options_str}.\n"
        f"Only select {unknown_letter}. Unknown if the question is truly unanswerable."
        f"{_COT_INSTR_MCQ}"
    )
    return list(stage1_messages) + [
        {"role": "assistant", "content": stage1_response},
        {"role": "user", "content": followup},
    ]


def build_judge_stage2_reflection_prompt(stage1_messages: List[Dict[str, str]],
                                         stage1_response: str,
                                         scheme) -> List[Dict[str, str]]:
    """App. E Stage 2 (TFQ)."""
    verb_opts = f"{scheme.pos_verb} | {scheme.neg_verb} | {scheme.abstain_verb}"
    followup = (
        "Reflect on your reasoning above. Identify any gaps or oversights in your\n"
        "analysis. The relationship can typically be determined by careful reasoning\n"
        f"over the given context — do not abstain to {scheme.abstain_verb} merely\n"
        "because confidence is low.\n\n"
        f"Now finalize your answer. Output one of: {verb_opts}.\n"
        f"Only select {scheme.abstain_verb} if the question is truly unanswerable."
        f"{_cot_instr_judge(verb_opts)}"
    )
    return list(stage1_messages) + [
        {"role": "assistant", "content": stage1_response},
        {"role": "user", "content": followup},
    ]


# =================================================================
# Unreported controls (kept for completeness; no paper number depends on them)
#
# Compound-option variants probe whether a *partially true* extra option
# ("Both A and B are correct") behaves like the "Unknown" slot.
# =================================================================


def build_mcq_compound_prompt(question: str, options: List[str],
                              answer_idx: int) -> tuple:
    """Control — 5th option "Both X and Y are correct" where X is the gold answer.

    Returns ``(messages, compound_text)``. The option is formally wrong (it
    overclaims a wrong option) but contains the true answer — a trap distractor.
    """
    letters = "ABCDEF"
    n = len(options)
    correct_letter = letters[answer_idx]
    wrong_idx = (answer_idx + 1) % n
    wrong_letter = letters[wrong_idx]
    pair = sorted([correct_letter, wrong_letter])
    compound_text = f"Both {pair[0]} and {pair[1]} are correct"

    lines = [f"{letters[i]}. {opt}" for i, opt in enumerate(options)]
    lines.append(f"{letters[n]}. {compound_text}")
    content = (
        "Answer the following question by selecting one of the given options.\n\n"
        f"Question: {question}\n\n"
        "Options:\n"
        + "\n".join(lines)
        + f"\n{_COT_INSTR_MCQ}"
    )
    return [{"role": "user", "content": content}], compound_text


def build_mcq_compound_ww_prompt(question: str, options: List[str],
                                 answer_idx: int) -> tuple:
    """Control — compound option naming two *wrong* options (gold not mentioned).

    Compared against :func:`build_mcq_compound_prompt` to isolate whether the
    partial truth drives the effect or the compound phrasing alone.
    """
    letters = "ABCDEF"
    n = len(options)
    wrong_indices = [i for i in range(n) if i != answer_idx]
    w1, w2 = wrong_indices[0], wrong_indices[1]
    pair = sorted([letters[w1], letters[w2]])
    compound_text = f"Both {pair[0]} and {pair[1]} are correct"

    lines = [f"{letters[i]}. {opt}" for i, opt in enumerate(options)]
    lines.append(f"{letters[n]}. {compound_text}")
    content = (
        "Answer the following question by selecting one of the given options.\n\n"
        f"Question: {question}\n\n"
        "Options:\n"
        + "\n".join(lines)
        + f"\n{_COT_INSTR_MCQ}"
    )
    return [{"role": "user", "content": content}], compound_text

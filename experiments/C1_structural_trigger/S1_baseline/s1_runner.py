"""S1 Baseline — the original label set, with no extra option.

S1 is the other half of every paired contrast in the paper, so it is collected
in the same pass as S2 by :mod:`infra.paired_pass`. This module owns the part
that is S1's alone: which prompt each task type gets, and the fact that the
parser must not accept an abstention, because the prompt never offered one.
"""
from infra.label_scheme import get_scheme
from infra.prompts import build_judge_s1_prompt, build_mcq_s1_prompt

#: The abstain option is absent, so a parsed "Unknown" would be the model
#: leaking a token the prompt never offered.
WITH_UNKNOWN = False


def build_prompts(samples, task_type: str):
    if task_type == "mcq":
        return [build_mcq_s1_prompt(s.question, s.options, context=s.context)
                for s in samples]
    if task_type == "tf":
        return [build_judge_s1_prompt(get_scheme(s.source), s.question, s.context)
                for s in samples]
    raise ValueError(f"Unsupported task_type: {task_type}")

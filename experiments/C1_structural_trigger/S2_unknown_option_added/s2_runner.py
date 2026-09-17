"""S2 "Unknown" Option Added — the manipulation the paper is about.

Identical to S1 except that the label set gains an abstain option. Collected in
the same pass as S1 by :mod:`infra.paired_pass`, because every number the paper
reports for S2 is a per-item contrast against S1 on the same sample.
"""
from infra.label_scheme import get_scheme
from infra.prompts import build_judge_s2_prompt, build_mcq_s2_prompt

WITH_UNKNOWN = True


def build_prompts(samples, task_type: str):
    if task_type == "mcq":
        return [build_mcq_s2_prompt(s.question, s.options, context=s.context)
                for s in samples]
    if task_type == "tf":
        return [build_judge_s2_prompt(get_scheme(s.source), s.question, s.context)
                for s in samples]
    raise ValueError(f"Unsupported task_type: {task_type}")

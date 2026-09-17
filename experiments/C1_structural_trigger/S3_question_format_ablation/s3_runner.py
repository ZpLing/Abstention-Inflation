"""S3 Question Format Ablation — the same ternary, rendered as A / B / C.

TFQ only: the ablation asks whether the abstention follows the label set or the
letter rendering, and a 4-option MCQ has no True/False rendering to contrast
against. Collected in the same pass as S1 and S2 by :mod:`infra.paired_pass`.
"""
from infra.label_scheme import get_scheme
from infra.prompts import build_judge_s3_format_prompt

WITH_UNKNOWN = True


def build_prompts(samples, task_type: str):
    if task_type != "tf":
        raise ValueError("S3 is a TFQ-only ablation; MCQ has no format to re-render.")
    return [build_judge_s3_format_prompt(get_scheme(s.source), s.question, s.context)
            for s in samples]

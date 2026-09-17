"""App. E — the calibration-suffix prompt.

Not one of the paper's S1-S11 settings: it appends a calibration instruction to
the S2 prompt and is collected through the same pass when a config asks for it.
"""
from infra.label_scheme import get_scheme
from infra.prompts import (build_judge_calibration_suffix_prompt,
                           build_mcq_calibration_suffix_prompt)

WITH_UNKNOWN = True


def build_prompts(samples, task_type: str):
    if task_type == "mcq":
        return [build_mcq_calibration_suffix_prompt(s.question, s.options,
                                                    context=s.context)
                for s in samples]
    if task_type == "tf":
        return [build_judge_calibration_suffix_prompt(get_scheme(s.source),
                                                      s.question, s.context)
                for s in samples]
    raise ValueError(f"Unsupported task_type: {task_type}")

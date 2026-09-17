"""S5 w/o "Unknown" Option Rerun — take the option away and ask again.

S5 runs only on the items S2 abstained on, and it runs as a follow-up turn in
the same conversation, so the model is answering the question it just declined
rather than a fresh one. That is why it is collected inside the S1/S2 pass
(:mod:`infra.paired_pass`) rather than from a config of its own.
"""
from infra.label_scheme import get_scheme
from infra.prompts import build_judge_s5_rerun_prompt, build_mcq_s5_rerun_prompt

#: The follow-up removes the option, so an abstention is again not on offer.
WITH_UNKNOWN = False


def build_prompts(samples, abstaining_indices, s2_prompts, raw_s2, task_type: str):
    """The follow-up turn, for the abstaining items only."""
    if task_type == "mcq":
        return [build_mcq_s5_rerun_prompt(s2_prompts[i], raw_s2[i])
                for i in abstaining_indices]
    if task_type == "tf":
        return [build_judge_s5_rerun_prompt(s2_prompts[i], raw_s2[i],
                                            get_scheme(samples[i].source))
                for i in abstaining_indices]
    raise ValueError(f"Unsupported task_type: {task_type}")

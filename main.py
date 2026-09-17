"""Unified dispatcher for the Abstention Inflation experiments.

Usage::

    python main.py --config configs/<group>/<config>.yaml

The YAML's ``run_tasks`` list selects the runner(s). Each task name maps to one
of the paper's settings (see the root README for the full paper↔code map):

    main_experiment      S1 / S2 / S3 + the S5 rerun on abstaining samples
    s6_self_diagnosis    S6 self-diagnosis follow-up (multi-turn)
    s9_truly_unknown     S9 perception on truly-Unknown samples
    s10_model_sweep      S10 alignment & model-size sweep (Gemma, Qwen)
    appendix_mitigation  App. E post-hoc stimulation+reflection baseline

Settings that are not driven from a YAML — S4 word content ablation, S7 trace
evaluation, S8 logit-lens probe, S9 persistence, S10 temperature / difficulty —
have their own scripts under ``experiments/Cx_*/Sy_*/``; each folder has a
README with the exact command.

Task names used before the paper's S1–S10 numbering are still accepted; they
print a notice pointing at the new name.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure ``core`` is importable when invoked as ``python main.py``.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.config_loader import load_config
from core.data_handler import DataHandler
from core.evaluator import Evaluator
from core.llm_handler import LLMHandler

#: pre-rename run_tasks name -> current name.
LEGACY_TASK_ALIASES = {
    "ab_experiment": "main_experiment",
    "supplementary_experiment": "s9_truly_unknown",
    "s5_supplementary": "s6_self_diagnosis",
    "exp2_model_sweep": "s10_model_sweep",
    "exp4_mitigation": "appendix_mitigation",
}


def _resolve_tasks(config: dict) -> list:
    tasks = []
    for raw in config.get("run_tasks", []) or []:
        if raw in LEGACY_TASK_ALIASES:
            new = LEGACY_TASK_ALIASES[raw]
            print(f"[main] run_tasks: {raw!r} is the pre-rename name; running {new!r}.")
            tasks.append(new)
        else:
            tasks.append(raw)
    return tasks


async def _dispatch(config: dict) -> None:
    data_handler = DataHandler(config)
    llm_handler = LLMHandler(config)
    evaluator = Evaluator()

    tasks = _resolve_tasks(config)
    if not tasks:
        print("[main] run_tasks is empty — nothing to do.")
        return

    if "main_experiment" in tasks:
        print("\n===== S1 / S2 / S3 + S5 rerun =====")
        from core.runners.ab_runner import ABRunner
        await ABRunner(config, data_handler, llm_handler, evaluator).run()

    if "s9_truly_unknown" in tasks:
        print("\n===== S9 perception on truly-Unknown samples =====")
        from core.runners.s9_truly_unknown_runner import TrulyUnknownRunner
        await TrulyUnknownRunner(config, data_handler, llm_handler, evaluator).run()

    if "s6_self_diagnosis" in tasks:
        print("\n===== S6 self-diagnosis =====")
        from core.runners.s6_self_diagnosis_runner import S6SelfDiagnosisRunner
        await S6SelfDiagnosisRunner(config, data_handler, llm_handler, evaluator).run()

    if "s10_model_sweep" in tasks:
        print("\n===== S10 alignment & model-size sweep =====")
        from core.runners.s10_model_sweep_runner import ModelSweepRunner
        await ModelSweepRunner(config, data_handler, llm_handler, evaluator).run()

    if "appendix_mitigation" in tasks:
        print("\n===== App. E post-hoc mitigation =====")
        from core.runners.appendix_mitigation_runner import PostHocMitigationRunner
        await PostHocMitigationRunner(config, data_handler, llm_handler, evaluator).run()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dispatcher for the Abstention Inflation experiments."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/C1_structural_trigger/S1_S3_TFQ_GPT_5_4_nano.yaml",
        help="Path to the experiment YAML.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    try:
        asyncio.run(_dispatch(config))
    except KeyboardInterrupt:
        print("\n[main] interrupted by user.")


if __name__ == "__main__":
    main()

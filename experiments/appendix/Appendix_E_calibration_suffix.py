"""App. E — the calibration-suffix prompt.

Not one of the paper's S1-S11 settings: it appends a calibration instruction to
the S2 prompt and is collected through the same pass when a config asks for it.
"""
import sys
from pathlib import Path

# Importable as a module and runnable as a file.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

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


def main() -> None:
    """Collect the calibration-suffix condition from a config."""
    import argparse
    import asyncio

    from loader.config_loader import block_key, load_config
    from loader.data_handler import DataHandler
    from infra.evaluator import Evaluator
    from infra.llm_handler import LLMHandler
    from infra.paired_pass import ABRunner

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, help="Experiment YAML.")
    args = ap.parse_args()

    config = load_config(args.config)
    block = config.setdefault(block_key(config, "ab_experiment"), {})
    block.setdefault("settings", ["S1", "S2", "calibration_suffix"])
    asyncio.run(ABRunner(config, DataHandler(config), LLMHandler(config),
                         Evaluator()).run())


if __name__ == "__main__":
    main()

"""S2 "Unknown" Option Added — the manipulation the paper is about.

Identical to S1 except that the label set gains an abstain option. Collected in
the same pass as S1 by :mod:`infra.paired_pass`, because every number the paper
reports for S2 is a per-item contrast against S1 on the same sample.
"""

import sys
from pathlib import Path

# Importable as a module and runnable as a file.
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pathlib import Path

from infra.label_scheme import get_scheme
from infra.prompts import build_judge_s2_prompt, build_mcq_s2_prompt

WITH_UNKNOWN = True


def build_prompts(samples, task_type: str):
    if task_type == "mcq":
        return [
            build_mcq_s2_prompt(s.question, s.options, context=s.context)
            for s in samples
        ]
    if task_type == "tf":
        return [
            build_judge_s2_prompt(get_scheme(s.source), s.question, s.context)
            for s in samples
        ]
    raise ValueError(f"Unsupported task_type: {task_type}")


def main() -> None:
    """Collect this setting from a config. The pass is shared: S2 is scored against S1 per item."""
    import argparse
    import asyncio

    from infra.evaluator import Evaluator
    from infra.llm_handler import LLMHandler
    from infra.paired_pass import ABRunner
    from loader.config_loader import block_key, load_config
    from loader.data_handler import DataHandler

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, help="Experiment YAML.")
    args = ap.parse_args()

    config = load_config(args.config)
    block = config.setdefault(block_key(config, "ab_experiment"), {})
    block.setdefault("settings", ["S1", "S2"])

    asyncio.run(
        ABRunner(config, DataHandler(config), LLMHandler(config), Evaluator()).run()
    )


if __name__ == "__main__":
    main()

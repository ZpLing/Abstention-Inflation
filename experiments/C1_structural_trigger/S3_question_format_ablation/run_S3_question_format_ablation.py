"""S3 Question Format Ablation — the same ternary, rendered as A / B / C.

TFQ only: the ablation asks whether the abstention follows the label set or the
letter rendering, and a 4-option MCQ has no True/False rendering to contrast
against. Collected in the same pass as S1 and S2 by :mod:`infra.paired_pass`.
"""
import sys
from pathlib import Path

# Importable as a module and runnable as a file.
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pathlib import Path
from infra.label_scheme import get_scheme
from infra.prompts import build_judge_s3_format_prompt

WITH_UNKNOWN = True


def build_prompts(samples, task_type: str):
    if task_type != "tf":
        raise ValueError("S3 is a TFQ-only ablation; MCQ has no format to re-render.")
    return [build_judge_s3_format_prompt(get_scheme(s.source), s.question, s.context)
            for s in samples]


def main() -> None:
    """Collect this setting from a config. The pass is shared: S3 is contrasted with S2 per item."""
    import argparse
    import asyncio

    from infra.config_loader import block_key, load_config
    from infra.data_handler import DataHandler
    from infra.evaluator import Evaluator
    from infra.llm_handler import LLMHandler
    from infra.paired_pass import ABRunner

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, help="Experiment YAML.")
    args = ap.parse_args()

    config = load_config(args.config)
    block = config.setdefault(block_key(config, "ab_experiment"), {})
    block.setdefault("settings", ["S1", "S2", "S3"])

    asyncio.run(ABRunner(config, DataHandler(config), LLMHandler(config),
                         Evaluator()).run())


if __name__ == "__main__":
    main()

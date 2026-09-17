"""Entry point for the S9 truly-Unknown perception sub-experiment.

Drives :class:`experiments.C4_stable_bias.S9_stability.s9_truly_unknown_runner.TrulyUnknownRunner` over the
``FLD_unknown`` / ``FOLIO_unknown`` subsets bundled under
``software/dataset/``. The runner runs S1 / S2 / S3 prompts on items whose
gold label is Unknown, so the resulting Abs Rate is the model's correct-rate
on truly-Unknown items (compared against its inflated Abs Rate on
answerable items, this gives the discrimination gap in Figure 6 right).

Usage::

    python experiments/C4_stable_bias/S9_stability/run_S9_truly_unknown.py \\
        --config configs/C4_stable_bias/truly_unknown_<model>.yaml
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from infra.config_loader import load_config
from infra.data_handler import DataHandler
from infra.evaluator import Evaluator
from infra.llm_handler import LLMHandler
from experiments.C4_stable_bias.S9_stability.s9_truly_unknown_runner import TrulyUnknownRunner


async def _run(config_path: str) -> None:
    config = load_config(config_path)
    data_handler = DataHandler(config)
    llm_handler = LLMHandler(config)
    evaluator = Evaluator()
    await TrulyUnknownRunner(config, data_handler, llm_handler, evaluator).run()


def main() -> None:
    parser = argparse.ArgumentParser(description="S9 truly-Unknown perception runner.")
    parser.add_argument("--config", required=True,
                        help="YAML config; relative to software/.")
    args = parser.parse_args()
    asyncio.run(_run(args.config))


if __name__ == "__main__":
    main()

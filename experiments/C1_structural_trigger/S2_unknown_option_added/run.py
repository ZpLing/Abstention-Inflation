"""Entry point for S2 (Unknown Option Added) — paper §C1.

This script is a thin wrapper around the central dispatcher (``main.py``).
The actual runner is :class:`core.runners.ab_runner.ABRunner`; this script just loads the
YAML config you point it at and invokes the runner.

Usage::

    python experiments/C1_structural_trigger/S2_unknown_option_added/run.py \\
        --config configs/C1_structural_trigger/GPT_5_4_nano_FLD.yaml
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from core.runners.ab_runner import ABRunner
from core.config_loader import block_key, load_config
from core.data_handler import DataHandler
from core.evaluator import Evaluator
from core.llm_handler import LLMHandler


async def _run(config_path: str) -> None:
    config = load_config(config_path)
    ab = config.setdefault(block_key(config, "main_experiment"), {})
    if "settings" not in ab:
        ab["settings"] = ["S1", "S2"]

    data_handler = DataHandler(config)
    llm_handler = LLMHandler(config)
    evaluator = Evaluator()
    await ABRunner(config, data_handler, llm_handler, evaluator).run()


def main() -> None:
    parser = argparse.ArgumentParser(description="S2 (Unknown Option Added).")
    parser.add_argument("--config", required=True,
                        help="Path to the YAML config (relative to software/).")
    args = parser.parse_args()
    asyncio.run(_run(args.config))


if __name__ == "__main__":
    main()

"""Entry point for S1 (Baseline) — paper §4.1.

This script is a thin wrapper around the central dispatcher (``main.py``).
The actual runner is :class:`infra.paired_pass.ABRunner`; S1 is selected by
declaring ``ab_experiment.settings: ["S1"]`` in the YAML config (S1 is
also always implicitly run when S2 is enabled, since the paper reports
per-item paired S1↔S2 comparisons).

Usage::

    python experiments/C1_structural_trigger/S1_baseline/run.py \\
        --config configs/C1_structural_trigger/<model>_<dataset>.yaml
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Make ``infra`` importable when this script is run directly from any cwd.
_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from infra.paired_pass import ABRunner
from infra.config_loader import block_key, load_config
from infra.data_handler import DataHandler
from infra.evaluator import Evaluator
from infra.llm_handler import LLMHandler


async def _run(config_path: str) -> None:
    config = load_config(config_path)
    # Force S1-only unless the YAML already restricted the settings list.
    ab = config.setdefault(block_key(config, "main_experiment"), {})
    if "settings" not in ab:
        ab["settings"] = ["S1"]
    data_handler = DataHandler(config)
    llm_handler = LLMHandler(config)
    evaluator = Evaluator()
    await ABRunner(config, data_handler, llm_handler, evaluator).run()


def main() -> None:
    parser = argparse.ArgumentParser(description="S1 Baseline runner.")
    parser.add_argument("--config", required=True,
                        help="Path to the YAML config (relative to software/).")
    args = parser.parse_args()
    asyncio.run(_run(args.config))


if __name__ == "__main__":
    main()

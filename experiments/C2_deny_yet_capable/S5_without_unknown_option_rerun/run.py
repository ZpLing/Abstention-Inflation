"""S5 — w/o "Unknown" Option Rerun (paper §4.2.1, Figure 4 left).

S5 is produced by the same runner as S1/S2: :class:`core.runners.ab_runner.ABRunner`
first runs S2 to find the samples the model abstains on, then replays that
conversation and appends a follow-up turn with the "Unknown" option removed
(``core.prompts.build_{judge,mcq}_s5_rerun_prompt``). There is therefore no
separate S5 pass — enabling ``run_s5_rerun`` on a normal S1/S2 config is all
that is needed, and the results land in the ``s5_rerun`` block of
``ab_summary_<dataset>_<model>.json``.

Usage::

    python experiments/C2_deny_yet_capable/S5_without_unknown_option_rerun/run.py \\
        --config configs/C1_structural_trigger/S1_S3_TFQ_GPT_5_4_nano.yaml

Then aggregate with ``analyze_S5.py`` in this folder.
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
    block = config.setdefault(block_key(config, "main_experiment"), {})
    # S5 needs S2 (to locate the abstaining samples) and S1 (paired baseline).
    block.setdefault("settings", ["S1", "S2"])
    block["run_s5_rerun"] = True
    await ABRunner(config, DataHandler(config), LLMHandler(config), Evaluator()).run()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="S5 — w/o \"Unknown\" Option Rerun (paper §4.2.1)."
    )
    parser.add_argument("--config", required=True, help="Path to the experiment YAML.")
    args = parser.parse_args()
    asyncio.run(_run(args.config))


if __name__ == "__main__":
    main()

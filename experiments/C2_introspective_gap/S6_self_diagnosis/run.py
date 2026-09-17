"""Entry point for S6 (Self-Diagnosis) — paper §C2.

Drives :mod:`core.s6_self_diagnosis_runner`. The runner takes a YAML config
describing model + dataset; see the per-claim configs/ folder for
paper-grade examples.

Usage::

    python experiments/C2_introspective_gap/S6_self_diagnosis/run.py \\
        --config configs/C2_introspective_gap/GPT_5_4_nano_FLD.yaml
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="S6 (Self-Diagnosis) — paper §C2.",
    )
    parser.add_argument("--config", required=True,
                        help="YAML config; relative to software/.")
    args = parser.parse_args()

    # Re-dispatch through main.py so all task wiring stays in one place.
    sys.argv = ["main.py", "--config", args.config]
    runpy.run_path(str(_REPO / "main.py"), run_name="__main__")


if __name__ == "__main__":
    main()

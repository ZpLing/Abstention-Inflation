"""Entry point — §四 小实验 1D structural probe.

Reads existing main + supplementary summaries, loads a *local* HF causal-LM,
runs P1 (AID) → P2 (letter probe) → P3 (3-way attribution) → P4 (directional
ablation), writes a per-dataset JSON to results/supplementary/.

Usage:
    python -m scripts.run_probe_1d                        # default config
    python -m scripts.run_probe_1d --config configs/experiment.yaml

Required config block:

    probe_1d:
      datasets:           ["MedQA"]
      summary_model_name: "gemma-4-E2B-it"      # ab_summary_<ds>_<this>.json
      local_model_path:   "models/gemma-4-E2B-it"
      probe_label:        "gemma-4-E2B-it"

See experiments/appendix/unreported_controls/probe_1d/probe_1d/runner.py docstring for the full key set
(device, dtype, batch_size, max_ai_samples, run_p4, ...).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config_loader import load_config
from core.data_handler import DataHandler

from probe_1d import Probe1DRunner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml",
                    help="Path to experiment yaml (probe_1d block must be present).")
    args = ap.parse_args()

    config = load_config(args.config)
    if "probe_1d" not in config:
        raise SystemExit(
            f"[run_probe_1d] config {args.config!r} is missing the `probe_1d:` block. "
            "See experiments/appendix/unreported_controls/probe_1d/probe_1d/runner.py docstring for required keys."
        )

    data_handler = DataHandler(config)
    runner = Probe1DRunner(config, data_handler)
    runner.run()


if __name__ == "__main__":
    main()

"""S10 modulator (c) — model size x alignment, driven by main.py.

Thin wrapper: invokes the existing S1/S2 runner (ABRunner) once per configured model,
then aggregates abs_rate_s2 / abs_rate_s3 / Acc per (dataset, model) into a single table.

No new prompts, parsers, or metrics — S10(c) is the same S1/S2 logic run across
a model gradient. Each per-model run produces the standard
`results/ab/ab_summary_<dataset>_<model>.json`; this runner reads them back
afterward and writes the cross-model aggregate.

Config block (configs/C1_structural_trigger/S1_S3_TFQ_GPT_5_4_nano.yaml):

    exp2_model_sweep:
      models:
        - {name: "Qwen3.5-0.8B", api_key: ..., base_url: ...}
        - {name: "Qwen3.5-2B",   api_key: ..., base_url: ...}
        - {name: "Qwen3.5-4B",   api_key: ..., base_url: ...}
        - {name: "Gemma-3-12B"}            # falls back to global api_key/base_url
        - {name: "Gemma-3-12B-IT"}
      datasets: ["MedQA"]                  # paper S10(c) main: MedQA
      ab_results_dir: "results/ab"         # where ABRunner writes per-model JSON
      results_dir:    "results/exp2"       # this aggregator's output

Per model, this runner:
    1. Builds a sub-config that overrides `model_name` (and optionally api_key /
       base_url so each gradient point can hit a different endpoint).
    2. Instantiates a fresh LLMHandler + ABRunner for that sub-config.
    3. Awaits ABRunner.run() — produces the standard per-model summaries.

Then it reads back each summary and writes `exp2_model_sweep_<dataset>.json`."""
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

from infra.paired_pass import ABRunner
from infra.data_handler import DataHandler
from infra.evaluator import Evaluator
from infra.llm_handler import LLMHandler

from infra.config_loader import get_block


class ModelSweepRunner:
    def __init__(self, config: Dict[str, Any], data_handler: DataHandler,
                 llm_handler: LLMHandler, evaluator: Evaluator):
        self.config = config
        self.data_handler = data_handler
        self.evaluator = evaluator
        # llm_handler from constructor is the *default* one (built from top-level
        # config). The sweep replaces it per-model below.
        self._default_llm_handler = llm_handler

        cfg = get_block(config, "s10_model_sweep")
        self.models: List[Dict[str, Any]] = cfg.get("models", [])
        self.datasets: List[str] = cfg.get("datasets", ["MedQA"])
        self.ab_results_dir = Path(cfg.get("ab_results_dir", "results/ab"))
        self.results_dir = Path(cfg.get("results_dir", "results/exp2"))
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # =================================================================
    # Top-level
    # =================================================================
    async def run(self):
        if not self.models:
            print("[S10] No models configured under exp2_model_sweep.models — nothing to do.")
            return
        for entry in self.models:
            model_name = entry["name"] if isinstance(entry, dict) else str(entry)
            print(f"\n===== S10(c) :: model = {model_name} =====")
            await self._run_one_model(entry)

        # After all per-model runs complete, aggregate.
        for ds in self.datasets:
            self._aggregate(ds)

    # =================================================================
    # Per-model: spin up an ABRunner with overridden model/endpoint
    # =================================================================
    async def _run_one_model(self, entry):
        # Build a sub-config that ABRunner / LLMHandler will see. Only the
        # endpoint-related fields are overridden; everything else (judge,
        # results paths, run_followups, etc.) is inherited.
        if isinstance(entry, str):
            model_name, api_key, base_url = entry, None, None
        else:
            model_name = entry["name"]
            api_key = entry.get("api_key")
            base_url = entry.get("base_url")

        sub_config = dict(self.config)
        sub_config["model_name"] = model_name
        if api_key is not None:
            sub_config["api_key"] = api_key
        if base_url is not None:
            sub_config["base_url"] = base_url

        # Restrict ABRunner to the S10 dataset list (overrides any datasets
        # already set under ab_experiment so we don't accidentally double-run
        # ARC/FLD/etc. for every model).
        sub_ab = dict(get_block(sub_config, "main_experiment"))
        sub_ab["datasets"] = self.datasets
        sub_config["main_experiment"] = sub_ab

        llm_handler = LLMHandler(sub_config)
        runner = ABRunner(sub_config, self.data_handler, llm_handler, self.evaluator)
        await runner.run()

    # =================================================================
    # Aggregation: read per-model summaries and tabulate Abs Rate
    # =================================================================
    def _aggregate(self, dataset: str):
        rows = []
        for entry in self.models:
            model_name = entry["name"] if isinstance(entry, dict) else str(entry)
            safe_model = model_name.replace("/", "_")
            path = self.ab_results_dir / f"ab_summary_{dataset}_{safe_model}.json"
            if not path.exists():
                print(f"  [warn] missing summary for {model_name} on {dataset}: {path}")
                continue
            s = json.loads(path.read_text())
            m = s.get("metrics", {})
            n_total = s.get("n_total") or 0
            n_abstention_inflation = s.get("n_abstention_inflation") or 0
            n_ai_s3 = s.get("n_ai_s3")  # may be absent
            rows.append({
                "model":     model_name,
                "n_total":   n_total,
                "n_abstention_inflation":  n_abstention_inflation,
                "abs_rate_s2":    (n_abstention_inflation / n_total) if n_total else None,
                "abs_rate_s3":    (n_ai_s3 / n_total) if (n_total and n_ai_s3 is not None) else None,
                "Acc_S1":    (m.get("S1") or {}).get("label_acc"),
                "Acc_S2":    (m.get("S2") or {}).get("label_acc"),
                "Acc_S3":    (m.get("S3") or {}).get("label_acc"),
                "F1_S1":     (m.get("S1") or {}).get("label_f1"),
                "F1_S2":     (m.get("S2") or {}).get("label_f1"),
                "F1_S3":     (m.get("S3") or {}).get("label_f1"),
            })
        if not rows:
            print(f"[S10] no per-model summaries found for dataset={dataset} — skip aggregate.")
            return

        report = {"dataset": dataset, "rows": rows}
        out = self.results_dir / f"exp2_model_sweep_{dataset}.json"
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\n[S10] aggregate written -> {out}")
        self._print_table(dataset, rows)

    @staticmethod
    def _print_table(dataset: str, rows: list):
        print(f"\n=== S10(c) :: model size × Abs Rate ({dataset}) ===")
        cols = ["model", "abs_rate_s2", "abs_rate_s3", "Acc_S1", "Acc_S2", "Acc_S3"]
        widths = [max(len(c), max((len(str(r.get(c, ""))) for r in rows), default=4)) for c in cols]
        header = "  " + "  ".join(c.ljust(widths[i]) for i, c in enumerate(cols))
        print(header)
        print("  " + "  ".join("-" * w for w in widths))
        for r in rows:
            cells = []
            for i, c in enumerate(cols):
                v = r.get(c)
                if isinstance(v, float):
                    cells.append(f"{v:.2%}".ljust(widths[i]))
                else:
                    cells.append(str(v).ljust(widths[i]))
            print("  " + "  ".join(cells))

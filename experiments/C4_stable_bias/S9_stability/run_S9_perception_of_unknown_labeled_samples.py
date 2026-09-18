"""S9 Stability — the Unknown-labeled mirror.

The other side of Abs Rate: on items that genuinely have no determinable
answer, abstaining is correct, so this run measures whether the models can
tell the two populations apart. Paired against the Abs Rate on answerable
items, it is what shows the bias is directional rather than indiscriminate.

Driven by main.py through configs/C4_stable_bias/S9_unknown_labeled_*.yaml.
"""
import sys
from pathlib import Path

# Importable as a module and runnable as a file.
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import asyncio
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from infra.label_scheme import get_scheme
from loader.dataset_loader import load_dataset as load_judge
from infra.prompts import (
    build_judge_s1_prompt,
    build_judge_s2_prompt,
    build_judge_calibration_suffix_prompt,
)
from loader.data_handler import DataHandler
from infra.evaluator import Evaluator
from infra.llm_handler import LLMHandler

from infra import metrics

from loader.config_loader import get_block
from infra.result_schema import results_dir, stamp


SUPPLEMENTARY_DATASETS = ("FLD", "FOLIO", "FLD_unknown", "FOLIO_unknown")

# Heuristic check: any of these characters mean a symbolic-logic field leaked into
# the prompt input. NL fields in FLD never contain these.
_SYMBOLIC_RE = re.compile(r"[∀∃∧∨¬→↔⊕⇒⇔]")


class UnknownLabeledRunner:
    """Runs S1/S2/S3 on the genuinely-Unknown subset of FLD."""

    def __init__(self, config: Dict[str, Any], data_handler: DataHandler,
                 llm_handler: LLMHandler, evaluator: Evaluator):
        self.config = config
        self.data_handler = data_handler
        self.llm_handler = llm_handler
        self.evaluator = evaluator

        sup = get_block(config, "s9_unknown_labeled")
        self.dataset_names = sup.get("datasets", list(SUPPLEMENTARY_DATASETS))
        self.sample_limits = sup.get("sample_limits", {}) or {}
        self.results_root = Path(sup.get("results_root", "results"))
        self.model_slug = sup.get("model_slug") or (Path(sup["results_dir"]).name if sup.get("results_dir") else None)
        if not self.model_slug:
            raise ValueError("config needs model_slug (e.g. nano / dsv4flash / gemini31) to place its cells")
        self.results_dir = Path(sup["results_dir"]) if sup.get("results_dir") \
            else results_dir("S9/unknown_labeled", self.results_root) / self.model_slug
        self.results_dir.mkdir(parents=True, exist_ok=True)


    async def run(self):
        for ds_name in self.dataset_names:
            if ds_name not in SUPPLEMENTARY_DATASETS:
                print(f"[UnknownLabeledRunner] Skipping {ds_name}: supported = {SUPPLEMENTARY_DATASETS}.")
                continue
            print(f"\n===== Supplementary :: {ds_name} =====")
            # Unknown-labeled items now live in their own files (FLD_unknown.json /
            # FOLIO_unknown.json) under the unified schema. We accept either
            # form: when a user supplies "FLD" we auto-route to FLD_unknown.
            target = ds_name if ds_name.endswith("_unknown") else f"{ds_name}_unknown"
            samples = load_judge(target)
            unknown_samples = [s for s in samples if s.answer_idx == -1]
            unknown_samples = self._apply_sample_limit(ds_name, unknown_samples)
            print(f"  loaded {len(samples)} total, {len(unknown_samples)} genuinely Unknown.")
            if not unknown_samples:
                continue
            await self._run_one(ds_name, unknown_samples)

    def _apply_sample_limit(self, ds_name: str, samples: list) -> list:
        spec = self.sample_limits.get(ds_name)
        if spec is None:
            return samples
        if isinstance(spec, int):
            return samples[:spec]
        raise ValueError(
            f"sample_limits[{ds_name}] must be int, got {type(spec)}"
        )

    async def _run_one(self, ds_name: str, samples: list):
        # Sanity check: no symbolic logic content should appear in question/context.
        # If it does, the loader is reading the wrong field and we abort loudly
        # rather than silently mixing symbolic content into the prompts.
        leaks = [
            s.id for s in samples
            if _SYMBOLIC_RE.search(s.question or "") or _SYMBOLIC_RE.search(s.context or "")
        ]
        if leaks:
            raise RuntimeError(
                f"[Supplementary :: {ds_name}] {len(leaks)} samples contain symbolic logic "
                f"characters in question/context — first few: {leaks[:5]}. "
                "The supplementary experiment is natural-language-only; check the loader."
            )

        scheme = get_scheme(ds_name)
        s1_prompts = [build_judge_s1_prompt(scheme, s.question, s.context) for s in samples]
        s2_prompts = [build_judge_s2_prompt(scheme, s.question, s.context) for s in samples]
        s3_prompts = [build_judge_calibration_suffix_prompt(scheme, s.question, s.context) for s in samples]

        print("  [Step] Querying S1 / S2 / S3 in parallel ...")
        raw_s1, raw_s2, raw_s3 = await asyncio.gather(
            self.llm_handler.batch_query(s1_prompts),
            self.llm_handler.batch_query(s2_prompts),
            self.llm_handler.batch_query(s3_prompts),
        )

        preds_s1, tiers_s1 = await self._parse_batch(raw_s1, samples, scheme,
                                                      with_unknown=False, label="S1")
        preds_s2, tiers_s2 = await self._parse_batch(raw_s2, samples, scheme,
                                                      with_unknown=True, label="S2")
        preds_s3, tiers_s3 = await self._parse_batch(raw_s3, samples, scheme,
                                                      with_unknown=True, label="S3")

        summary = self._build_summary(
            ds_name, samples,
            preds_s1, preds_s2, preds_s3,
            tiers_s1, tiers_s2, tiers_s3,
            raw_s1, raw_s2, raw_s3,
        )
        self._save(ds_name, summary)

    # =================================================================
    # Output parsing — deterministic tiers (parity with core.paired_pass.ABRunner)
    # =================================================================
    async def _parse_batch(self, raw_outputs, samples, scheme, *,
                            with_unknown, label: str = ""):
        results = [
            self.evaluator.parse_judge_tiered(r, scheme, with_unknown=with_unknown)
            for r in raw_outputs
        ]
        return [r[0] for r in results], [r[1] for r in results]

    # =================================================================
    # Summary + persistence
    # =================================================================
    def _build_summary(self, ds_name, samples,
                       preds_s1, preds_s2, preds_s3,
                       tiers_s1, tiers_s2, tiers_s3,
                       raw_s1, raw_s2, raw_s3) -> Dict[str, Any]:
        car_s2 = metrics.car(preds_s2)
        car_s3 = metrics.car(preds_s3)
        forced_commit_s1 = metrics.forced_commitment_rate(preds_s1)

        def _tier_breakdown(tiers):
            counts = {"strict_em": 0, "lenient_em": 0, "unparseable": 0}
            for t in tiers:
                if t in counts:
                    counts[t] += 1
                else:
                    counts["unparseable"] += 1
            return counts

        return {
            "dataset": ds_name,
            "model": self.config.get("model_name"),
            "n_total": len(samples),
            "n_unparseable": {
                "s1": sum(1 for p in preds_s1 if p == "UNPARSEABLE"),
                "s2": sum(1 for p in preds_s2 if p == "UNPARSEABLE"),
                "s3": sum(1 for p in preds_s3 if p == "UNPARSEABLE"),
            },
            "tier_counts": {
                "s1": _tier_breakdown(tiers_s1),
                "s2": _tier_breakdown(tiers_s2),
                "s3": _tier_breakdown(tiers_s3),
            },
            "metrics": {
                "correct_abstention_s2":                  car_s2,
                "correct_abstention_s3":                  car_s3,
                "correct_abstention_improvement_s2_to_s3": car_s3 - car_s2,
                "forced_commitment_rate_s1": forced_commit_s1,
                "pos_rate_s1":              metrics.commit_rate(preds_s1, "A"),
                "neg_rate_s1":              metrics.commit_rate(preds_s1, "B"),
            },
            "per_sample": [
                {
                    "id":      samples[i].id,
                    "source":  samples[i].source,
                    "pred_s1": preds_s1[i],
                    "pred_s2": preds_s2[i],
                    "pred_s3": preds_s3[i],
                    "raw_s1":  raw_s1[i],
                    "raw_s2":  raw_s2[i],
                    "raw_s3":  raw_s3[i],
                }
                for i in range(len(samples))
            ],
        }

    def _save(self, ds_name: str, summary: Dict[str, Any]):
        model = self.config.get("model_name", "unknown").replace("/", "_")
        path = self.results_dir / f"{ds_name}_{model}.json"
        summary = {**stamp("S9/unknown_labeled"), **summary}
        self.data_handler.save_json(summary, path)
        m = summary["metrics"]
        print(
            f"  [Result] CAR_S2={m['CAR_S2']:.2%}  CAR_S3={m['CAR_S3']:.2%}  "
            f"Δ={m['CAR_Improvement_S2_to_S3']:+.2%}  "
            f"S1 ForcedCommit={m['forced_commitment_rate_s1']:.2%}"
        )


def main() -> None:
    """Run this setting from a config."""
    import argparse
    import asyncio

    from loader.config_loader import load_config
    from loader.data_handler import DataHandler
    from infra.evaluator import Evaluator
    from infra.llm_handler import LLMHandler

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, help="Experiment YAML.")
    args = ap.parse_args()

    config = load_config(args.config)
    asyncio.run(UnknownLabeledRunner(config, DataHandler(config), LLMHandler(config),
                      Evaluator()).run())


if __name__ == "__main__":
    main()

"""Supplementary experiment runner — paper §4.

Goal
----
Mirror image of the main Abstention Inflation experiment:

    main (Exp 1)            supplementary (this file)
    ------------            -------------------------
    answerable samples      genuinely Unknown samples
    Abs Rate ↑ is bad            CAR ↑ is good
    S1 = ground truth       S1 = forced wrong commitment
    S2/S3 = inflated abst.  S2/S3 = correct abstention

Datasets: FLD (300 genuinely-Unknown samples). FEVER is NOT included — the
supplementary experiment is intentionally scoped to the logic-reasoning
dataset that ships with native genuinely-Unknown labels.

Natural-language only
---------------------
FLD ships with symbolic logic fields on disk (`original_data.hypothesis_formula`,
`original_data.facts_formula`, etc.). The supplementary experiment uses only
the natural-language fields `Conclusion` (question) and `Facts` (context),
which is what `judge_loader._load_fld` already extracts. Symbolic fields
are never read here and never reach the model. A regex sanity check at the
top of `_run_one` enforces this contract at runtime.

Settings (reused verbatim from core.prompts)
--------------------------------------------------------
    S1 (no Unknown)   → forced binary; measures forced-commitment cost
    S2 (with Unknown) → CAR (Correct Abstention Rate)
    S3 (S2 + suffix)  → CAR with structural note

Output: results/supplementary/supp_summary_<dataset>_<model>.json

Parsing parity with ABRunner
----------------------------
Mirrors `core.ab_runner.ABRunner`:
    * `Evaluator.parse_judge_tiered` (not the legacy single-return parser)
    * Optional LLM-as-Judge fallback via `maybe_build_judge` (reuses the same
      `judge` config block under `supplementary_experiment.judge`; falls back
      to the global `ab_experiment.judge` block if unset).
    * Summary records `tier_counts`, `judge_model`, `judge_stats` so a reader
      can audit how many predictions came from strict / lenient EM vs the
      LLM judge fallback.
    * Optional `sample_limits` map (per-dataset cap, int).
"""
import asyncio
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.label_scheme import get_scheme
from core.dataset_loader import load_dataset as load_judge
from core.judge_fallback import LLMJudge, maybe_build_judge
from core.prompts import (
    build_judge_s1_prompt,
    build_judge_s2_prompt,
    build_judge_calibration_suffix_prompt,
)
from core.data_handler import DataHandler
from core.evaluator import Evaluator
from core.llm_handler import LLMHandler

from . import metrics

from .config_loader import get_block


SUPPLEMENTARY_DATASETS = ("FLD", "FOLIO", "FLD_unknown", "FOLIO_unknown")

# Heuristic check: any of these characters mean a symbolic-logic field leaked into
# the prompt input. NL fields in FLD never contain these.
_SYMBOLIC_RE = re.compile(r"[∀∃∧∨¬→↔⊕⇒⇔]")


class TrulyUnknownRunner:
    """Runs S1/S2/S3 on the genuinely-Unknown subset of FLD."""

    def __init__(self, config: Dict[str, Any], data_handler: DataHandler,
                 llm_handler: LLMHandler, evaluator: Evaluator):
        self.config = config
        self.data_handler = data_handler
        self.llm_handler = llm_handler
        self.evaluator = evaluator

        sup = get_block(config, "s9_truly_unknown")
        self.dataset_names = sup.get("datasets", list(SUPPLEMENTARY_DATASETS))
        self.sample_limits = sup.get("sample_limits", {}) or {}
        self.results_dir = Path(sup.get("results_dir", "results/supplementary"))
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # LLM-as-Judge fallback. Prefer a supplementary-specific judge block;
        # fall back to the main ab_experiment.judge block so the user can
        # configure the judge once and have it apply to both runners.
        self.judge: Optional[LLMJudge] = self._build_judge(config)
        if self.judge:
            print(f"[TrulyUnknownRunner] LLM-as-Judge fallback enabled "
                  f"(model={self.judge.model_name}).")
        self._judge_calls = 0
        self._judge_recovered = 0

    @staticmethod
    def _build_judge(config: Dict[str, Any]) -> Optional[LLMJudge]:
        sup_cfg = get_block(config, "s9_truly_unknown").get("judge")
        if sup_cfg is not None:
            scoped = {"main_experiment": {"judge": sup_cfg}}
            return maybe_build_judge(scoped)
        return maybe_build_judge(config)

    async def run(self):
        for ds_name in self.dataset_names:
            if ds_name not in SUPPLEMENTARY_DATASETS:
                print(f"[TrulyUnknownRunner] Skipping {ds_name}: supported = {SUPPLEMENTARY_DATASETS}.")
                continue
            print(f"\n===== Supplementary :: {ds_name} =====")
            # Truly-Unknown items now live in their own files (FLD_unknown.json /
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
    # Output parsing — tiered + optional LLM-as-Judge fallback
    # (parity with core.ab_runner.ABRunner)
    # =================================================================
    async def _parse_batch(self, raw_outputs, samples, scheme, *,
                            with_unknown, label: str = ""):
        results = [
            self.evaluator.parse_judge_tiered(r, scheme, with_unknown=with_unknown)
            for r in raw_outputs
        ]
        preds = [r[0] for r in results]
        tiers = [r[1] for r in results]

        if self.judge is None:
            return preds, tiers
        return await self._judge_fallback(preds, tiers, raw_outputs, scheme,
                                           with_unknown=with_unknown, label=label)

    async def _judge_fallback(self, preds, tiers, raw_outputs, scheme, *,
                               with_unknown, label: str):
        unparseable = [i for i, p in enumerate(preds) if p == "UNPARSEABLE"]
        if not unparseable:
            return preds, tiers

        calls = [self.judge.judge_tf(raw_outputs[i], scheme, with_unknown)
                 for i in unparseable]
        print(f"  [Judge:{label}] {len(unparseable)} unparseable → calling judge ...")
        judge_raw = await asyncio.gather(*calls)
        self._judge_calls += len(judge_raw)

        recovered = 0
        for k, i in enumerate(unparseable):
            new_pred, _ = self.evaluator.parse_judge_tiered(
                judge_raw[k], scheme, with_unknown=with_unknown
            )
            if new_pred != "UNPARSEABLE":
                preds[i] = new_pred
                tiers[i] = "judge"
                recovered += 1
        self._judge_recovered += recovered
        print(f"  [Judge:{label}] recovered {recovered}/{len(unparseable)}.")
        return preds, tiers

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
            counts = {"strict_em": 0, "lenient_em": 0, "judge": 0, "unparseable": 0}
            for t in tiers:
                if t in counts:
                    counts[t] += 1
                else:
                    counts["unparseable"] += 1
            return counts

        return {
            "dataset": ds_name,
            "model": self.config.get("model_name"),
            "judge_model": self.judge.model_name if self.judge else None,
            "judge_stats": {
                "calls":     self._judge_calls,
                "recovered": self._judge_recovered,
            },
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
                "CAR_S2":                  car_s2,
                "CAR_S3":                  car_s3,
                "CAR_Improvement_S2_to_S3": car_s3 - car_s2,
                "ForcedCommitmentRate_S1": forced_commit_s1,
                "POS_rate_S1":             metrics.commit_rate(preds_s1, "A"),
                "NEG_rate_S1":             metrics.commit_rate(preds_s1, "B"),
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
        path = self.results_dir / f"supp_summary_{ds_name}_{model}.json"
        self.data_handler.save_json(summary, path)
        m = summary["metrics"]
        print(
            f"  [Result] CAR_S2={m['CAR_S2']:.2%}  CAR_S3={m['CAR_S3']:.2%}  "
            f"Δ={m['CAR_Improvement_S2_to_S3']:+.2%}  "
            f"S1 ForcedCommit={m['ForcedCommitmentRate_S1']:.2%}"
        )

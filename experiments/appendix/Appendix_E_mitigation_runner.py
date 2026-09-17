"""App. E mitigation — Post-hoc mitigation via two-stage stimulation+reflection stimulation+reflection.

Pipeline
--------
For each configured dataset:
    1.  Read the existing S1/S2 summary from results/ab/ab_summary_<ds>_<model>.json.
    2.  Identify Abstention Inflation samples (those S2 returned UNKNOWN on) from `s5_rerun`.
    3.  Reconstruct each Abstention Inflation sample's S2 conversation:
            user_msg  = build_<kind>_s2_prompt(...)        (deterministic)
            asst_reply = s1_s2_summary.per_sample[id].raw_s2  (saved earlier)
    4.  Stage 1 (Stimulation): multi-turn extension of S2 with the shared S4
        stimulation core. Unknown is still on offer; the model is stimulated
        against picking it (post-hoc mitigation ending").
    5.  Stage 2 (Reflection): on the subset still UNKNOWN after Stage 1,
        multi-turn extension of Stage 1 asking the model to reflect on its
        own reasoning and finalize.
    6.  Compute PMR (Pipeline Mitigation Rate) and PMR_correct.
    7.  Save results/exp4/exp4_summary_<ds>_<model>.json.

Key invariants
--------------
- We do NOT re-run S1/S2/S3. Abs Rate identification reuses the saved summary.
- Stage 1 and Stage 2 prompt cores are aligned with S4 (its contract:
  "S4 shares its stimulation prompt core with the Stage 1 prior-work template").
- PMR has the same numerator structure as Recovery Rate (forced choice from
  S4) but allows the model to commit to a wrong answer; PMR_correct is the
  apples-to-apples comparison against Recovery Rate.

Per-dataset metrics
-------------------
    abs_rate_s2_subset                = 1.0 (by construction — input is the Abstention Inflation set)
    n_ai                        = |Abstention Inflation samples|
    n_no_longer_unknown          = |stage2 final ≠ UNKNOWN|
    n_correct_after_pipeline     = |stage2 final letter == correct answer|

    PMR          = n_no_longer_unknown / n_ai
    PMR_correct  = n_correct_after_pipeline / n_ai     (≤ PMR by construction)
    RecoveryRate = (from S1/S2 summary)                  (theoretical upper bound)
"""
import asyncio
import json
from pathlib import Path
import sys

# Importable as a module and runnable as a file.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from typing import Any, Dict, List

from infra import metrics
from infra.label_scheme import get_scheme
from infra.prompts import (
    build_mcq_s2_prompt,
    build_judge_s2_prompt,
    build_mcq_stage1_mitigation_prompt,
    build_mcq_stage2_reflection_prompt,
    build_judge_stage1_mitigation_prompt,
    build_judge_stage2_reflection_prompt,
)
from loader.data_handler import DataHandler
from infra.evaluator import Evaluator
from infra.llm_handler import LLMHandler

from loader.config_loader import get_block
from infra.result_schema import get_field


class PostHocMitigationRunner:
    """Reads S1/S2 summaries, applies 2-stage stimulation+reflection on Abstention Inflation samples."""

    def __init__(self, config: Dict[str, Any], data_handler: DataHandler,
                 llm_handler: LLMHandler, evaluator: Evaluator):
        self.config = config
        self.data_handler = data_handler
        self.llm_handler = llm_handler
        self.evaluator = evaluator

        cfg = get_block(config, "appendix_mitigation")
        self.dataset_names = cfg.get("datasets", [])
        self.s1_s2_results_dir = Path(cfg.get("s1_s2_results_dir", "results/ab"))
        self.results_dir = Path(cfg.get("results_dir", "results/exp4"))
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # =================================================================
    # Top-level
    # =================================================================
    async def run(self):
        if not self.dataset_names:
            print("[Exp4] No datasets configured under exp4_mitigation.datasets — nothing to do.")
            return
        for ds_name in self.dataset_names:
            print(f"\n===== App. E mitigation :: {ds_name} =====")
            await self._run_one(ds_name)

    async def _run_one(self, ds_name: str):
        # 1. Locate S1/S2 summary
        model = self.config.get("model_name", "unknown").replace("/", "_")
        s1_s2_path = self.s1_s2_results_dir / f"ab_summary_{ds_name}_{model}.json"
        if not s1_s2_path.exists():
            print(f"  [skip] {ds_name}: S1/S2 summary not found at {s1_s2_path}")
            return
        s1_s2 = json.loads(s1_s2_path.read_text())
        s5_rerun = get_field(s1_s2, "s5_rerun", [])
        if not s5_rerun:
            print(f"  [skip] {ds_name}: no Abstention Inflation samples in S1/S2 summary.")
            return

        # 2. Load samples and pick the Abstention Inflation subset
        all_samples = self.data_handler.load_dataset(ds_name)
        all_samples = [s for s in all_samples if s.answer_idx >= 0]
        sample_by_id = {s.id: s for s in all_samples}
        ai_ids = [a["sample_id"] for a in s5_rerun]
        ai_samples = [sample_by_id[i] for i in ai_ids if i in sample_by_id]
        if not ai_samples:
            print(f"  [skip] {ds_name}: Abstention Inflation sample IDs not found in loaded dataset.")
            return
        if len(ai_samples) != len(ai_ids):
            missing = set(ai_ids) - set(sample_by_id)
            print(f"  [warn] {ds_name}: {len(missing)} Abstention Inflation ids missing from dataset; proceeding with {len(ai_samples)}.")
        task_type = ai_samples[0].task_type

        # 3. Reconstruct S2 conversations (S2 user msg + S2 asst reply)
        per_sample_by_id = {p["id"]: p for p in s1_s2["per_sample"]}
        s2_prompts = self._build_s2_prompts(ai_samples, task_type)
        raw_s2 = [per_sample_by_id[s.id]["raw_s2"] for s in ai_samples]

        # 4. Stage 1 stimulation
        stage1_prompts = self._build_stage1(ai_samples, s2_prompts, raw_s2, task_type)
        print(f"  [Stage 1] Querying mitigation on {len(ai_samples)} Abstention Inflation samples ...")
        raw_stage1 = await self.llm_handler.batch_query(stage1_prompts)
        preds_stage1 = self._parse(raw_stage1, ai_samples, task_type)

        # 5. Stage 2 reflection (only on still-UNKNOWN)
        still_unknown_idx = [i for i, p in enumerate(preds_stage1) if p == "UNKNOWN"]
        raw_stage2 = ["" for _ in ai_samples]
        preds_final = list(preds_stage1)
        if still_unknown_idx:
            sub_samples = [ai_samples[i] for i in still_unknown_idx]
            sub_stage1_prompts = [stage1_prompts[i] for i in still_unknown_idx]
            sub_raw_stage1 = [raw_stage1[i] for i in still_unknown_idx]
            stage2_prompts = self._build_stage2(sub_samples, sub_stage1_prompts, sub_raw_stage1, task_type)
            print(f"  [Stage 2] Reflection on {len(still_unknown_idx)} still-UNKNOWN samples ...")
            raw_stage2_sub = await self.llm_handler.batch_query(stage2_prompts)
            preds_stage2_sub = self._parse(raw_stage2_sub, sub_samples, task_type)
            for k, i in enumerate(still_unknown_idx):
                raw_stage2[i] = raw_stage2_sub[k]
                preds_final[i] = preds_stage2_sub[k]

        # 6. Metrics + 7. Save
        summary = self._build_summary(
            ds_name, task_type, ai_samples,
            raw_s2, raw_stage1, raw_stage2,
            preds_stage1, preds_final,
            s1_s2.get("metrics", {}),
        )
        self._save(ds_name, summary)

    # =================================================================
    # Prompt assembly + parsing (task_type-aware)
    # =================================================================
    def _build_s2_prompts(self, samples, task_type):
        if task_type == "mcq":
            return [build_mcq_s2_prompt(s.question, s.options) for s in samples]
        if task_type == "tf":
            return [build_judge_s2_prompt(get_scheme(s.source), s.question, s.context)
                    for s in samples]
        raise ValueError(f"Unsupported task_type: {task_type}")

    def _build_stage1(self, samples, s2_prompts, raw_s2, task_type):
        if task_type == "mcq":
            return [
                build_mcq_stage1_mitigation_prompt(s2_prompts[i], raw_s2[i],
                                                    n_options=len(samples[i].options))
                for i in range(len(samples))
            ]
        if task_type == "tf":
            return [
                build_judge_stage1_mitigation_prompt(s2_prompts[i], raw_s2[i],
                                                      get_scheme(samples[i].source))
                for i in range(len(samples))
            ]
        raise ValueError(f"Unsupported task_type: {task_type}")

    def _build_stage2(self, samples, stage1_prompts, raw_stage1, task_type):
        if task_type == "mcq":
            return [
                build_mcq_stage2_reflection_prompt(stage1_prompts[i], raw_stage1[i],
                                                    n_options=len(samples[i].options))
                for i in range(len(samples))
            ]
        if task_type == "tf":
            return [
                build_judge_stage2_reflection_prompt(stage1_prompts[i], raw_stage1[i],
                                                      get_scheme(samples[i].source))
                for i in range(len(samples))
            ]
        raise ValueError(f"Unsupported task_type: {task_type}")

    def _parse(self, raws, samples, task_type):
        if task_type == "mcq":
            return [self.evaluator.parse_mcq_output(r, with_unknown=True) for r in raws]
        if task_type == "tf":
            return [
                self.evaluator.parse_judge_output(r, get_scheme(samples[i].source),
                                                   with_unknown=True)
                for i, r in enumerate(raws)
            ]
        raise ValueError(f"Unsupported task_type: {task_type}")

    # =================================================================
    # Summary + persistence
    # =================================================================
    def _build_summary(self, ds_name, task_type, ai_samples,
                        raw_s2, raw_stage1, raw_stage2,
                        preds_stage1, preds_final,
                        s1_s2_metrics) -> Dict[str, Any]:
        n_ai = len(ai_samples)
        answer_idxs = [s.answer_idx for s in ai_samples]
        n_still_unknown = sum(1 for p in preds_final if p == "UNKNOWN")
        n_no_longer_unknown = n_ai - n_still_unknown
        n_correct = sum(metrics.is_correct(p, a)
                        for p, a in zip(preds_final, answer_idxs))

        pmr = n_no_longer_unknown / n_ai if n_ai else 0.0
        pmr_correct = n_correct / n_ai if n_ai else 0.0

        return {
            "dataset":   ds_name,
            "task_type": task_type,
            "model":     self.config.get("model_name"),
            "n_ai":     n_ai,
            "counts": {
                "stage1_resolved":           sum(1 for p in preds_stage1 if p != "UNKNOWN"),
                "stage1_still_unknown":      sum(1 for p in preds_stage1 if p == "UNKNOWN"),
                "final_no_longer_unknown":   n_no_longer_unknown,
                "final_still_unknown":       n_still_unknown,
                "final_correct":             n_correct,
                "final_unparseable":         sum(1 for p in preds_final if p == "UNPARSEABLE"),
            },
            "metrics": {
                "PMR":                       pmr,
                "PMR_correct":               pmr_correct,
                # New unified summary stores per-setting blocks; pull S4 label_acc
                # (the unified equivalent of the old "RecoveryRate") if present.
                "RecoveryRate_from_Exp1":    (
                    s1_s2_metrics.get("S4", {}).get("label_acc")
                    if isinstance(s1_s2_metrics.get("S4"), dict)
                    else s1_s2_metrics.get("RecoveryRate")
                ),
                # Abs Rate equivalent: derived from S2 abstention rate. New summary
                # exposes this via n_abstention_inflation / n_total at top level.
                "abs_rate_s2_from_s1_s2":          s1_s2_metrics.get("abs_rate_s2"),
            },
            "per_sample": [
                {
                    "id":           ai_samples[i].id,
                    "source":       ai_samples[i].source,
                    "answer_idx":   answer_idxs[i],
                    "raw_s2":       raw_s2[i],
                    "raw_stage1":   raw_stage1[i],
                    "raw_stage2":   raw_stage2[i],
                    "pred_stage1":  preds_stage1[i],
                    "pred_final":   preds_final[i],
                    "correct":      metrics.is_correct(preds_final[i], answer_idxs[i]),
                }
                for i in range(n_ai)
            ],
        }

    def _save(self, ds_name: str, summary: Dict[str, Any]):
        model = self.config.get("model_name", "unknown").replace("/", "_")
        path = self.results_dir / f"exp4_summary_{ds_name}_{model}.json"
        self.data_handler.save_json(summary, path)
        m = summary["metrics"]
        rec = m.get("RecoveryRate_from_Exp1")
        rec_str = f"{rec:.2%}" if rec is not None else "n/a"
        print(
            f"  [Result] PMR={m['PMR']:.2%}  PMR_correct={m['PMR_correct']:.2%}  "
            f"RecoveryRate(Exp1)={rec_str}"
        )

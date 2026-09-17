"""S5 self-diagnosis (metacognition) supplementary experiment.

S5 was deliberately moved OUT of the main S1–S4 framework because it is not a
label-prediction task on the original question — it is a meta-question
("is your abstention objectively necessary?") with A/B options, whose gold
label is derived from S4 behavior. So it doesn't fit the unified
(Label-Acc, Label-F1, Trace-Acc, Trace-F1) reporting that S1–S4 share.

What this runner does:
    1. For each (dataset, model) tuple, read the ABRunner summary JSON
       at `results/ab/ab_summary_<dataset>_<model>.json`.
    2. Identify Abstention Inflation samples (S2 == UNKNOWN) and the corresponding raw_s2 +
       prior S2 prompt history (re-built from sample data).
    3. Build S5 self-diagnosis prompts (verb-coded for Judge, letter-coded
       for MCQ) and query the model.
    4. Parse A/B responses; cross with S4 correctness flags from the same
       summary; compute SelfDiagnosisAcc + 4-bucket cross-tab.
    5. Write `results/supplementary/s5_<dataset>_<model>.json`.

Key design: this runner does NOT re-query S1/S2/S3 — it consumes existing
ABRunner output. So running S5 is cheap (~|Abs Rate| extra calls per dataset).

Parsing parity with ABRunner
----------------------------
Mirrors `core.ab_runner.ABRunner`:
    * `Evaluator.parse_ab_tiered` (returns `(pred, tier)`).
    * Optional LLM-as-Judge fallback for unparseable A/B replies. The judge
      is the same `LLMJudge` instance; we just dispatch through `judge_mcq`
      with the A/B option texts plumbed in as the option list (the judge
      replies with a single letter, which `parse_ab_tiered` handles).
    * Summary records `tier_counts`, `judge_model`, `judge_stats`.
"""
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.label_scheme import get_scheme
from core.judge_fallback import LLMJudge, maybe_build_judge
from core.prompts import (
    _format_mcq_options,
    _format_judge_options,
    _COT_INSTR_MCQ,
    _cot_instr_judge,
    build_mcq_s2_prompt,
    build_judge_s2_prompt,
)
from core import metrics as main_metrics

from core.data_handler import DataHandler
from core.evaluator import Evaluator
from core.llm_handler import LLMHandler

from .config_loader import get_block
from .result_schema import get_field


# =================================================================
# S5 prompt builders (kept here, not in main prompts.py)
# =================================================================

# A/B option texts surfaced to the model. Reused as the option list when the
# LLM-as-Judge fallback re-asks `judge_mcq` to map an unparseable raw reply
# onto a letter — keeping these in one place makes that mapping unambiguous.
S5_OPTION_A = (
    "The question is objectively unanswerable — the given information "
    "is genuinely insufficient to determine a correct answer."
)
S5_OPTION_B = (
    "You are uncertain, but the question likely has a correct answer "
    "that you were unable to determine."
)
S5_AB_OPTIONS: List[str] = [S5_OPTION_A, S5_OPTION_B]


def build_mcq_s5_prompt(prior_messages: List[Dict[str, str]],
                         prior_response: str) -> List[Dict[str, str]]:
    """S5 self-diagnosis (MCQ). Branches off S2 history; A/B answer."""
    followup = (
        "You previously selected \"Unknown\" for this question.\n"
        "Please identify the reason:\n"
        f"A. {S5_OPTION_A}\n"
        f"B. {S5_OPTION_B}"
        + _cot_instr_judge("A | B")
    )
    return list(prior_messages) + [
        {"role": "assistant", "content": prior_response},
        {"role": "user", "content": followup},
    ]


def build_judge_s5_prompt(prior_messages: List[Dict[str, str]],
                           prior_response: str,
                           scheme) -> List[Dict[str, str]]:
    """S5 self-diagnosis (Judge variant). Same A/B structure, references the
    scheme's actual abstain verb so future schemes (where abstain != "Unknown")
    stay correct."""
    followup = (
        f"You previously selected \"{scheme.abstain_verb}\" for this question.\n"
        "Please identify the reason:\n"
        f"A. {S5_OPTION_A}\n"
        f"B. {S5_OPTION_B}"
        + _cot_instr_judge("A | B")
    )
    return list(prior_messages) + [
        {"role": "assistant", "content": prior_response},
        {"role": "user", "content": followup},
    ]


# =================================================================
# S5 metrics (A/B specific — not part of the unified main framework)
# =================================================================

def self_diagnosis_acc(s5_preds: List[str], s4_correct_flags: List[bool]) -> float:
    """Agreement between S5 self-report and S4 behavior.

    S5='A' (claims objective unanswerability) ↔ S4 wrong (genuine inability)
    S5='B' (admits subjective uncertainty)    ↔ S4 correct (over-caution, recoverable)
    """
    n = len(s5_preds)
    if n == 0:
        return 0.0
    agree = 0
    for s5, s4c in zip(s5_preds, s4_correct_flags):
        if s5 == "A" and not s4c:
            agree += 1
        elif s5 == "B" and s4c:
            agree += 1
    return agree / n


def s4_s5_cross_buckets(s5_preds: List[str], s4_correct_flags: List[bool]) -> Dict[str, int]:
    buckets = {
        "overcaution_misdiagnosed": 0,  # S5=A & S4 correct
        "inability_misdiagnosed":   0,  # S5=A & S4 wrong
        "overcaution_selfaware":    0,  # S5=B & S4 correct
        "genuine_unknown":          0,  # S5=B & S4 wrong
        "unparseable":              0,
    }
    for s5, s4c in zip(s5_preds, s4_correct_flags):
        if s5 not in ("A", "B"):
            buckets["unparseable"] += 1
        elif s5 == "A" and s4c:
            buckets["overcaution_misdiagnosed"] += 1
        elif s5 == "A" and not s4c:
            buckets["inability_misdiagnosed"] += 1
        elif s5 == "B" and s4c:
            buckets["overcaution_selfaware"] += 1
        else:
            buckets["genuine_unknown"] += 1
    return buckets


# =================================================================
# Runner
# =================================================================

class S6SelfDiagnosisLegacyRunner:
    def __init__(self, config: Dict[str, Any], data_handler: DataHandler,
                 llm_handler: LLMHandler, evaluator: Evaluator):
        self.config = config
        self.data_handler = data_handler
        self.llm_handler = llm_handler
        self.evaluator = evaluator

        cfg = get_block(config, "s6_self_diagnosis")
        self.dataset_names = cfg.get("datasets", [])
        self.s1_s2_results_dir = Path(cfg.get("s1_s2_results_dir", "results/ab"))
        self.results_dir = Path(cfg.get("results_dir", "results/supplementary"))
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # LLM-as-Judge fallback. Prefer an s5_supplementary-specific judge
        # block; fall back to ab_experiment.judge so the user can configure
        # the judge in one place.
        self.judge: Optional[LLMJudge] = self._build_judge(config)
        if self.judge:
            print(f"[S6SelfDiagnosisLegacyRunner] LLM-as-Judge fallback enabled "
                  f"(model={self.judge.model_name}).")
        self._judge_calls = 0
        self._judge_recovered = 0

    @staticmethod
    def _build_judge(config: Dict[str, Any]) -> Optional[LLMJudge]:
        s5_cfg = get_block(config, "s6_self_diagnosis").get("judge")
        if s5_cfg is not None:
            scoped = {"main_experiment": {"judge": s5_cfg}}
            return maybe_build_judge(scoped)
        return maybe_build_judge(config)

    async def run(self):
        if not self.dataset_names:
            print("[S6SelfDiagnosisLegacyRunner] No datasets configured — nothing to do.")
            return
        for ds_name in self.dataset_names:
            print(f"\n===== S5 Supplementary :: {ds_name} =====")
            await self._run_one_dataset(ds_name)

    async def _run_one_dataset(self, ds_name: str):
        model = self.config.get("model_name", "unknown").replace("/", "_")
        in_path = self.s1_s2_results_dir / f"ab_summary_{ds_name}_{model}.json"
        if not in_path.exists():
            print(f"  [skip] missing {in_path} — run main ab_experiment first.")
            return
        summary = json.loads(in_path.read_text())
        task_type = summary["task_type"]

        # Load samples in same order as ab_runner produced.
        samples_all = self.data_handler.load_dataset(ds_name)
        samples_all = [s for s in samples_all if s.answer_idx >= 0]
        id_to_sample = {s.id: s for s in samples_all}

        # Abstention Inflation samples (S2 == UNKNOWN) + their raw_s2 + S4 correctness.
        ai_records = []
        for ps, fu in zip(summary["per_sample"], get_field(summary, "s5_rerun", [])):
            if ps["pred_s2"] != "UNKNOWN":
                continue
            sid = ps["id"]
            if sid not in id_to_sample:
                continue
            sample = id_to_sample[sid]
            # Reconstruct S2 prompt (deterministic builder).
            if task_type == "mcq":
                s2_msgs = build_mcq_s2_prompt(sample.question, sample.options)
            else:
                s2_msgs = build_judge_s2_prompt(get_scheme(sample.source),
                                                  sample.question, sample.context)
            ai_records.append({
                "sample": sample,
                "s2_messages": s2_msgs,
                "raw_s2": ps["raw_s2"],
                # S4 correctness comes from the followup record.
                "s4_correct": _is_correct_letter(fu.get("pred_s4"), fu["answer_idx"]),
            })

        if not ai_records:
            print(f"  [skip] {ds_name}: no Abstention Inflation samples in main summary.")
            return

        # Build S5 prompts.
        s5_prompts = []
        for rec in ai_records:
            sample = rec["sample"]
            if task_type == "mcq":
                s5_prompts.append(
                    build_mcq_s5_prompt(rec["s2_messages"], rec["raw_s2"])
                )
            else:
                s5_prompts.append(
                    build_judge_s5_prompt(rec["s2_messages"], rec["raw_s2"],
                                           get_scheme(sample.source))
                )

        print(f"  Querying S5 on {len(s5_prompts)} Abstention Inflation samples ...")
        raw_s5 = await self.llm_handler.batch_query(s5_prompts)

        # Parse A/B (tiered + optional LLM-as-Judge fallback).
        preds_s5, tiers_s5 = await self._parse_ab_batch(raw_s5, label="S5")

        s4_correct_flags = [r["s4_correct"] for r in ai_records]
        sd_acc = self_diagnosis_acc(preds_s5, s4_correct_flags)
        sd_f1 = main_metrics.label_macro_f1(
            preds_s5,
            # Synthetic gold: A if S4 wrong (objective), B if S4 correct (subjective).
            answer_idxs=[(0 if not c else 1) for c in s4_correct_flags],
            classes=["A", "B"],
        )
        buckets = s4_s5_cross_buckets(preds_s5, s4_correct_flags)

        out = {
            "dataset":           ds_name,
            "task_type":         task_type,
            "model":             self.config.get("model_name"),
            "judge_model":       self.judge.model_name if self.judge else None,
            "judge_stats": {
                "calls":     self._judge_calls,
                "recovered": self._judge_recovered,
            },
            "n_ai_evaluated":   len(ai_records),
            "tier_counts":       _tier_breakdown(tiers_s5),
            "metrics": {
                "self_diagnosis_acc": sd_acc,
                "self_diagnosis_f1":  sd_f1,
            },
            "s4_s5_buckets": buckets,
            "per_sample": [
                {
                    "sample_id":  rec["sample"].id,
                    "answer_idx": rec["sample"].answer_idx,
                    "s4_correct": rec["s4_correct"],
                    "pred_s5":    preds_s5[k],
                    "raw_s5":     raw_s5[k],
                }
                for k, rec in enumerate(ai_records)
            ],
        }
        out_path = self.results_dir / f"s5_{ds_name}_{self.config.get('model_name','unknown').replace('/','_')}.json"
        self.data_handler.save_json(out, out_path)
        print(f"  [Result] SelfDiagAcc={sd_acc:.2%}  SelfDiagF1={sd_f1:.2%}")
        print(f"  [Buckets] {buckets}")

    # =================================================================
    # A/B parsing — tiered + optional LLM-as-Judge fallback
    # (parity with core.ab_runner.ABRunner)
    # =================================================================
    async def _parse_ab_batch(self, raw_outputs, *, label: str = ""):
        results = [self.evaluator.parse_ab_tiered(r) for r in raw_outputs]
        preds = [r[0] for r in results]
        tiers = [r[1] for r in results]

        if self.judge is None:
            return preds, tiers

        unparseable = [i for i, p in enumerate(preds) if p == "UNPARSEABLE"]
        if not unparseable:
            return preds, tiers

        # Reuse judge_mcq with the A/B option texts; the judge replies with a
        # single letter that parse_ab_tiered already understands. Avoids
        # adding a new method to LLMJudge for this single use site.
        calls = [
            self.judge.judge_mcq(raw_outputs[i], S5_AB_OPTIONS, with_unknown=False)
            for i in unparseable
        ]
        print(f"  [Judge:{label}] {len(unparseable)} unparseable → calling judge ...")
        judge_raw = await asyncio.gather(*calls)
        self._judge_calls += len(judge_raw)

        recovered = 0
        for k, i in enumerate(unparseable):
            new_pred, _ = self.evaluator.parse_ab_tiered(judge_raw[k])
            if new_pred != "UNPARSEABLE":
                preds[i] = new_pred
                tiers[i] = "judge"
                recovered += 1
        self._judge_recovered += recovered
        print(f"  [Judge:{label}] recovered {recovered}/{len(unparseable)}.")
        return preds, tiers


def _tier_breakdown(tiers: List[str]) -> Dict[str, int]:
    counts = {"strict_em": 0, "lenient_em": 0, "judge": 0, "unparseable": 0}
    for t in tiers:
        if t in counts:
            counts[t] += 1
        else:
            counts["unparseable"] += 1
    return counts


def _is_correct_letter(letter, answer_idx) -> bool:
    """Boolean: did S4 (which never offers Unknown) recover the gold letter?"""
    if letter in (None, "UNKNOWN", "UNPARSEABLE"):
        return False
    return ord(letter) - ord("A") == answer_idx

"""S6 Self-Diagnosis — the model attributes its own abstention.

S6 sits outside the label-prediction settings: it is a meta-question ("why
did you abstain?") with two options, read against what the S5 rerun then did
on the same item. It has no label accuracy of its own.

What this runner does:
    1. For each (dataset, model) tuple, read the ABRunner summary JSON
       under results/S{1,2,5}_*/<family>/<slug>/<dataset>_<model>.json, joined by load_cell.
    2. Identify Abstention Inflation samples (S2 == UNKNOWN) and the corresponding raw_s2 +
       prior S2 prompt history (re-built from sample data).
    3. Build the S6 self-diagnosis prompts (verb-coded for Judge, letter-coded
       for MCQ) and query the model.
    4. Parse the A/B replies; cross them with whether the S5 rerun recovered
       the gold label; report the two attribution shares + the 4-bucket table.
    5. Write `results/S6_self_diagnosis/<model>/<dataset>_<model>.json`.

Key design: this runner does NOT re-query S1/S2/S3 — it consumes existing
ABRunner output. So running S6 is cheap (~|Abs Rate| extra calls per dataset).

Parsing parity with ABRunner
----------------------------
Mirrors `core.paired_pass.ABRunner`:
    * `Evaluator.parse_ab_tiered` (returns `(pred, tier)`).
    * Optional LLM-as-Judge fallback for unparseable A/B replies. The judge
      is the same deterministic parser
      with the A/B option texts plumbed in as the option list (the judge
      replies with a single letter, which `parse_ab_tiered` handles).
    * Summary records `tier_counts`.
"""

import sys
from pathlib import Path

# Importable as a module and runnable as a file.
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import asyncio
from pathlib import Path
from typing import Any, Dict, List

from infra.evaluator import Evaluator
from infra.label_scheme import get_scheme
from infra.llm_handler import LLMHandler
from infra.prompts import (
    S6_OPTION_A,
    S6_OPTION_B,
    build_judge_s2_prompt,
    build_judge_s6_selfdiag_prompt,
    build_mcq_s2_prompt,
    build_mcq_s6_selfdiag_prompt,
)
from infra.result_schema import get_field, load_cell, results_dir, stamp
from loader.config_loader import get_block
from loader.data_handler import DataHandler

# =================================================================
# S6 A/B option texts
# =================================================================

# A/B option texts surfaced to the model. Reused as the option list when the
# LLM-as-Judge fallback re-asks `judge_mcq` to map an unparseable raw reply
# onto a letter — keeping these in one place makes that mapping unambiguous.
#: The follow-up is built by :mod:`infra.prompts`, not here. This module used
#: to carry its own copy with A and B the other way round -- A "objectively
#: unanswerable", B "uncertain but answerable" -- while the paper and the
#: builder define A as the model's own inability and B as the item being
#: unanswerable. Two definitions of the same two letters is how a self-report
#: gets read as its own opposite, so there is now one.
S6_AB_OPTIONS: List[str] = [S6_OPTION_A, S6_OPTION_B]


# =================================================================
# S6 metrics (A/B specific — not part of the unified main framework)
# =================================================================


def attribution_share(preds: List[str], letter: str) -> float:
    """Share of the abstaining subset that attributed its abstention to
    ``letter`` -- A, the model's own inability; B, the item being
    unanswerable. Unparseable replies stay in the denominator. The paper's S6
    number is the B share."""
    return sum(p == letter for p in preds) / len(preds) if preds else 0.0


def self_diagnosis_x_rerun_buckets(
    preds: List[str], rerun_correct: List[bool]
) -> Dict[str, int]:
    # A = "I could not work it out" (own inability); B = "the question is
    # objectively unanswerable". Paired with whether the S5 rerun then got the
    # item right, that gives four cases.
    buckets = {
        "overcaution_misdiagnosed": 0,  # B & rerun correct: called it
        # unanswerable, then answered it
        "genuine_unknown": 0,  # B & rerun wrong
        "inability_selfaware": 0,  # A & rerun wrong: could not do it,
        # and said so
        "inability_misdiagnosed": 0,  # A & rerun correct: could do it,
        # but blamed itself
        "unparseable": 0,
    }
    for pred, ok in zip(preds, rerun_correct):
        if pred not in ("A", "B"):
            buckets["unparseable"] += 1
        elif pred == "B" and ok:
            buckets["overcaution_misdiagnosed"] += 1
        elif pred == "B" and not ok:
            buckets["genuine_unknown"] += 1
        elif pred == "A" and not ok:
            buckets["inability_selfaware"] += 1
        else:
            buckets["inability_misdiagnosed"] += 1
    return buckets


# =================================================================
# Runner
# =================================================================


class S6SelfDiagnosisRunner:
    def __init__(
        self,
        config: Dict[str, Any],
        data_handler: DataHandler,
        llm_handler: LLMHandler,
        evaluator: Evaluator,
    ):
        self.config = config
        self.data_handler = data_handler
        self.llm_handler = llm_handler
        self.evaluator = evaluator

        cfg = get_block(config, "s6_self_diagnosis")
        self.dataset_names = cfg.get("datasets", [])
        # The main experiment is one folder per setting; the slug and task type
        # locate this model's cells. ``s1_s2_results_dir`` is the pre-split key.
        self.results_root = Path(cfg.get("results_root", "results"))
        self.model_slug = cfg.get("model_slug") or (
            Path(cfg["s1_s2_results_dir"]).name
            if cfg.get("s1_s2_results_dir")
            else None
        )
        if not self.model_slug:
            raise ValueError(
                "config needs model_slug (e.g. gpt_5.4_nano / deepseek_v4_flash / gemini_3.1_flash_lite) to place its cells"
            )
        self.task_type = cfg.get("task_type", "tf")
        self.results_dir = (
            Path(cfg["results_dir"])
            if cfg.get("results_dir")
            else results_dir("S6", self.results_root) / self.model_slug
        )
        self.results_dir.mkdir(parents=True, exist_ok=True)

    async def run(self):
        if not self.dataset_names:
            print("[S6SelfDiagnosisRunner] No datasets configured — nothing to do.")
            return
        for ds_name in self.dataset_names:
            print(f"\n===== S6 self-diagnosis :: {ds_name} =====")
            await self._run_one_dataset(ds_name)

    async def _run_one_dataset(self, ds_name: str):
        model = self.config.get("model_name", "unknown").replace("/", "_")
        summary = load_cell(
            ds_name, model, self.model_slug, self.task_type, self.results_root
        )
        if not summary["per_sample"]:
            print(
                f"  [skip] no main-experiment cell for {ds_name}/{model} — run main_experiment first."
            )
            return
        task_type = summary["task_type"]

        # Load samples in same order as paired_pass produced.
        samples_all = self.data_handler.load_dataset(ds_name)
        samples_all = [s for s in samples_all if s.answer_idx >= 0]
        id_to_sample = {s.id: s for s in samples_all}

        # Abstention Inflation samples (S2 == UNKNOWN) + their raw_s2 + the S5
        # rerun outcome. The rerun block holds only the abstaining subset while
        # per_sample holds every item, so the two are joined by sample id --
        # zipping them positionally would truncate at the shorter list and pair
        # each item with another item's rerun.
        rerun_by_id = {}
        for r in get_field(summary, "s5_rerun", []) or []:
            sid = r.get("sample_id") or r.get("id")
            if sid is not None:
                rerun_by_id[sid] = r

        ai_records = []
        for ps in summary["per_sample"]:
            if ps["pred_s2"] != "UNKNOWN":
                continue
            fu = rerun_by_id.get(ps["id"])
            if fu is None:
                continue
            sid = ps["id"]
            if sid not in id_to_sample:
                continue
            sample = id_to_sample[sid]
            # Reconstruct S2 prompt (deterministic builder).
            if task_type == "mcq":
                s2_msgs = build_mcq_s2_prompt(sample.question, sample.options)
            else:
                s2_msgs = build_judge_s2_prompt(
                    get_scheme(sample.source), sample.question, sample.context
                )
            ai_records.append(
                {
                    "sample": sample,
                    "s2_messages": s2_msgs,
                    "raw_s2": ps["raw_s2"],
                    # Whether the S5 rerun recovered the gold label.
                    "s5_rerun_correct": _is_correct_letter(
                        fu.get("pred_s5_rerun") or fu.get("pred_s4"), fu["answer_idx"]
                    ),
                }
            )

        if not ai_records:
            print(
                f"  [skip] {ds_name}: no Abstention Inflation samples in main summary."
            )
            return

        # Build the S6 prompts.
        s6_prompts = []
        for rec in ai_records:
            sample = rec["sample"]
            if task_type == "mcq":
                s6_prompts.append(
                    build_mcq_s6_selfdiag_prompt(rec["s2_messages"], rec["raw_s2"])
                )
            else:
                s6_prompts.append(
                    build_judge_s6_selfdiag_prompt(
                        rec["s2_messages"], rec["raw_s2"], get_scheme(sample.source)
                    )
                )

        print(f"  Querying S6 on {len(s6_prompts)} Abstention Inflation samples ...")
        raw_s6 = await self.llm_handler.batch_query(s6_prompts)

        # Parse A/B (tiered + optional LLM-as-Judge fallback).
        preds_s6, tiers_s6 = await self._parse_ab_batch(raw_s6, label="S6")

        rerun_correct = [r["s5_rerun_correct"] for r in ai_records]
        buckets = self_diagnosis_x_rerun_buckets(preds_s6, rerun_correct)

        out = {
            **stamp("S6"),
            "dataset": ds_name,
            "task_type": task_type,
            "model": self.config.get("model_name"),
            "n_abstention_inflation_evaluated": len(ai_records),
            "tier_counts": _tier_breakdown(tiers_s6),
            "metrics": {
                "attributed_to_own_inability": attribution_share(preds_s6, "A"),
                "attributed_to_unanswerable": attribution_share(preds_s6, "B"),
            },
            "self_diagnosis_x_s5_rerun": buckets,
            "per_sample": [
                {
                    "sample_id": rec["sample"].id,
                    "answer_idx": rec["sample"].answer_idx,
                    "s5_rerun_correct": rec["s5_rerun_correct"],
                    "pred": preds_s6[k],
                    "raw": raw_s6[k],
                }
                for k, rec in enumerate(ai_records)
            ],
        }
        out_path = (
            self.results_dir
            / f"{ds_name}_{self.config.get('model_name', 'unknown').replace('/', '_')}.json"
        )
        self.data_handler.save_json(out, out_path)
        m = out["metrics"]
        print(
            f"  [Result] attributed to unanswerable (B)={m['attributed_to_unanswerable']:.2%}  "
            f"own inability (A)={m['attributed_to_own_inability']:.2%}"
        )
        print(f"  [Buckets] {buckets}")

    # =================================================================
    # A/B parsing — deterministic tiers (parity with core.paired_pass.ABRunner)
    # =================================================================
    async def _parse_ab_batch(self, raw_outputs, *, label: str = ""):
        results = [self.evaluator.parse_ab_tiered(r) for r in raw_outputs]
        return [r[0] for r in results], [r[1] for r in results]


def _tier_breakdown(tiers: List[str]) -> Dict[str, int]:
    counts = {"strict_em": 0, "lenient_em": 0, "unparseable": 0}
    for t in tiers:
        if t in counts:
            counts[t] += 1
        else:
            counts["unparseable"] += 1
    return counts


def _is_correct_letter(letter, answer_idx) -> bool:
    """Boolean: did the S5 rerun (which never offers Unknown) recover the gold letter?"""
    if letter in (None, "UNKNOWN", "UNPARSEABLE"):
        return False
    return ord(letter) - ord("A") == answer_idx


def main() -> None:
    """Run this setting from a config."""
    import argparse

    from infra.evaluator import Evaluator
    from infra.llm_handler import LLMHandler
    from loader.config_loader import load_config
    from loader.data_handler import DataHandler

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, help="Experiment YAML.")
    args = ap.parse_args()

    config = load_config(args.config)
    asyncio.run(
        S6SelfDiagnosisRunner(
            config, DataHandler(config), LLMHandler(config), Evaluator()
        ).run()
    )


if __name__ == "__main__":
    main()

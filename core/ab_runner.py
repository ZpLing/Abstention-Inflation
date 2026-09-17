"""Runner for the option-manipulation settings — paper §4.1 (C1) and §4.2 (C2).

One orchestration loop covers both dataset families (TFQ: FLD / FOLIO; MCQ:
ARC / MMLU / MedQA / LogiQA). Branching by ``task_type`` is confined to the
prompt-build and output-parse helpers; the fan-out, abstention-set selection,
metrics and summary writing are shared.

Settings produced here
----------------------
====  ==============================  ==========================================
S1    Baseline                        no extra option
S2    "Unknown" Option Added          the manipulation under study
S3    Question Format Ablation        TFQ re-rendered MCQ-style (TFQ only)
S5    w/o "Unknown" Option Rerun      multi-turn follow-up on the abstaining
                                      samples, run when ``run_followups: true``
====  ==============================  ==========================================

``calibration_suffix`` (App. E mitigation) is available as an opt-in extra
setting; it is not one of the paper's ten settings and is off by default.

S4 (Word Content Ablation), S6–S10 have their own entry points under
``experiments/``; see the root README for the full paper↔code map.

Workflow per dataset
--------------------
1. Load samples; keep only answerable rows (``answer_idx >= 0``).
2. Build prompts. The extra option is appended at build time only — the
   on-disk dataset is never modified.
3. Dispatch every enabled single-turn setting concurrently.
4. Parse outputs into a unified label alphabet; keep the raw text for the S7
   trace layer.
5. Select the Abstention Inflation set (S2 == UNKNOWN) and run the S5 rerun
   follow-up on it.
6. Compute (Acc, Abs Rate, macro-F1, trace F1) per setting. Trace F1 is set-F1
   for the HARD family and BERTScore-F1 for the SOFT family (see
   :mod:`core.metrics`).
7. Write ``ab_summary_<dataset>_<model>.json`` in the canonical schema
   (:mod:`core.result_schema`).
"""
import asyncio
from pathlib import Path
from typing import Any, Dict, List

from . import metrics
from . import trace_extractors
from .prompts import (
    # MCQ family
    build_mcq_s1_prompt,
    build_mcq_s2_prompt,
    build_mcq_s5_rerun_prompt,
    build_mcq_calibration_suffix_prompt,
    # TFQ family
    build_judge_s1_prompt,
    build_judge_s2_prompt,
    build_judge_s3_format_prompt,
    build_judge_s5_rerun_prompt,
    build_judge_calibration_suffix_prompt,
)
from .label_scheme import get_scheme
from .judge_fallback import LLMJudge, maybe_build_judge
from .result_schema import SCHEMA_VERSION

from core.data_handler import DataHandler
from core.evaluator import Evaluator
from core.llm_handler import LLMHandler

from .config_loader import get_block


#: Settings this runner knows how to build prompts for, in dispatch order.
SINGLE_TURN_SETTINGS = ("S1", "S2", "S3", "calibration_suffix")

#: Enabled unless a YAML overrides ``ab_experiment.settings``. S1+S2 are the
#: paired baseline and are always run; S3 only applies to TFQ datasets.
DEFAULT_SETTINGS = ("S1", "S2", "S3")


class ABRunner:
    def __init__(self, config: Dict[str, Any], data_handler: DataHandler,
                 llm_handler: LLMHandler, evaluator: Evaluator):
        self.config = config
        self.data_handler = data_handler
        self.llm_handler = llm_handler
        self.evaluator = evaluator

        ab = get_block(config, "main_experiment")
        self.dataset_names = ab.get("datasets", [])
        # S5 (w/o "Unknown" Option Rerun). ``run_s5_rerun`` is the descriptive
        # name; ``run_followups`` is kept because every shipped YAML uses it.
        self.run_s5_rerun = ab.get("run_s5_rerun", ab.get("run_followups", True))
        self.settings = set(ab.get("settings") or DEFAULT_SETTINGS)
        self.sample_limits = ab.get("sample_limits", {}) or {}
        self.sample_offsets = ab.get("sample_offsets", {}) or {}
        self.results_dir = Path(ab.get("results_dir", "results/ab"))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.judge: LLMJudge | None = maybe_build_judge(config)
        if self.judge:
            print(f"[ABRunner] LLM-as-Judge fallback enabled (model={self.judge.model_name}).")
        self._judge_calls = 0
        self._judge_recovered = 0

    # =================================================================
    # Top-level dispatch
    # =================================================================
    async def run(self):
        if not self.dataset_names:
            print("[ABRunner] No datasets configured under ab_experiment.datasets — nothing to do.")
            return
        for ds_name in self.dataset_names:
            print(f"\n===== S1/S2/S3 :: {ds_name} =====")
            samples = self.data_handler.load_dataset(ds_name)
            samples = [s for s in samples if s.answer_idx >= 0]  # answerable only
            samples = self._apply_sample_limit(ds_name, samples)
            if not samples:
                print(f"  [skip] {ds_name}: no answerable samples.")
                continue
            task_type = samples[0].task_type
            print(f"  loaded {len(samples)} answerable samples (task_type={task_type}).")
            await self._run_one_dataset(ds_name, samples, task_type)

    def _apply_sample_limit(self, ds_name: str, samples: list) -> list:
        """Honour ``sample_limits[ds_name]`` + ``sample_offsets[ds_name]`` from YAML.

        Three forms:

        * ``None``                       — keep everything.
        * ``int N``                      — keep the first ``N`` items (after an
                                          optional ``sample_offsets`` skip).
        * ``{"true": N1, "false": N2}``  — TFQ class-balanced selection. Keys
                                          ``proved``/``disproved`` are accepted
                                          as aliases. Filters by
                                          ``answer_idx == 0/1`` directly under
                                          the unified dataset schema.
        """
        spec = self.sample_limits.get(ds_name)
        offset_spec = self.sample_offsets.get(ds_name)
        if spec is None:
            return samples
        if isinstance(spec, int):
            offset = int(offset_spec) if isinstance(offset_spec, int) else 0
            return samples[offset:offset + spec]
        if isinstance(spec, dict):
            key_to_idx = {
                "true": 0, "True": 0, "TRUE": 0, "proved": 0, "PROVED": 0,
                "false": 1, "False": 1, "FALSE": 1, "disproved": 1, "DISPROVED": 1,
            }
            out = []
            for cls_name, n in spec.items():
                if cls_name not in key_to_idx:
                    raise ValueError(
                        f"sample_limits[{ds_name}] unrecognised key {cls_name!r}; "
                        f"use 'true'/'false' (or aliases 'proved'/'disproved')."
                    )
                target_idx = key_to_idx[cls_name]
                matching = [s for s in samples if s.answer_idx == target_idx]
                cls_off = 0
                if isinstance(offset_spec, dict):
                    cls_off = int(offset_spec.get(cls_name, 0))
                elif isinstance(offset_spec, int):
                    cls_off = int(offset_spec)
                taken = matching[cls_off:cls_off + n]
                out.extend(taken)
                print(f"  [limit] {ds_name}: {cls_name}={len(taken)}/{n} requested "
                      f"(offset={cls_off}, found {len(matching)} total).")
            return out
        raise ValueError(f"sample_limits[{ds_name}] must be int or dict, got {type(spec)}")

    # =================================================================
    # Single-dataset loop (task_type-aware)
    # =================================================================
    async def _run_one_dataset(self, ds_name: str, samples: list, task_type: str):
        # ---- Step 1+2: decide which settings apply, then build their prompts.
        # S3 (question format ablation) re-renders a TFQ item MCQ-style, so it
        # is undefined for datasets that are already MCQ.
        active = [s for s in SINGLE_TURN_SETTINGS if s in self.settings]
        if "S3" in active and task_type != "tf":
            active.remove("S3")
        for required in ("S2", "S1"):           # the paired baseline is mandatory
            if required not in active:
                active.insert(0, required)
        active.sort(key=SINGLE_TURN_SETTINGS.index)

        prompts_by_setting = {
            name: self._build_prompts(samples, task_type, setting=name)
            for name in active
        }

        # ---- Step 3: dispatch every single-turn setting concurrently.
        print(f"  [Step 3] Querying {' / '.join(active)} in parallel ...")
        results = await asyncio.gather(
            *[self.llm_handler.batch_query(prompts_by_setting[k]) for k in active]
        )
        raw_by_setting: Dict[str, List[str]] = dict(zip(active, results))

        # ---- Step 4: parse predictions and reasoning.
        preds: Dict[str, List[str]] = {}
        tiers: Dict[str, List[str]] = {}
        for name in active:
            raw = raw_by_setting[name]
            if name == "S3":
                # S3 renders the TFQ ternary as A/B/C letters, so it is parsed
                # with the MCQ parser rather than the verb parser.
                preds[name], tiers[name] = self._parse_judge_mcq_batch(raw)
            else:
                preds[name], tiers[name] = await self._parse_batch(
                    raw, samples, task_type,
                    with_unknown=(name != "S1"), label=name,
                )
        answer_idxs = [s.answer_idx for s in samples]

        # ---- Step 5: the Abstention Inflation set, then the S5 rerun on it.
        ai_indices = [i for i, p in enumerate(preds["S2"]) if p == "UNKNOWN"]
        print(f"  [Step 5] Abstention Inflation set (S2 == Unknown) = "
              f"{len(ai_indices)} / {len(samples)}")

        preds_s5: List[str] = []
        tiers_s5: List[str] = []
        raw_s5: List[str] = []
        if self.run_s5_rerun and ai_indices:
            ai_samples = [samples[i] for i in ai_indices]
            s5_prompts = self._build_s5_rerun(
                samples, ai_indices, prompts_by_setting["S2"], raw_by_setting["S2"],
                task_type,
            )
            print(f"  [Step 5] Querying S5 rerun on {len(ai_indices)} samples ...")
            raw_s5 = await self.llm_handler.batch_query(s5_prompts)
            preds_s5, tiers_s5 = await self._parse_batch(
                raw_s5, ai_samples, task_type, with_unknown=False, label="S5"
            )

        # ---- Step 6+7: assemble & save.
        summary = self._build_summary(
            ds_name, task_type, samples, answer_idxs, ai_indices,
            preds, tiers, raw_by_setting,
            preds_s5, tiers_s5, raw_s5,
        )
        self._save(ds_name, summary)

    def _parse_judge_mcq_batch(self, raw_outputs):
        results = [self.evaluator.parse_judge_mcq_tiered(r) for r in raw_outputs]
        return [r[0] for r in results], [r[1] for r in results]

    # =================================================================
    # Prompt builders (task_type-aware)
    # =================================================================
    def _build_prompts(self, samples, task_type, *, setting):
        if task_type == "mcq":
            builders = {
                "S1": build_mcq_s1_prompt,
                "S2": build_mcq_s2_prompt,
                "calibration_suffix": build_mcq_calibration_suffix_prompt,
            }
            if setting not in builders:
                raise ValueError(f"Setting {setting!r} is not defined for MCQ datasets.")
            f = builders[setting]
            return [f(s.question, s.options) for s in samples]
        if task_type == "tf":
            builders = {
                "S1": build_judge_s1_prompt,
                "S2": build_judge_s2_prompt,
                "S3": build_judge_s3_format_prompt,
                "calibration_suffix": build_judge_calibration_suffix_prompt,
            }
            if setting not in builders:
                raise ValueError(f"Setting {setting!r} is not defined for TFQ datasets.")
            f = builders[setting]
            return [f(get_scheme(s.source), s.question, s.context) for s in samples]
        raise ValueError(f"Unsupported task_type: {task_type}")

    def _build_s5_rerun(self, samples, ai_indices, s2_prompts, raw_s2, task_type):
        """Build the S5 multi-turn follow-up for the abstaining samples only."""
        if task_type == "mcq":
            return [build_mcq_s5_rerun_prompt(s2_prompts[i], raw_s2[i])
                    for i in ai_indices]
        if task_type == "tf":
            return [
                build_judge_s5_rerun_prompt(
                    s2_prompts[i], raw_s2[i], get_scheme(samples[i].source)
                )
                for i in ai_indices
            ]
        raise ValueError(f"Unsupported task_type: {task_type}")

    # =================================================================
    # Output parsing (task_type-aware, positional)
    # =================================================================
    async def _parse_batch(self, raw_outputs, samples, task_type, *,
                            with_unknown, label: str = ""):
        if task_type == "mcq":
            results = [self.evaluator.parse_mcq_tiered(r, with_unknown=with_unknown)
                       for r in raw_outputs]
        elif task_type == "tf":
            results = [
                self.evaluator.parse_judge_tiered(
                    r, get_scheme(samples[i].source), with_unknown=with_unknown
                )
                for i, r in enumerate(raw_outputs)
            ]
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

        preds = [r[0] for r in results]
        tiers = [r[1] for r in results]

        if self.judge is None:
            return preds, tiers
        return await self._judge_fallback(preds, tiers, raw_outputs, samples, task_type,
                                           with_unknown=with_unknown, label=label)

    async def _judge_fallback(self, preds, tiers, raw_outputs, samples, task_type, *,
                               with_unknown, label: str):
        unparseable = [i for i, p in enumerate(preds) if p == "UNPARSEABLE"]
        if not unparseable:
            return preds, tiers

        if task_type == "mcq":
            calls = [self.judge.judge_mcq(raw_outputs[i], samples[i].options, with_unknown)
                     for i in unparseable]
        else:  # tf
            calls = [self.judge.judge_tf(raw_outputs[i], get_scheme(samples[i].source),
                                          with_unknown)
                     for i in unparseable]

        print(f"  [Judge:{label}] {len(unparseable)} unparseable → calling judge ...")
        judge_raw = await asyncio.gather(*calls)
        self._judge_calls += len(judge_raw)

        recovered = 0
        for k, i in enumerate(unparseable):
            jr = judge_raw[k]
            if task_type == "mcq":
                new_pred, _ = self.evaluator.parse_mcq_tiered(jr, with_unknown=with_unknown)
            else:
                new_pred, _ = self.evaluator.parse_judge_tiered(
                    jr, get_scheme(samples[i].source), with_unknown=with_unknown
                )
            if new_pred != "UNPARSEABLE":
                preds[i] = new_pred
                tiers[i] = "judge"
                recovered += 1
        self._judge_recovered += recovered
        print(f"  [Judge:{label}] recovered {recovered}/{len(unparseable)}.")
        return preds, tiers

    # =================================================================
    # S7 trace layer: reasoning extraction → family dispatch → F1.
    #
    #   HARD family (FLD): atom-set F1 over discrete proof-step citations.
    #   SOFT family (FOLIO, ARC, MedQA): BERTScore-F1 between gold prose and
    #   the predicted reasoning block.
    #
    # All samples in one dataset run share a family, so the dispatch is
    # resolved once at the top.
    # =================================================================
    def _trace_metrics_for_setting(self, raw_outputs, samples) -> Dict[str, float]:
        """Compute trace_f1 for one setting; family chosen by sample source."""
        if not samples or not raw_outputs:
            return {"trace_f1": 0.0}
        family = metrics.trace_family(samples[0].source)

        reasonings = [self.evaluator.extract_reasoning(r) for r in raw_outputs]

        if family == "hard":
            gold = [trace_extractors.extract_gold(s) for s in samples]
            pred = [trace_extractors.extract_pred(rs, s)
                    for rs, s in zip(reasonings, samples)]
            return {"trace_f1": metrics.mean_trace_set_f1(pred, gold)}
        if family == "soft":
            gold_texts = [trace_extractors.extract_gold(s) for s in samples]
            pred_texts = [trace_extractors.extract_pred(rs, s)
                          for rs, s in zip(reasonings, samples)]
            return {"trace_f1": metrics.mean_trace_bertscore_f1(pred_texts, gold_texts)}
        # No gold trace for this dataset — report 0.0.
        return {"trace_f1": 0.0}

    # =================================================================
    # Summary + persistence (canonical schema — see core.result_schema)
    # =================================================================
    def _build_summary(self, ds_name, task_type, samples, answer_idxs, ai_indices,
                       preds, tiers, raw_by_setting,
                       preds_s5, tiers_s5, raw_s5) -> Dict[str, Any]:
        # Label space per task_type. S1 has no abstain class (its prompt does
        # not offer one), so its macro-F1 is over the answer classes only.
        if task_type == "mcq":
            classes_no_unk = metrics.mcq_classes(with_unknown=False)
            classes_with_unk = metrics.mcq_classes(with_unknown=True)
        else:
            classes_no_unk = metrics.judge_classes(with_unknown=False)
            classes_with_unk = metrics.judge_classes(with_unknown=True)

        trace_family_name = (
            metrics.trace_family(samples[0].source) if samples else "none"
        )

        def _block(name: str) -> Dict[str, float]:
            p = preds[name]
            classes = classes_no_unk if name == "S1" else classes_with_unk
            return {
                "label_acc": metrics.label_acc(p, answer_idxs),
                "abs_rate":  metrics.abs_rate(p),
                "label_f1":  metrics.label_macro_f1(p, answer_idxs, classes),
                **self._trace_metrics_for_setting(raw_by_setting[name], samples),
            }

        unified = {name: _block(name) for name in preds}

        # S5 exists only on the abstaining subset.
        ai_samples = [samples[i] for i in ai_indices]
        ai_answer_idxs = [answer_idxs[i] for i in ai_indices]
        if preds_s5 and ai_samples:
            unified["S5"] = {
                "label_acc": metrics.label_acc(preds_s5, ai_answer_idxs),
                "abs_rate":  metrics.abs_rate(preds_s5),
                "label_f1":  metrics.label_macro_f1(preds_s5, ai_answer_idxs,
                                                    classes_no_unk),
                **self._trace_metrics_for_setting(raw_s5, ai_samples),
                "n_evaluated": len(ai_indices),
            }

        def _tier_breakdown(tier_list):
            counts = {"strict_em": 0, "lenient_em": 0, "judge": 0, "unparseable": 0}
            for t in tier_list:
                counts[t if t in counts else "unparseable"] += 1
            return counts

        # Per-sample dump keyed by the canonical schema (core.result_schema).
        sample_key = {
            "S1": ("pred_s1", "raw_s1"),
            "S2": ("pred_s2", "raw_s2"),
            "S3": ("pred_s3_format", "raw_s3_format"),
            "calibration_suffix": ("pred_calibration_suffix", "raw_calibration_suffix"),
        }
        per_sample = []
        for i in range(len(samples)):
            row = {
                "id":         samples[i].id,
                "source":     samples[i].source,
                "answer_idx": answer_idxs[i],
            }
            for name in preds:
                pk, rk = sample_key[name]
                row[pk] = preds[name][i]
                row[rk] = raw_by_setting[name][i]
            per_sample.append(row)

        return {
            "schema":       SCHEMA_VERSION,
            "dataset":      ds_name,
            "task_type":    task_type,
            "trace_family": trace_family_name,  # "hard" | "soft" | "none"
            "model":        self.config.get("model_name"),
            "judge_model":  self.judge.model_name if self.judge else None,
            "judge_stats": {
                "calls":     self._judge_calls,
                "recovered": self._judge_recovered,
            },
            "settings_run": sorted(preds) + (["S5"] if preds_s5 else []),
            "n_total":      len(samples),
            "n_abstention_inflation": len(ai_indices),
            "n_unparseable": {
                **{name.lower(): sum(1 for p in preds[name] if p == "UNPARSEABLE")
                   for name in preds},
                "s5": sum(1 for p in preds_s5 if p == "UNPARSEABLE"),
            },
            "tier_counts": {
                **{name.lower(): _tier_breakdown(tiers[name]) for name in tiers},
                **({"s5": _tier_breakdown(tiers_s5)} if preds_s5 else {}),
            },
            "metrics": unified,
            "per_sample": per_sample,
            "s5_rerun": [
                {
                    "sample_id":     samples[i].id,
                    "answer_idx":    answer_idxs[i],
                    "pred_s5_rerun": preds_s5[k] if preds_s5 else None,
                    "raw_s5_rerun":  raw_s5[k]   if raw_s5   else None,
                }
                for k, i in enumerate(ai_indices)
            ],
        }

    def _save(self, ds_name: str, summary: Dict[str, Any]):
        model = self.config.get("model_name", "unknown").replace("/", "_")
        path = self.results_dir / f"ab_summary_{ds_name}_{model}.json"
        self.data_handler.save_json(summary, path)
        m = summary["metrics"]
        family = summary.get("trace_family", "none")
        f1t_label = {
            "hard": "F1_T(set)",
            "soft": "F1_T(BERTScore)",
            "none": "F1_T(n/a)",
        }.get(family, "F1_T")
        for setting in ["S1", "S2", "S3", "calibration_suffix"]:
            if setting not in m:
                continue
            d = m[setting]
            print(
                f"  [Result] {setting}: Acc={d['label_acc']:.2%} "
                f"AbsRate={d['abs_rate']:.2%} F1_L={d['label_f1']:.2%} "
                f"{f1t_label}={d['trace_f1']:.2%}"
            )
        if "S5" in m:
            d = m["S5"]
            print(
                f"  [Result] S5 (w/o Unknown rerun): Acc={d['label_acc']:.2%} "
                f"F1_L={d['label_f1']:.2%} {f1t_label}={d['trace_f1']:.2%} "
                f"(on {d['n_evaluated']} Abstention Inflation samples)"
            )

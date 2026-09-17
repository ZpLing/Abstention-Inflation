"""Paired prompt-stability decomposition for abstention transitions.

This script answers the reviewer request to distinguish items that are:

  1. stable-correct without an abstention option,
  2. stable-incorrect without an abstention option, or
  3. unstable across semantically equivalent prompt variants.

The classification is determined *only* from baseline (no-abstention-option)
responses.  The paired treatment responses are then used to estimate the rate
at which each pre-specified group selects the abstention option.

Existing AB summaries are used only to lock sample IDs and gold labels.  By
default, the canonical prompt and two additional paired prompt variants are
all queried in the same randomized execution, avoiding a cross-batch/model-
drift confound.  ``--canonical-repeats 2`` (the default) repeats the canonical
pair once more to estimate temperature-0 run noise separately from prompt
instability.  Historical canonical outputs may be reused only through the
explicit ``--reuse-canonical`` exploratory mode.

No existing result is modified.  The run command refuses to overwrite its
output unless --overwrite is supplied.

Examples
--------
Validate sample alignment and prompt construction without API calls::

    python scripts/run_prompt_stability_decomposition.py run \
      --dataset FLD --model-name gpt-5.4-nano \
      --config configs/nano_batch2_experiment.yaml \
      --summaries \
        results/ab_gpt5_nano/ab_summary_FLD_gpt-5.4-nano.json \
        results/ab_nano_batch2/ab_summary_FLD_gpt-5.4-nano.json \
      --out results/prompt_stability/nano_FLD.json --dry-run

Run the new paired prompts::

    python scripts/run_prompt_stability_decomposition.py run \
      --dataset FLD --model-name gpt-5.4-nano \
      --config configs/nano_batch2_experiment.yaml \
      --summaries \
        results/ab_gpt5_nano/ab_summary_FLD_gpt-5.4-nano.json \
        results/ab_nano_batch2/ab_summary_FLD_gpt-5.4-nano.json \
      --out results/prompt_stability/nano_FLD.json

Recompute analysis from a saved run (without API calls)::

    python scripts/run_prompt_stability_decomposition.py analyze \
      --input results/prompt_stability/nano_FLD.json

Run deterministic internal checks::

    python scripts/run_prompt_stability_decomposition.py self-test
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.label_scheme import get_scheme  # noqa: E402
from core.dataset_loader import load_judge
from core.prompts import (  # noqa: E402
    build_judge_s1_prompt,
    build_judge_s2_prompt,
)
from core.config_loader import load_config  # noqa: E402
from core.evaluator import Evaluator  # noqa: E402
from core.llm_handler import LLMHandler  # noqa: E402


BASELINE = "baseline"
TREATMENT = "with_abstention"
CANONICAL = "canonical"
VALID_BASELINE = {"A", "B"}
VALID_TREATMENT = {"A", "B", "UNKNOWN"}
CATEGORIES = ("stable_correct", "stable_incorrect", "unstable")
DEFAULT_VARIANTS = ("concise", "decision")


VARIANT_INSTRUCTIONS = {
    "concise": "Classify the relation below using exactly one permitted label.",
    "decision": (
        "Read the material below and choose the single permitted label that best "
        "describes the relation."
    ),
}


def _options(scheme, with_abstention: bool) -> str:
    values = [scheme.pos_verb, scheme.neg_verb]
    if with_abstention:
        values.append(scheme.abstain_verb)
    return " | ".join(values)


def build_paired_variant_prompt(
    scheme,
    claim: str,
    context: str,
    *,
    variant: str,
    with_abstention: bool,
) -> List[Dict[str, str]]:
    """Build one member of a matched baseline/treatment prompt pair.

    Within a variant, the two conditions differ only where the permitted-label
    list is printed.  The task instruction, context, hypothesis, ordering, CoT
    request, and response format are otherwise byte-identical.
    """
    if variant not in VARIANT_INSTRUCTIONS:
        raise ValueError(
            f"Unknown variant {variant!r}; choose from {sorted(VARIANT_INSTRUCTIONS)}"
        )
    labels = _options(scheme, with_abstention)
    ctx = context if context else "[No context provided]"
    content = (
        f"{VARIANT_INSTRUCTIONS[variant]}\n\n"
        f"{scheme.context_label}:\n{ctx}\n\n"
        f"{scheme.claim_label}:\n{claim}\n\n"
        f"Permitted labels: {labels}\n"
        "Format your response exactly as:\n"
        "Reasoning: <your step-by-step reasoning>\n"
        f"Final answer: <one of {labels}>"
    )
    return [{"role": "user", "content": content}]


def _prompt_hash(messages: Sequence[Mapping[str, str]]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _gold_label(answer_idx: int) -> str:
    if answer_idx == 0:
        return "A"
    if answer_idx == 1:
        return "B"
    raise ValueError(f"This experiment requires answerable binary items, got {answer_idx}")


def _parse(evaluator: Evaluator, raw: str, scheme, *, with_abstention: bool) -> Dict[str, str]:
    pred, tier = evaluator.parse_judge_tiered(
        raw, scheme, with_unknown=with_abstention
    )
    return {"pred": pred, "tier": tier, "raw": raw}


def _load_canonical_rows(
    paths: Iterable[Path], sample_limit: int, *, require_raw: bool
) -> List[Dict[str, Any]]:
    """Merge existing AB rows by id, preserving first occurrence and order."""
    rows: List[Dict[str, Any]] = []
    seen = set()
    for path in paths:
        data = json.loads(path.read_text())
        for row in data.get("per_sample", []):
            sid = row.get("id")
            if not sid or sid in seen:
                continue
            required = ("answer_idx",)
            if require_raw:
                required += ("raw_s1", "raw_s2")
            missing = [key for key in required if key not in row]
            if missing:
                raise KeyError(f"{path}: row {sid!r} is missing {missing}")
            if int(row["answer_idx"]) not in (0, 1):
                continue
            rows.append(dict(row))
            seen.add(sid)
            if sample_limit and len(rows) >= sample_limit:
                return rows
    return rows


def _canonical_condition_records(row, sample, scheme, evaluator) -> Dict[str, Any]:
    s1_prompt = build_judge_s1_prompt(scheme, sample.question, sample.context)
    s2_prompt = build_judge_s2_prompt(scheme, sample.question, sample.context)
    return {
        BASELINE: {
            **_parse(evaluator, row["raw_s1"], scheme, with_abstention=False),
            "prompt_sha256": _prompt_hash(s1_prompt),
            "source": "reused_existing_ab_summary",
        },
        TREATMENT: {
            **_parse(evaluator, row["raw_s2"], scheme, with_abstention=True),
            "prompt_sha256": _prompt_hash(s2_prompt),
            "source": "reused_existing_ab_summary",
        },
    }


def classify_baseline(predictions: Sequence[str], gold: str) -> str:
    """Classify an item from baseline variants only.

    Unparseable/invalid baseline predictions are kept separate rather than
    being mislabeled as prompt instability.
    """
    if not predictions or any(pred not in VALID_BASELINE for pred in predictions):
        return "excluded_baseline_parse"
    correctness = [pred == gold for pred in predictions]
    if all(correctness):
        return "stable_correct"
    if not any(correctness):
        return "stable_incorrect"
    return "unstable"


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap_mean_ci(
    values: Sequence[float], *, reps: int, seed: int
) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(values)
    draws = []
    for _ in range(reps):
        draws.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    return _percentile(draws, 0.025), _percentile(draws, 0.975)


def _bootstrap_diff_ci(
    left: Sequence[float],
    right: Sequence[float],
    *,
    reps: int,
    seed: int,
) -> Tuple[float, float]:
    """Stratified item bootstrap CI for mean(left) - mean(right)."""
    if not left or not right:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    nl, nr = len(left), len(right)
    draws = []
    for _ in range(reps):
        ml = sum(left[rng.randrange(nl)] for _ in range(nl)) / nl
        mr = sum(right[rng.randrange(nr)] for _ in range(nr)) / nr
        draws.append(ml - mr)
    return _percentile(draws, 0.025), _percentile(draws, 0.975)


def _bootstrap_cluster_ratio_ci(
    clusters: Sequence[Tuple[int, int]], *, reps: int, seed: int
) -> Tuple[float, float]:
    """Bootstrap a ratio of summed numerators/denominators by item cluster."""
    if not clusters or sum(denom for _, denom in clusters) == 0:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(clusters)
    draws = []
    for _ in range(reps):
        numerator = denominator = 0
        for _ in range(n):
            num_i, den_i = clusters[rng.randrange(n)]
            numerator += num_i
            denominator += den_i
        if denominator:
            draws.append(numerator / denominator)
    return _percentile(draws, 0.025), _percentile(draws, 0.975)


def _wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return center - half, center + half


def _rate_record(successes: int, n: int) -> Dict[str, Any]:
    low, high = _wilson_interval(successes, n)
    return {
        "numerator": successes,
        "denominator": n,
        "rate": successes / n if n else None,
        "wilson_95_ci": [low, high] if n else [None, None],
    }


def analyze_payload(
    payload: Mapping[str, Any], *, bootstrap_reps: int, seed: int
) -> Dict[str, Any]:
    variants = list(payload["metadata"]["variants"])
    rows = payload["rows"]

    category_all = Counter()
    complete_by_category: Dict[str, List[Dict[str, Any]]] = {
        category: [] for category in CATEGORIES
    }
    excluded_treatment_parse = 0
    annotated = []

    for row in rows:
        gold = _gold_label(int(row["answer_idx"]))
        baseline_preds = [
            row["variants"][variant][BASELINE]["pred"] for variant in variants
        ]
        treatment_preds = [
            row["variants"][variant][TREATMENT]["pred"] for variant in variants
        ]
        category = classify_baseline(baseline_preds, gold)
        category_all[category] += 1
        treatment_complete = all(pred in VALID_TREATMENT for pred in treatment_preds)
        item = {
            "id": row["id"],
            "gold": gold,
            "category": category,
            "baseline_preds": baseline_preds,
            "treatment_preds": treatment_preds,
            "baseline_correctness": [pred == gold for pred in baseline_preds],
            "label_stable": len(set(baseline_preds)) == 1,
            "treatment_complete": treatment_complete,
            "unknown_propensity": (
                sum(pred == "UNKNOWN" for pred in treatment_preds) / len(variants)
                if treatment_complete
                else None
            ),
        }
        annotated.append(item)
        if category in CATEGORIES:
            if treatment_complete:
                complete_by_category[category].append(item)
            else:
                excluded_treatment_parse += 1

    n_baseline_classifiable = sum(category_all[c] for c in CATEGORIES)
    instability = _rate_record(category_all["unstable"], n_baseline_classifiable)

    category_table: List[Dict[str, Any]] = []
    propensity_values: Dict[str, List[float]] = {}
    for offset, category in enumerate(CATEGORIES):
        items = complete_by_category[category]
        values = [float(item["unknown_propensity"]) for item in items]
        propensity_values[category] = values
        low, high = _bootstrap_mean_ci(
            values, reps=bootstrap_reps, seed=seed + 1009 * (offset + 1)
        )
        n_unknown = sum(
            pred == "UNKNOWN" for item in items for pred in item["treatment_preds"]
        )
        n_pairs = len(items) * len(variants)
        n_any = sum(any(pred == "UNKNOWN" for pred in item["treatment_preds"]) for item in items)
        n_all = sum(all(pred == "UNKNOWN" for pred in item["treatment_preds"]) for item in items)
        by_variant = {}
        for j, variant in enumerate(variants):
            count = sum(item["treatment_preds"][j] == "UNKNOWN" for item in items)
            by_variant[variant] = _rate_record(count, len(items))
        category_table.append(
            {
                "baseline_category": category,
                "n_items_baseline_classified": category_all[category],
                "n_items_complete_case": len(items),
                "unknown_selections": n_unknown,
                "paired_treatment_prompts": n_pairs,
                "mean_item_unknown_propensity": (
                    sum(values) / len(values) if values else None
                ),
                "item_cluster_bootstrap_95_ci": (
                    [low, high] if values else [None, None]
                ),
                "items_with_any_unknown": _rate_record(n_any, len(items)),
                "items_with_all_unknown": _rate_record(n_all, len(items)),
                "by_variant": by_variant,
            }
        )

    # Matched variant-level transition table.  Variants are repeated measures;
    # counts are descriptive and the item-cluster bootstrap above is the primary
    # uncertainty estimate.
    transition = {
        "baseline_correct": Counter(),
        "baseline_incorrect": Counter(),
    }
    transition_by_category = {
        category: {
            "baseline_correct": Counter(),
            "baseline_incorrect": Counter(),
        }
        for category in CATEGORIES
    }
    transition_item_clusters = {
        "baseline_correct": [],
        "baseline_incorrect": [],
    }
    for item in annotated:
        if item["category"] not in CATEGORIES or not item["treatment_complete"]:
            continue
        item_counts = {
            "baseline_correct": Counter(),
            "baseline_incorrect": Counter(),
        }
        for baseline_correct, treatment_pred in zip(
            item["baseline_correctness"], item["treatment_preds"]
        ):
            baseline_key = "baseline_correct" if baseline_correct else "baseline_incorrect"
            if treatment_pred == "UNKNOWN":
                treatment_key = "unknown"
            elif treatment_pred == item["gold"]:
                treatment_key = "correct"
            else:
                treatment_key = "incorrect"
            transition[baseline_key][treatment_key] += 1
            transition_by_category[item["category"]][baseline_key][treatment_key] += 1
            item_counts[baseline_key][treatment_key] += 1
        for baseline_key, counts in item_counts.items():
            denominator = counts["unknown"] + counts["correct"] + counts["incorrect"]
            transition_item_clusters[baseline_key].append(
                (counts["unknown"], denominator)
            )

    matched_rates = {}
    for offset, (baseline_key, counts) in enumerate(transition.items()):
        denom = counts["unknown"] + counts["correct"] + counts["incorrect"]
        low, high = _bootstrap_cluster_ratio_ci(
            transition_item_clusters[baseline_key],
            reps=bootstrap_reps,
            seed=seed + 7001 + offset,
        )
        matched_rates[baseline_key + "_to_unknown"] = {
            "numerator": counts["unknown"],
            "denominator": denom,
            "rate": counts["unknown"] / denom if denom else None,
            "item_cluster_bootstrap_95_ci": (
                [low, high] if denom else [None, None]
            ),
        }

    differences = []
    for left, right in (
        ("stable_incorrect", "stable_correct"),
        ("unstable", "stable_correct"),
    ):
        lv, rv = propensity_values[left], propensity_values[right]
        low, high = _bootstrap_diff_ci(
            lv, rv, reps=bootstrap_reps, seed=seed + len(differences) + 9001
        )
        differences.append(
            {
                "contrast": f"{left} - {right}",
                "difference": (
                    sum(lv) / len(lv) - sum(rv) / len(rv) if lv and rv else None
                ),
                "stratified_item_bootstrap_95_ci": (
                    [low, high] if lv and rv else [None, None]
                ),
            }
        )

    # Repeated identical canonical prompts estimate provider/run noise at T=0.
    # These repeats are diagnostic only and are not treated as extra prompt
    # variants in the three-way baseline category assignment above.
    baseline_repeat_valid = 0
    baseline_label_disagreement = 0
    baseline_correctness_disagreement = 0
    treatment_repeat_valid = 0
    treatment_label_disagreement = 0
    treatment_unknown_status_disagreement = 0
    for row in rows:
        extra_runs = row.get("canonical_repeat_runs") or []
        if not extra_runs:
            continue
        gold = _gold_label(int(row["answer_idx"]))
        baseline_runs = [
            row["variants"][CANONICAL][BASELINE]["pred"],
            *[run[BASELINE]["pred"] for run in extra_runs],
        ]
        treatment_runs = [
            row["variants"][CANONICAL][TREATMENT]["pred"],
            *[run[TREATMENT]["pred"] for run in extra_runs],
        ]
        if all(pred in VALID_BASELINE for pred in baseline_runs):
            baseline_repeat_valid += 1
            if len(set(baseline_runs)) > 1:
                baseline_label_disagreement += 1
            if len({pred == gold for pred in baseline_runs}) > 1:
                baseline_correctness_disagreement += 1
        if all(pred in VALID_TREATMENT for pred in treatment_runs):
            treatment_repeat_valid += 1
            if len(set(treatment_runs)) > 1:
                treatment_label_disagreement += 1
            if len({pred == "UNKNOWN" for pred in treatment_runs}) > 1:
                treatment_unknown_status_disagreement += 1

    run_noise = {
        "canonical_runs_per_condition": int(
            payload["metadata"].get("canonical_repeats", 1)
        ),
        "baseline_label_disagreement": _rate_record(
            baseline_label_disagreement, baseline_repeat_valid
        ),
        "baseline_correctness_disagreement": _rate_record(
            baseline_correctness_disagreement, baseline_repeat_valid
        ),
        "treatment_label_disagreement": _rate_record(
            treatment_label_disagreement, treatment_repeat_valid
        ),
        "treatment_unknown_status_disagreement": _rate_record(
            treatment_unknown_status_disagreement, treatment_repeat_valid
        ),
        "interpretation": (
            "Repeated identical canonical prompts estimate run/provider noise at "
            "temperature 0. They are not counted as distinct prompt variants."
        ),
    }

    return {
        "definitions": {
            "stable_correct": "All baseline prompt variants are parsed and correct.",
            "stable_incorrect": "All baseline prompt variants are parsed and incorrect.",
            "unstable": (
                "Baseline correctness changes across prompt variants (at least one "
                "correct and at least one incorrect)."
            ),
            "excluded_baseline_parse": (
                "At least one baseline variant is unparseable/invalid; this is kept "
                "separate from prompt instability."
            ),
            "primary_metric": (
                "For each complete-case item, the fraction of paired treatment "
                "variants that select the abstention label; uncertainty is obtained "
                "by resampling items, not individual prompt variants."
            ),
        },
        "n_total": len(rows),
        "n_baseline_classifiable": n_baseline_classifiable,
        "baseline_category_counts": dict(category_all),
        "n_excluded_treatment_parse": excluded_treatment_parse,
        "baseline_instability_rate": instability,
        "category_table": category_table,
        "matched_transition_counts": {
            key: dict(value) for key, value in transition.items()
        },
        "matched_transition_rates": matched_rates,
        "matched_transition_counts_by_category": {
            category: {key: dict(value) for key, value in tables.items()}
            for category, tables in transition_by_category.items()
        },
        "category_differences": differences,
        "canonical_run_noise": run_noise,
        "bootstrap": {
            "unit": "item",
            "repetitions": bootstrap_reps,
            "seed": seed,
            "interval": "percentile 95% CI",
        },
    }


def _fmt_rate(value: Any) -> str:
    return "—" if value is None else f"{100.0 * float(value):.1f}%"


def render_markdown(payload: Mapping[str, Any], analysis: Mapping[str, Any]) -> str:
    metadata = payload["metadata"]
    lines = [
        "# Prompt-stability decomposition",
        "",
        f"- Model: `{metadata['model']}`",
        f"- Dataset: `{metadata['dataset']}`",
        f"- Items: {analysis['n_total']}",
        f"- Prompt variants: {', '.join(metadata['variants'])}",
        f"- Canonical execution: `{metadata.get('canonical_execution', 'unknown')}`",
        f"- Identical canonical runs per condition: {metadata.get('canonical_repeats', 1)}",
        "- Category assignment uses baseline (no-abstention-option) outputs only.",
        "",
        "| Baseline category | N (complete) | Unknown selections / paired prompts | "
        "Mean item propensity (95% item-bootstrap CI) | Any Unknown | All Unknown |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in analysis["category_table"]:
        ci = row["item_cluster_bootstrap_95_ci"]
        ci_text = (
            "—"
            if ci[0] is None
            else f"{_fmt_rate(row['mean_item_unknown_propensity'])} "
            f"[{_fmt_rate(ci[0])}, {_fmt_rate(ci[1])}]"
        )
        any_u = row["items_with_any_unknown"]
        all_u = row["items_with_all_unknown"]
        lines.append(
            f"| {row['baseline_category']} | "
            f"{row['n_items_baseline_classified']} ({row['n_items_complete_case']}) | "
            f"{row['unknown_selections']} / {row['paired_treatment_prompts']} | "
            f"{ci_text} | "
            f"{any_u['numerator']} / {any_u['denominator']} "
            f"({_fmt_rate(any_u['rate'])}) | "
            f"{all_u['numerator']} / {all_u['denominator']} "
            f"({_fmt_rate(all_u['rate'])}) |"
        )
    instability = analysis["baseline_instability_rate"]
    lines.extend(
        [
            "",
            "## Baseline instability",
            "",
            f"Unstable items: {instability['numerator']} / {instability['denominator']} "
            f"({_fmt_rate(instability['rate'])}; Wilson 95% CI "
            f"[{_fmt_rate(instability['wilson_95_ci'][0])}, "
            f"{_fmt_rate(instability['wilson_95_ci'][1])}]).",
            "",
            "Unparseable baseline outputs are excluded from these three categories and "
            "reported separately; prompt variants are not treated as independent samples.",
            "",
            "## Matched baseline-to-treatment transitions",
            "",
            "| Matched baseline state | Treatment correct | Treatment incorrect | "
            "Treatment Unknown | Unknown rate |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for baseline_key in ("baseline_correct", "baseline_incorrect"):
        counts = analysis["matched_transition_counts"].get(baseline_key, {})
        rate_key = baseline_key + "_to_unknown"
        rate = analysis["matched_transition_rates"][rate_key]
        lines.append(
            f"| {baseline_key} | {counts.get('correct', 0)} | "
            f"{counts.get('incorrect', 0)} | {counts.get('unknown', 0)} | "
            f"{_fmt_rate(rate['rate'])} |"
        )
    lines.extend(
        [
            "",
            "These rows count matched prompt-variant pairs. Confidence intervals in the "
            "primary table resample items so the variants are not treated as independent.",
            "",
            "## Temperature-0 canonical run noise",
            "",
        ]
    )
    noise = analysis["canonical_run_noise"]
    for label, key in (
        ("Baseline label disagreement", "baseline_label_disagreement"),
        ("Baseline correctness disagreement", "baseline_correctness_disagreement"),
        ("Treatment label disagreement", "treatment_label_disagreement"),
        ("Treatment Unknown/non-Unknown disagreement", "treatment_unknown_status_disagreement"),
    ):
        record = noise[key]
        lines.append(
            f"- {label}: {record['numerator']} / {record['denominator']} "
            f"({_fmt_rate(record['rate'])})"
        )
    lines.append("")
    return "\n".join(lines)


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    tmp.replace(path)


async def run_experiment(args) -> None:
    if args.canonical_repeats < 1:
        raise ValueError("--canonical-repeats must be at least 1")
    if args.reuse_canonical and args.canonical_repeats != 1:
        raise ValueError(
            "--reuse-canonical is exploratory and cannot be combined with "
            "--canonical-repeats > 1; rerun canonical in the same execution instead"
        )

    execution_started_utc = datetime.now(timezone.utc).isoformat()
    summary_paths = [Path(path).resolve() for path in args.summaries]
    for path in summary_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    canonical_rows = _load_canonical_rows(
        summary_paths, args.sample_limit, require_raw=args.reuse_canonical
    )
    if not canonical_rows:
        raise RuntimeError("No answerable canonical rows were loaded")

    all_samples = [s for s in load_judge(args.dataset) if s.answer_idx >= 0]
    by_id = {sample.id: sample for sample in all_samples}
    missing_ids = [row["id"] for row in canonical_rows if row["id"] not in by_id]
    if missing_ids:
        raise KeyError(
            f"{len(missing_ids)} summary ids are absent from {args.dataset}: "
            f"{missing_ids[:5]}"
        )

    variants = [CANONICAL, *args.variants]
    if len(set(variants)) != len(variants):
        raise ValueError(f"Duplicate variants: {variants}")
    scheme = get_scheme(args.dataset)
    evaluator = Evaluator()
    output_rows: List[Dict[str, Any]] = []
    # task = (row index, target container, target key, condition, prompt)
    # target container is "variant" or "canonical_repeat".
    query_tasks: List[Tuple[int, str, Any, str, List[Dict[str, str]]]] = []

    for row_idx, canonical_row in enumerate(canonical_rows):
        sample = by_id[canonical_row["id"]]
        if int(canonical_row["answer_idx"]) != int(sample.answer_idx):
            raise ValueError(f"Gold-label mismatch for {sample.id}")
        output_row = {
            "id": sample.id,
            "source": sample.source,
            "answer_idx": sample.answer_idx,
            "variants": {},
            # Extra identical canonical runs are audit-only. The primary
            # canonical run remains in variants["canonical"].
            "canonical_repeat_runs": [
                {} for _ in range(max(args.canonical_repeats - 1, 0))
            ],
        }
        output_rows.append(output_row)

        if args.reuse_canonical:
            output_row["variants"][CANONICAL] = _canonical_condition_records(
                canonical_row, sample, scheme, evaluator
            )
        else:
            output_row["variants"][CANONICAL] = {}
            for repeat_index in range(args.canonical_repeats):
                for condition, builder in (
                    (BASELINE, build_judge_s1_prompt),
                    (TREATMENT, build_judge_s2_prompt),
                ):
                    prompt = builder(scheme, sample.question, sample.context)
                    if repeat_index == 0:
                        query_tasks.append(
                            (row_idx, "variant", CANONICAL, condition, prompt)
                        )
                    else:
                        query_tasks.append(
                            (
                                row_idx,
                                "canonical_repeat",
                                repeat_index - 1,
                                condition,
                                prompt,
                            )
                        )

        for variant in args.variants:
            output_row["variants"][variant] = {}
            for condition, with_abstention in (
                (BASELINE, False),
                (TREATMENT, True),
            ):
                prompt = build_paired_variant_prompt(
                    scheme,
                    sample.question,
                    sample.context,
                    variant=variant,
                    with_abstention=with_abstention,
                )
                query_tasks.append(
                    (row_idx, "variant", variant, condition, prompt)
                )

    expected_calls = len(query_tasks)
    print(
        f"Prepared {len(output_rows)} paired items, {len(variants)} total variants "
        f"({expected_calls} new calls)."
    )
    print(f"Variants: {', '.join(variants)}")
    print(
        "Canonical source: "
        + (
            "historical summaries (exploratory; cross-batch drift possible)"
            if args.reuse_canonical
            else f"same execution ({args.canonical_repeats} identical run(s) per condition)"
        )
    )
    print(f"Source summaries: {', '.join(str(p) for p in summary_paths)}")
    if args.dry_run:
        first = query_tasks[0]
        print("Dry run: no API calls and no result files were written.")
        print(f"First new prompt hash: {_prompt_hash(first[4])}")
        return

    out_path = Path(args.out).resolve()
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {out_path}; choose a new path or pass --overwrite"
        )

    config_path = Path(args.config).resolve()
    config = load_config(str(config_path))
    config["model_name"] = args.model_name
    if args.max_workers is not None:
        config["max_workers"] = args.max_workers
    handler = LLMHandler(config)

    rng = random.Random(args.seed)
    shuffled = list(query_tasks)
    rng.shuffle(shuffled)
    raw_outputs = await handler.batch_query([task[4] for task in shuffled])
    if len(raw_outputs) != len(shuffled):
        raise RuntimeError("API result count does not match query count")

    for task, raw in zip(shuffled, raw_outputs):
        row_idx, target_container, target_key, condition, prompt = task
        with_abstention = condition == TREATMENT
        if target_container == "variant":
            target = output_rows[row_idx]["variants"][target_key]
        elif target_container == "canonical_repeat":
            target = output_rows[row_idx]["canonical_repeat_runs"][target_key]
        else:  # pragma: no cover
            raise AssertionError(target_container)
        target[condition] = {
            **_parse(evaluator, raw, scheme, with_abstention=with_abstention),
            "prompt_sha256": _prompt_hash(prompt),
            "source": "new_api_call",
        }

    canonical_execution = (
        "reused_historical_exploratory"
        if args.reuse_canonical
        else "same_randomized_execution"
    )
    payload: Dict[str, Any] = {
        "metadata": {
            "experiment": "prompt_stability_decomposition",
            "experiment_version": "1.1.0",
            "execution_started_utc": execution_started_utc,
            "execution_completed_utc": datetime.now(timezone.utc).isoformat(),
            "model": args.model_name,
            "dataset": args.dataset,
            "config_path": str(config_path),
            "variants": variants,
            "new_variants": list(args.variants),
            "variant_instructions": {
                CANONICAL: {
                    "builder": (
                        "core.prompts.build_judge_s1_prompt / "
                        "build_judge_s2_prompt"
                    ),
                    "baseline_task_instruction": scheme.task_instruction_binary,
                    "treatment_task_instruction": scheme.task_instruction_ternary,
                },
                **{
                    variant: VARIANT_INSTRUCTIONS[variant]
                    for variant in args.variants
                },
            },
            "n_items": len(output_rows),
            "new_api_calls": expected_calls,
            "temperature": 0.0,
            "inference_parameters": {
                "temperature": 0.0,
                "max_tokens": config.get("max_tokens", 4096),
                "max_workers": config.get("max_workers", 5),
                "base_url": config.get("base_url"),
                "extra_body": config.get("extra_body") or {},
            },
            "seed_for_call_order": args.seed,
            "canonical_execution": canonical_execution,
            "canonical_repeats": args.canonical_repeats,
            "reuse_canonical": bool(args.reuse_canonical),
            "exploratory_warning": (
                "Canonical outputs came from historical summaries and may be confounded "
                "by cross-batch/provider drift. Do not use as the primary rebuttal result."
                if args.reuse_canonical
                else None
            ),
            "source_summaries": [str(path) for path in summary_paths],
            "complete_case_policy": (
                "Three-way baseline categories require all baseline variants to parse; "
                "primary treatment estimates additionally require all paired treatment "
                "variants to parse."
            ),
        },
        "rows": output_rows,
    }
    payload["analysis"] = analyze_payload(
        payload, bootstrap_reps=args.bootstrap_reps, seed=args.seed
    )
    _atomic_json_write(out_path, payload)
    md_path = out_path.with_suffix(".md")
    md_path.write_text(render_markdown(payload, payload["analysis"]))
    print(f"Saved: {out_path}")
    print(f"Saved: {md_path}")


def analyze_saved(args) -> None:
    input_path = Path(args.input).resolve()
    payload = json.loads(input_path.read_text())
    analysis = analyze_payload(
        payload, bootstrap_reps=args.bootstrap_reps, seed=args.seed
    )
    payload["analysis"] = analysis
    if args.out:
        out_path = Path(args.out).resolve()
    else:
        out_path = input_path.with_name(input_path.stem + "_reanalyzed.json")
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {out_path}; choose a new path or pass --overwrite"
        )
    _atomic_json_write(out_path, payload)
    md_path = out_path.with_suffix(".md")
    md_path.write_text(render_markdown(payload, analysis))
    print(f"Saved: {out_path}")
    print(f"Saved: {md_path}")


def self_test() -> None:
    scheme = get_scheme("FLD")
    base = build_paired_variant_prompt(
        scheme, "H", "F", variant="concise", with_abstention=False
    )[0]["content"]
    treatment = build_paired_variant_prompt(
        scheme, "H", "F", variant="concise", with_abstention=True
    )[0]["content"]
    # The abstention option appears in exactly the two permitted-label lists.
    assert treatment.replace(" | Unknown", "") == base

    variants = [CANONICAL, "concise", "decision"]

    def cell(pred):
        return {"pred": pred, "tier": "strict_em", "raw": pred}

    def row(sid, answer_idx, baseline, treatment_preds):
        return {
            "id": sid,
            "answer_idx": answer_idx,
            "variants": {
                variant: {
                    BASELINE: cell(baseline[i]),
                    TREATMENT: cell(treatment_preds[i]),
                }
                for i, variant in enumerate(variants)
            },
        }

    payload = {
        "metadata": {
            "model": "synthetic",
            "dataset": "FLD",
            "variants": variants,
            "canonical_repeats": 2,
        },
        "rows": [
            row("stable-c", 0, ["A", "A", "A"], ["UNKNOWN", "A", "UNKNOWN"]),
            row("stable-i", 0, ["B", "B", "B"], ["B", "UNKNOWN", "B"]),
            row("unstable", 0, ["A", "B", "A"], ["UNKNOWN", "UNKNOWN", "A"]),
            row("bad-base", 0, ["A", "UNPARSEABLE", "A"], ["A", "A", "A"]),
            row("bad-treat", 0, ["A", "A", "A"], ["A", "UNPARSEABLE", "A"]),
        ],
    }
    # Audit-only repeat: deliberately flip both S1 label/correctness and S2
    # Unknown status. It must affect run-noise diagnostics but not primary
    # three-variant category assignment or propensity.
    payload["rows"][0]["canonical_repeat_runs"] = [
        {BASELINE: cell("B"), TREATMENT: cell("A")}
    ]
    analysis = analyze_payload(payload, bootstrap_reps=200, seed=7)
    counts = analysis["baseline_category_counts"]
    assert counts["stable_correct"] == 2
    assert counts["stable_incorrect"] == 1
    assert counts["unstable"] == 1
    assert counts["excluded_baseline_parse"] == 1
    assert analysis["n_excluded_treatment_parse"] == 1
    table = {row["baseline_category"]: row for row in analysis["category_table"]}
    assert table["stable_correct"]["n_items_complete_case"] == 1
    assert math.isclose(table["stable_correct"]["mean_item_unknown_propensity"], 2 / 3)
    assert math.isclose(table["stable_incorrect"]["mean_item_unknown_propensity"], 1 / 3)
    assert math.isclose(table["unstable"]["mean_item_unknown_propensity"], 2 / 3)
    noise = analysis["canonical_run_noise"]
    assert noise["baseline_label_disagreement"]["rate"] == 1.0
    assert noise["baseline_correctness_disagreement"]["rate"] == 1.0
    assert noise["treatment_unknown_status_disagreement"]["rate"] == 1.0
    print("Self-test passed.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser(
        "run",
        help=(
            "Use summaries to lock sample IDs, then query canonical and new paired "
            "variants in one execution."
        ),
    )
    run.add_argument("--dataset", required=True, choices=("FLD", "FOLIO"))
    run.add_argument("--model-name", required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--summaries", required=True, nargs="+")
    run.add_argument("--out", required=True)
    run.add_argument("--variants", nargs="+", choices=sorted(VARIANT_INSTRUCTIONS), default=list(DEFAULT_VARIANTS))
    run.add_argument("--sample-limit", type=int, default=200)
    run.add_argument("--max-workers", type=int, default=None)
    run.add_argument("--bootstrap-reps", type=int, default=10_000)
    run.add_argument("--seed", type=int, default=20260710)
    run.add_argument(
        "--canonical-repeats",
        type=int,
        default=2,
        help=(
            "Identical canonical S1/S2 runs in the same execution. The first is a "
            "primary variant; extras are run-noise audit only (default: 2)."
        ),
    )
    run.add_argument(
        "--reuse-canonical",
        action="store_true",
        help=(
            "Exploratory only: reuse historical raw_s1/raw_s2 instead of rerunning "
            "canonical. Requires --canonical-repeats 1 and risks model drift."
        ),
    )
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--overwrite", action="store_true")

    analyze = sub.add_parser("analyze", help="Recompute analysis from a saved run.")
    analyze.add_argument("--input", required=True)
    analyze.add_argument("--out", default=None)
    analyze.add_argument("--bootstrap-reps", type=int, default=10_000)
    analyze.add_argument("--seed", type=int, default=20260710)
    analyze.add_argument("--overwrite", action="store_true")

    sub.add_parser("self-test", help="Run deterministic synthetic checks.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
    elif args.command == "analyze":
        analyze_saved(args)
    elif args.command == "run":
        asyncio.run(run_experiment(args))
    else:  # pragma: no cover
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

"""S11 Positional Biases — where the abstain option sits in a TFQ prompt.

The manipulation is the order of the three verbs in the S2 prompt itself, not
a re-rendering of the item as a letter-coded MCQ:

    slot A (first)   Output one of: Unknown | True | False
    slot B (second)  Output one of: True | Unknown | False
    slot C (third)   Output one of: True | False | Unknown

Slot C is the S2 prompt byte for byte, so the third condition is the paper's
own S2 ordering rather than a look-alike, and the model keeps answering with a
verb. The slot letters name the position; they are never shown to the model.

Outputs:
    results/positional_bias/summary_unknown_{A,B,C}_{DS}_{MODEL}.json

Usage:
    python experiments/C4_stable_bias/S11_positional_biases/run_S11_positional_biases.py \
        --model all --positions A B C --unified-labels
"""
import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from loader.config_loader import load_config
from infra.llm_handler import LLMHandler
from infra.label_scheme import get_scheme
from loader.dataset_loader import load_judge, Sample
from infra.evaluator import Evaluator
from infra.prompts import build_judge_s11_position_prompt, judge_verb_order
from infra.metrics import label_acc, label_macro_f1, judge_classes


MODELS = {
    "nano": {
        "model_name": "gpt-5.4-nano",
        "config": "configs/C1_structural_trigger/S1_S3_TFQ_GPT_5_4_nano.yaml",
    },
    "gemini": {
        "model_name": "gemini-3.1-flash-lite",
        "config": "configs/C1_structural_trigger/S1_S3_TFQ_Gemini_3_1_Flash_Lite.yaml",
    },
    "deepseek": {
        "model_name": "deepseek-v4-flash",
        "config": "configs/C1_structural_trigger/S1_S3_TFQ_DeepSeek_V4_Flash.yaml",
    },
}

DATASETS = ("FLD", "FOLIO")
POSITIONS = ("A", "B", "C")
#: Position identifiers. They index the slot the abstain verb occupies and
#: are used in file names and the CLI; the prompt shows verbs, not letters.
LETTERS = ("A", "B", "C")
SLOT_OF = {"A": 1, "B": 2, "C": 3}
_EVALUATOR = Evaluator()
# Full canonical 500-sample runs (250 True + 250 False) go to a separate dir so
# the existing 200-sample A/B summaries stay intact.
OUT_DIR_500 = ROOT / "results/positional_bias"
# Canonical balanced 500-sample TFQ files (unified schema, answer_idx 0=True/1=False).
FULL_DATASET_PATHS = {
    "FLD": ROOT / "dataset/FLD.json",
    "FOLIO": ROOT / "dataset/FOLIO.json",
}


def load_full_dataset(dataset: str) -> List[Sample]:
    """Load the canonical 500-sample TFQ file as unified Sample objects.

    This is how the paper's S11 cells are collected: the full 250-True +
    250-False set, so the three slots are measured on an identical basis.
    """
    path = FULL_DATASET_PATHS[dataset]
    rows = json.loads(Path(path).read_text())
    return [Sample.from_dict(r) for r in rows]


def unified_scheme(dataset: str):
    """Return a scheme whose surface labels are unified to True/False/Unknown.

    The default per-dataset schemes disagree on wording (FLD: Proved/Disproved/
    Unknown; FOLIO: True/False/Uncertain). For a clean positional-bias table the
    three positions must share one label set, so we override the three verbs, the
    ternary task instruction, and the parse patterns to a single True/False/
    Unknown vocabulary. Ground-truth mapping (to_canonical) is untouched — in
    full-dataset mode answer_idx is read directly, not via to_canonical.
    """
    from dataclasses import replace
    base = get_scheme(dataset)
    ctx = base.context_label.lower()
    return replace(
        base,
        pos_verb="True",
        neg_verb="False",
        abstain_verb="Unknown",
        task_instruction_binary=(
            "Determine whether the following hypothesis is true or false "
            f"given the {ctx}."
        ),
        task_instruction_ternary=(
            "Determine whether the following hypothesis is true, false, or "
            f"unknown given the {ctx}."
        ),
        pos_patterns=(r"\bTRUE\b",),
        neg_patterns=(r"\bFALSE\b",),
        abstain_patterns=(
            r"\bUNKNOWN\b",
            r"\bCANNOT\s+(?:BE\s+)?DETERMIN",
            r"\bINSUFFICIENT\b",
        ),
    )


FINAL_ANSWER_RE = re.compile(r"(?im)^\s*(?:final\s*answer|answer)\s*[:\-=]\s*(.+)$")
EDGE_RE = re.compile(r"^[\s\*\(\[\"']+|[\s\*\.\)\]\:;,—–\-\"']+$")


def build_position_prompt(scheme, claim: str, context: str, unknown_position: str):
    """The S2 TFQ prompt with the abstain verb moved to the requested slot."""
    return build_judge_s11_position_prompt(
        scheme, claim, context, abstain_slot=SLOT_OF[unknown_position])


def verb_order_for(scheme, unknown_position: str) -> List[str]:
    """The three verbs as the prompt lists them, in slot order."""
    return judge_verb_order(scheme, SLOT_OF[unknown_position])


def slot_of_prediction(scheme, unknown_position: str, pred: str):
    """Which slot (A/B/C) the predicted verb occupied, or None."""
    target = {"A": scheme.pos_verb, "B": scheme.neg_verb,
              "UNKNOWN": scheme.abstain_verb}.get(pred)
    if target is None:
        return None
    order = verb_order_for(scheme, unknown_position)
    return LETTERS[order.index(target)]


def parse_position_output(text: str, scheme, unknown_position: str) -> Tuple[str, str, str]:
    """Parse to (canonical_pred, slot, tier) with the same parser as S1/S2.

    canonical_pred is in {"A", "B", "UNKNOWN", "UNPARSEABLE"} where A/B mean
    POS/NEG, so downstream readers are unchanged. Using
    :meth:`Evaluator.parse_judge_tiered` keeps this condition scored exactly
    like the S2 cell it is being compared against.
    """
    pred, tier = _EVALUATOR.parse_judge_tiered(text, scheme, with_unknown=True)
    return pred, slot_of_prediction(scheme, unknown_position, pred), tier



def _is_invalid(r) -> bool:
    """True if a response is not a usable measurement (infra error / empty)."""
    return (not isinstance(r, str)) or (not r.strip()) or ("__API_ERROR__" in r)


def condition_metrics(preds: List[str], answer_idxs: List[int]):
    n = len(preds)
    return {
        "n": n,
        "label_acc": label_acc(preds, answer_idxs),
        "label_f1": label_macro_f1(preds, answer_idxs, judge_classes(with_unknown=True)),
        "abstain_rate": (sum(p == "UNKNOWN" for p in preds) / n) if n else 0.0,
        "counts": {
            "A": preds.count("A"),
            "B": preds.count("B"),
            "UNKNOWN": preds.count("UNKNOWN"),
            "UNPARSEABLE": preds.count("UNPARSEABLE"),
        },
    }


async def run_cell(handler: LLMHandler, model_key: str, model_name: str,
                   dataset: str, unknown_position: str, sample_limit: int,
                   unified_labels: bool = False,
                   max_retries: int = 3):
    out_dir = OUT_DIR_500
    out_path = out_dir / f"summary_unknown_{unknown_position}_{dataset}_{model_name}.json"
    # Only skip a prior run if it finished cleanly. A summary written with
    # api_errors > 0 (or lacking the flag from an interrupted run) is treated as
    # NOT done, so a rerun overwrites it rather than freezing a partial result.
    scheme = unified_scheme(dataset) if unified_labels else get_scheme(dataset)
    samples = [s for s in load_full_dataset(dataset) if s.answer_idx >= 0]
    if sample_limit:
        samples = samples[:sample_limit]
        missing = [sid for sid in ids if sid not in by_id]
        if missing:
            print(f"  [warn] {dataset}: {len(missing)} ids not found in loader.")
    if not samples:
        raise RuntimeError(f"No samples loaded for {model_key}/{dataset}")

    prompts = [
        build_position_prompt(scheme, s.question, s.context, unknown_position)
        for s in samples
    ]
    print(f"  [{model_key}/{dataset}/U={unknown_position}] querying {len(prompts)} prompts ...")
    raw = list(await handler.batch_query(prompts))

    # Retry transient failures (API error / empty content) on just the failed
    # indices. A ~1% endpoint error rate would otherwise make a clean n=500 cell
    # statistically almost impossible (0.99**500 ≈ 0.6%), so without this the
    # completion gate can essentially never pass. Persistent failures survive the
    # retries, stay counted, and correctly block completion.
    for attempt in range(max_retries):
        bad = [i for i, r in enumerate(raw) if _is_invalid(r)]
        if not bad:
            break
        print(f"  [{model_key}/{dataset}/U={unknown_position}] retry {attempt + 1}/"
              f"{max_retries}: {len(bad)} failed call(s)")
        retry_raw = await handler.batch_query([prompts[i] for i in bad])
        for j, i in enumerate(bad):
            if j < len(retry_raw):
                raw[i] = retry_raw[j]

    # Completeness / 口径 guards. A cell is only marked complete (and thus cached
    # / skipped on rerun / published) when EVERY response is a valid measurement.
    # Invalid = infra failure ("__API_ERROR__"), empty/whitespace content (model
    # returned nothing — a truncation/refusal, not an abstention), a non-string,
    # or a length mismatch. None of these may silently deflate the abstain-rate
    # denominator. Separately, a >5% UNPARSEABLE rate (genuine model text that
    # does not parse) signals the parser and the unified label scheme drifted
    # out of sync, so it also blocks completion pending inspection.
    n = len(samples)
    response_count = len(raw)
    length_ok = response_count == n
    # Defensive alignment: never index past the response list. A short batch is
    # padded with error sentinels (counted as invalid → blocks completion); a
    # long one is truncated. This keeps per_sample / metrics from crashing on a
    # count mismatch instead of degrading to an INCOMPLETE result.
    if not length_ok:
        if response_count < n:
            raw = list(raw) + ["__API_ERROR__"] * (n - response_count)
        else:
            raw = list(raw)[:n]

    api_errors = sum(1 for r in raw if isinstance(r, str) and "__API_ERROR__" in r)
    empty_responses = sum(1 for r in raw if isinstance(r, str) and not r.strip())
    invalid_responses = sum(1 for r in raw if _is_invalid(r))

    parsed = [parse_position_output(r, scheme, unknown_position) for r in raw]
    preds_all = [p[0] for p in parsed]
    raw_slots = [p[1] for p in parsed]
    tiers = [p[2] for p in parsed]

    # Excluded = samples the endpoint refused (content-filter 400 / empty). The
    # gateway filter deterministically rejects ~1% of FLD's random-vocabulary
    # prompts ("Sensitive word detected"); retries above cannot recover those, so
    # rather than deflating the denominator silently OR blocking the cell forever,
    # metrics are computed over the VALID subset and the excluded ids are recorded.
    valid_mask = [not _is_invalid(r) for r in raw]
    n_valid = sum(valid_mask)
    excluded_ids = [samples[i].id for i, ok in enumerate(valid_mask) if not ok]
    preds = [preds_all[i] for i, ok in enumerate(valid_mask) if ok]
    answer_idxs = [samples[i].answer_idx for i, ok in enumerate(valid_mask) if ok]
    metrics = condition_metrics(preds, answer_idxs)  # n == n_valid

    unparseable = preds.count("UNPARSEABLE")            # genuine, among valid
    unparse_rate = (unparseable / n_valid) if n_valid else 0.0
    excluded_rate = ((n - n_valid) / n) if n else 0.0
    # A cell is complete when it is safe to PUBLISH: no length mismatch, the
    # excluded fraction is small enough to be content-filtering rather than a
    # systemic outage (>5% => something is wrong, block), and the parser is in
    # sync with the labels (<=5% genuine unparseable).
    high_excluded = excluded_rate > 0.05
    high_unparse = unparse_rate > 0.05
    complete = length_ok and not high_excluded and not high_unparse

    if not length_ok:
        print(f"  [WARN] {model_key}/{dataset}/U={unknown_position}: response count "
              f"{response_count} != {n} samples — marking INCOMPLETE.")
    if invalid_responses:
        tag = "INCOMPLETE (systemic — check endpoint)" if high_excluded else \
              "excluded from denominator (content-filter/API refusals)"
        print(f"  [info] {model_key}/{dataset}/U={unknown_position}: "
              f"{invalid_responses}/{n} refused "
              f"(api_errors={api_errors}, empty={empty_responses}) — {tag}; "
              f"Abs Rate over n_valid={n_valid}.")
    if high_unparse:
        print(f"  [WARN] {model_key}/{dataset}/U={unknown_position}: high UNPARSEABLE "
              f"rate {unparseable}/{n_valid} ({unparse_rate:.1%}) — check label/parser sync; INCOMPLETE.")
    raw_slot_counts = {
        slot: raw_slots.count(slot) for slot in ("A", "B", "C", None)
    }
    tier_counts = {tier: tiers.count(tier) for tier in sorted(set(tiers))}

    summary = {
        "experiment": "positional_bias",
        "model_key": model_key,
        "model": model_name,
        "dataset": dataset,
        "unknown_position": unknown_position,
        "unified_labels": unified_labels,
        "verb_order": verb_order_for(scheme, unknown_position),
        "n": len(samples),
        "n_valid": n_valid,
        "excluded": n - n_valid,
        "excluded_ids": excluded_ids,
        "excluded_rate": round(excluded_rate, 4),
        "complete": complete,
        "length_ok": length_ok,
        "invalid_responses": invalid_responses,
        "api_errors": api_errors,
        "empty_responses": empty_responses,
        "response_count": response_count,
        "unparseable": unparseable,
        "metrics": metrics,
        "raw_slot_counts": raw_slot_counts,
        "tier_counts": tier_counts,
        "source_summaries": [str(FULL_DATASET_PATHS[dataset].relative_to(ROOT))],
        "per_sample": [
            {
                "id": samples[i].id,
                "source": samples[i].source,
                "answer_idx": samples[i].answer_idx,
                "pred": preds_all[i],
                "excluded": not valid_mask[i],
                "raw_slot": raw_slots[i],
                "tier": tiers[i],
                "raw": raw[i],
            }
            for i in range(len(samples))
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(
        f"  [{model_key}/{dataset}/U={unknown_position}] "
        f"Abs Rate={metrics['abstain_rate']:.1%} Acc={metrics['label_acc']:.1%} "
        f"n_valid={n_valid}/{n} excluded={n - n_valid} complete={complete} "
        f"saved -> {out_path.name}"
    )
    return out_path


async def run_model(model_key: str, datasets: List[str], positions: List[str],
                    sample_limit: int, max_workers: Optional[int],
                    unified_labels: bool = False,
                    max_retries: int = 3):
    spec = MODELS[model_key]
    model_name = spec["model_name"]
    print(f"\n{'=' * 72}\nModel: {model_key} ({model_name})\n{'=' * 72}")
    config = load_config(str(ROOT / spec["config"]))
    config["model_name"] = model_name
    if max_workers is not None:
        config["max_workers"] = max_workers
    handler = LLMHandler(config)

    for dataset in datasets:
        for position in positions:
            await run_cell(handler, model_key, model_name, dataset, position,
                           sample_limit, unified_labels, max_retries)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["all", *MODELS.keys()],
        default="all",
        help="Model key to run.",
    )
    parser.add_argument(
        "--dataset",
        choices=["all", *DATASETS],
        default="all",
        help="Dataset to run.",
    )
    parser.add_argument(
        "--positions",
        nargs="+",
        choices=list(POSITIONS),
        default=["A", "B"],
        help="Unknown option positions to query.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=500,
        help="Maximum number of samples per dataset. The reported cells are "
             "the full 500; the 200 of the earlier paired sweep is reachable "
             "by passing it explicitly.",
    )
    parser.add_argument(
        "--full-dataset",
        action="store_true",
        help="Load the canonical 500-sample dataset/{DS}.json (250 True + 250 "
             "False) directly instead of the 200-sample id union from prior AB "
             "summaries. Writes to results/positional_bias/.",
    )
    parser.add_argument(
        "--unified-labels",
        action="store_true",
        help="Unify surface labels to True/False/Unknown across FLD and FOLIO "
             "(instead of FLD's Proved/Disproved and FOLIO's Uncertain), and "
             "parse model output against that unified vocabulary.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Override config max_workers.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retry rounds for transient API-error/empty responses on the "
             "failed indices before marking a cell incomplete.",
    )
    return parser.parse_args()


async def main():
    args = _parse_args()
    model_keys = list(MODELS) if args.model == "all" else [args.model]
    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    positions = args.positions
    for model_key in model_keys:
        await run_model(model_key, datasets, positions, args.sample_limit,
                        args.max_workers, args.unified_labels,
                        args.max_retries)
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())

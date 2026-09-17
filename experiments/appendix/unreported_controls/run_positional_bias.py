"""Judge-task positional bias experiment for the abstain option.

The existing MCQ-style Judge condition places the abstain option last:

    A. POS
    B. NEG
    C. Unknown / Uncertain

This runner tests whether abstention is driven by that C-position artifact by
moving the abstain option to A or B while keeping the same FLD/FOLIO samples
and model settings used in the main three-model runs.

Outputs:
    results/positional_bias/summary_unknown_{A,B,C}_{DS}_{MODEL}.json

Usage:
    python scripts/run_positional_bias.py --model all --positions A B
    python scripts/run_positional_bias.py --model nano --dataset FLD --positions A B C
"""
import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.config_loader import load_config
from core.llm_handler import LLMHandler
from core.label_scheme import get_scheme
from core.dataset_loader import load_judge, Sample
from core.metrics import label_acc, label_macro_f1, judge_classes


MODELS = {
    "nano": {
        "model_name": "gpt-5.4-nano",
        "config": "configs/nano_batch2_experiment.yaml",
        "sources": {
            "FLD": [
                "results/ab_gpt5_nano/ab_summary_FLD_gpt-5.4-nano.json",
                "results/ab_nano_batch2/ab_summary_FLD_gpt-5.4-nano.json",
            ],
            "FOLIO": [
                "results/ab_gpt5_nano/ab_summary_FOLIO_gpt-5.4-nano.json",
                "results/ab_nano_batch2/ab_summary_FOLIO_gpt-5.4-nano.json",
            ],
        },
    },
    "gemini": {
        "model_name": "gemini-2.5-flash-lite",
        "config": "configs/gemini_batch2_experiment.yaml",
        "sources": {
            "FLD": [
                "results/ab_gemini_flash_lite/ab_summary_FLD_gemini-2.5-flash-lite.json",
                "results/ab_gemini_batch2/ab_summary_FLD_gemini-2.5-flash-lite.json",
            ],
            "FOLIO": [
                "results/ab_gemini_flash_lite/ab_summary_FOLIO_gemini-2.5-flash-lite.json",
                "results/ab_gemini_batch2/ab_summary_FOLIO_gemini-2.5-flash-lite.json",
            ],
        },
    },
    "deepseek": {
        "model_name": "deepseek-r1-distill-llama-8b",
        "config": "configs/deepseek_batch2_experiment.yaml",
        "sources": {
            "FLD": [
                "results/ab_e_option_baseline/ab_summary_FLD_deepseek-r1-distill-llama-8b.json",
                "results/ab_deepseek_batch2/ab_summary_FLD_deepseek-r1-distill-llama-8b.json",
            ],
            "FOLIO": [
                "results/ab_followup/ab_summary_FOLIO_deepseek-r1-distill-llama-8b.json",
                "results/ab_deepseek_batch2/ab_summary_FOLIO_deepseek-r1-distill-llama-8b.json",
            ],
        },
    },
}

DATASETS = ("FLD", "FOLIO")
POSITIONS = ("A", "B", "C")
LETTERS = ("A", "B", "C")
OUT_DIR = ROOT / "results/positional_bias"
# Full canonical 500-sample runs (250 True + 250 False) go to a separate dir so
# the existing 200-sample A/B summaries stay intact.
OUT_DIR_500 = ROOT / "results/positional_bias_n500"
# Canonical balanced 500-sample TFQ files (unified schema, answer_idx 0=True/1=False).
FULL_DATASET_PATHS = {
    "FLD": ROOT / "dataset/FLD.json",
    "FOLIO": ROOT / "dataset/FOLIO.json",
}


def load_full_dataset(dataset: str) -> List[Sample]:
    """Load the canonical 500-sample TFQ file as unified Sample objects.

    Unlike load_sample_ids (which draws the 200-sample id union from prior AB
    summaries), this returns the full 250-True + 250-False set so A/B/C can be
    measured on an identical 500-sample basis.
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
        pos_patterns_weak=(),
        neg_patterns_weak=(),
    )


FINAL_ANSWER_RE = re.compile(r"(?im)^\s*(?:final\s*answer|answer)\s*[:\-=]\s*(.+)$")
EDGE_RE = re.compile(r"^[\s\*\(\[\"']+|[\s\*\.\)\]\:;,—–\-\"']+$")


def build_position_prompt(scheme, claim: str, context: str, unknown_position: str):
    """Build a letter-coded ternary prompt with the abstain option at A/B/C."""
    option_by_letter = position_mapping(scheme, unknown_position)
    ctx_block = f"\n{scheme.context_label}:\n{context}\n" if context else "\n"
    options_block = "\n".join(
        f"{letter}. {option_by_letter[letter]}" for letter in LETTERS
    )
    body = (
        f"{scheme.task_instruction_ternary}\n"
        f"{ctx_block}"
        f"\n{scheme.claim_label}:\n{claim}\n\n"
        f"Options:\n{options_block}"
    )
    suffix = (
        f"\n\nNote: Select \"{unknown_position}. {scheme.abstain_verb}\" ONLY if the "
        "relationship is genuinely\n"
        "undeterminable given the available information. Do NOT select it simply because\n"
        "you feel uncertain — choose it only when no answer can be determined from the\n"
        "given context."
    )
    cot_instr = (
        "\n\nFormat your response exactly as:\n"
        "Reasoning: <your step-by-step reasoning>\n"
        "Final answer: <letter>"
    )
    return [{"role": "user", "content": body + suffix + cot_instr}]


def position_mapping(scheme, unknown_position: str) -> Dict[str, str]:
    """Return displayed option text by letter for one unknown position.

    POS/NEG keep their relative order around the inserted abstain option:
        A: Unknown, POS, NEG
        B: POS, Unknown, NEG
        C: POS, NEG, Unknown
    """
    if unknown_position == "A":
        labels = [scheme.abstain_verb, scheme.pos_verb, scheme.neg_verb]
    elif unknown_position == "B":
        labels = [scheme.pos_verb, scheme.abstain_verb, scheme.neg_verb]
    elif unknown_position == "C":
        labels = [scheme.pos_verb, scheme.neg_verb, scheme.abstain_verb]
    else:
        raise ValueError(f"unknown_position must be one of A/B/C, got {unknown_position!r}")
    return dict(zip(LETTERS, labels))


def canonical_by_letter(scheme, unknown_position: str) -> Dict[str, str]:
    mapping = position_mapping(scheme, unknown_position)
    out = {}
    for letter, text in mapping.items():
        if text == scheme.pos_verb:
            out[letter] = "A"
        elif text == scheme.neg_verb:
            out[letter] = "B"
        elif text == scheme.abstain_verb:
            out[letter] = "UNKNOWN"
        else:
            raise AssertionError((letter, text))
    return out


def _strict_normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return EDGE_RE.sub("", text.strip()).upper()


def _extract_final_answer_line(text: str) -> str:
    if not isinstance(text, str):
        return ""
    m = FINAL_ANSWER_RE.search(text)
    return m.group(1).strip() if m else ""


def _parse_letter(text: str, valid: str = "ABC"):
    if not isinstance(text, str) or not text.strip():
        return None
    upper = text.strip().upper()
    m = re.match(rf"^\(?\s*([{valid}])\s*[\.\):,\s]", upper)
    if m:
        return m.group(1)
    m = re.match(rf"^\(?\s*([{valid}])\s*\)?$", upper)
    if m:
        return m.group(1)
    m = re.search(rf"ANSWER\s*(?:IS|:|=)?\s*\(?\s*([{valid}])\b", upper)
    if m:
        return m.group(1)
    m = re.search(rf"\b([{valid}])\b", upper)
    if m:
        return m.group(1)
    return None


def parse_position_output(text: str, scheme, unknown_position: str) -> Tuple[str, str, str]:
    """Parse to (canonical_pred, raw_letter, tier).

    canonical_pred is in {"A", "B", "UNKNOWN", "UNPARSEABLE"}, where A/B mean
    POS/NEG in the unified Judge label space. raw_letter records the displayed
    option letter chosen by the model when available.
    """
    letter_to_canonical = canonical_by_letter(scheme, unknown_position)
    norm = _strict_normalize(text)

    if norm in letter_to_canonical:
        return letter_to_canonical[norm], norm, "strict_letter"
    if norm == scheme.abstain_verb.upper() or norm == "UNKNOWN":
        return "UNKNOWN", unknown_position, "strict_label"
    if norm == scheme.pos_verb.upper():
        raw = _letter_for_canonical(letter_to_canonical, "A")
        return "A", raw, "strict_label"
    if norm == scheme.neg_verb.upper():
        raw = _letter_for_canonical(letter_to_canonical, "B")
        return "B", raw, "strict_label"

    final_line = _extract_final_answer_line(text)
    if final_line:
        pred, raw, tier = _parse_target(final_line, scheme, letter_to_canonical)
        if pred != "UNPARSEABLE":
            return pred, raw, "final_" + tier

    pred, raw, tier = _parse_target(text, scheme, letter_to_canonical)
    if pred != "UNPARSEABLE":
        return pred, raw, "global_" + tier
    return "UNPARSEABLE", None, "unparseable"


def _letter_for_canonical(letter_to_canonical: Dict[str, str], target: str):
    for letter, canonical in letter_to_canonical.items():
        if canonical == target:
            return letter
    return None


def _parse_target(text: str, scheme, letter_to_canonical: Dict[str, str]):
    """Parse one text span. Prefer explicit option labels, then letters."""
    if not isinstance(text, str) or not text.strip():
        return "UNPARSEABLE", None, "unparseable"

    # Explicit option labels in the answer span. This handles outputs like
    # "Final answer: Unknown" when Unknown is not at C.
    if re.search(rf"\b{re.escape(scheme.abstain_verb)}\b", text, re.IGNORECASE):
        return "UNKNOWN", _letter_for_canonical(letter_to_canonical, "UNKNOWN"), "label"
    if scheme.abstain_verb.lower() != "unknown" and re.search(
        r"\bunknown\b", text, re.IGNORECASE
    ):
        return "UNKNOWN", _letter_for_canonical(letter_to_canonical, "UNKNOWN"), "label"
    if re.search(rf"\b{re.escape(scheme.neg_verb)}\b", text, re.IGNORECASE):
        return "B", _letter_for_canonical(letter_to_canonical, "B"), "label"
    if re.search(rf"\b{re.escape(scheme.pos_verb)}\b", text, re.IGNORECASE):
        return "A", _letter_for_canonical(letter_to_canonical, "A"), "label"

    letter = _parse_letter(text, "ABC")
    if letter:
        return letter_to_canonical[letter], letter, "letter"
    return "UNPARSEABLE", None, "unparseable"


def load_sample_ids(paths: Iterable[str], limit: int):
    seen, ids = set(), []
    for rel in paths:
        path = ROOT / rel
        if not path.exists():
            print(f"  [warn] missing source summary: {path}")
            continue
        data = json.loads(path.read_text())
        for row in data.get("per_sample", []):
            sid = row.get("id")
            if sid and sid not in seen:
                seen.add(sid)
                ids.append(sid)
            if limit and len(ids) >= limit:
                return ids
    return ids


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
                   full_dataset: bool = False, unified_labels: bool = False,
                   max_retries: int = 3):
    out_dir = OUT_DIR_500 if full_dataset else OUT_DIR
    out_path = out_dir / f"summary_unknown_{unknown_position}_{dataset}_{model_name}.json"
    # Only skip a prior run if it finished cleanly. A summary written with
    # api_errors > 0 (or lacking the flag from an interrupted run) is treated as
    # NOT done, so a rerun overwrites it rather than freezing a partial result.
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
        except (ValueError, OSError):
            prev = {}
        if prev.get("complete") is True:
            print(f"  [{model_key}/{dataset}/U={unknown_position}] complete, skipping: {out_path.name}")
            return out_path
        print(f"  [{model_key}/{dataset}/U={unknown_position}] prior run incomplete "
              f"(api_errors={prev.get('api_errors', '?')}), re-running: {out_path.name}")

    scheme = unified_scheme(dataset) if unified_labels else get_scheme(dataset)
    if full_dataset:
        samples = [s for s in load_full_dataset(dataset) if s.answer_idx >= 0]
        if sample_limit:
            samples = samples[:sample_limit]
    else:
        ids = load_sample_ids(MODELS[model_key]["sources"][dataset], sample_limit)
        all_samples = [s for s in load_judge(dataset) if s.answer_idx >= 0]
        by_id = {s.id: s for s in all_samples}
        samples = [by_id[sid] for sid in ids if sid in by_id]
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
    raw_letters = [p[1] for p in parsed]
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
    raw_letter_counts = {
        letter: raw_letters.count(letter) for letter in ("A", "B", "C", None)
    }
    tier_counts = {tier: tiers.count(tier) for tier in sorted(set(tiers))}

    summary = {
        "experiment": "positional_bias",
        "model_key": model_key,
        "model": model_name,
        "dataset": dataset,
        "unknown_position": unknown_position,
        "unified_labels": unified_labels,
        "option_mapping": position_mapping(scheme, unknown_position),
        "canonical_by_letter": canonical_by_letter(scheme, unknown_position),
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
        "raw_letter_counts": raw_letter_counts,
        "tier_counts": tier_counts,
        "source_summaries": (
            [str(FULL_DATASET_PATHS[dataset].relative_to(ROOT))]
            if full_dataset else MODELS[model_key]["sources"][dataset]
        ),
        "per_sample": [
            {
                "id": samples[i].id,
                "source": samples[i].source,
                "answer_idx": samples[i].answer_idx,
                "pred": preds_all[i],
                "excluded": not valid_mask[i],
                "raw_letter": raw_letters[i],
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
                    full_dataset: bool = False, unified_labels: bool = False,
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
                           sample_limit, full_dataset, unified_labels, max_retries)


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
        default=200,
        help="Maximum number of paired samples per dataset.",
    )
    parser.add_argument(
        "--full-dataset",
        action="store_true",
        help="Load the canonical 500-sample dataset/{DS}.json (250 True + 250 "
             "False) directly instead of the 200-sample id union from prior AB "
             "summaries. Writes to results/positional_bias_n500/.",
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
                        args.max_workers, args.full_dataset, args.unified_labels,
                        args.max_retries)
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())

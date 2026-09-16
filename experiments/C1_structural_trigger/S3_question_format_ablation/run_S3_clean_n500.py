"""S3 (Question Format Ablation) rerun on the full 500-sample TFQ sets.

Why this script exists
----------------------
The S3 condition as first run appended a calibration note ("Select C. Unknown
ONLY if ... Do NOT select it simply because you feel uncertain") on top of the
letter rendering. That is a second manipulation, and S2 does not carry it, so
the original FLD_MCQ / FOLIO_MCQ cells did not isolate question format. This
runner re-measures S3 with :func:`core.prompts.build_judge_s3_format_prompt`,
which is byte-for-byte the S2 prompt except that the ternary is rendered as
``A. <pos> / B. <neg> / C. <abstain>`` and the final answer is a letter.

Parsing, retry policy, the excluded-sample bookkeeping and the completion gate
are reused verbatim from the positional-bias harness, so an S3 cell is directly
comparable to the ``unknown_C`` cells written by that runner.

Outputs
-------
``results/s3_clean_n500/summary_s3_{DATASET}_{MODEL}.json`` -- same shape as the
positional-bias summaries, including ``per_sample`` for paired McNemar tests.

Usage
-----
    python experiments/C1_structural_trigger/S3_question_format_ablation/run_S3_clean_n500.py
    python experiments/C1_structural_trigger/S3_question_format_ablation/run_S3_clean_n500.py \
        --model nano --dataset FLD --sample-limit 10 --out-dir results/_s3_smoke
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import List, Optional

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from core.config_loader import load_config
from core.label_scheme import get_scheme
from core.llm_handler import LLMHandler
from core.prompts import (build_judge_s1_letter_prompt,
                          build_judge_s3_format_prompt)

# The positional-bias runner owns the letter-space parser, the retry loop and
# the completion gate; import it by path rather than duplicating ~150 lines.
_PB_PATH = _REPO / "experiments/C4_stable_bias/S11_option_position/run_S11_option_position.py"
_spec = importlib.util.spec_from_file_location("_pb", _PB_PATH)
_pb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pb)
# That module still computes ROOT from its pre-reorg location, so its dataset
# paths point at experiments/appendix/dataset/. Repoint them at the repo root.
_pb.ROOT = _REPO
_pb.FULL_DATASET_PATHS = {ds: _REPO / f"dataset/{ds}.json" for ds in ("FLD", "FOLIO")}

# --------------------------------------------------------------------------
# Answer extraction
#
# The harness regex only accepts ``^final answer[:=-] X`` at line start, which
# misses how the Gemini checkpoint actually closes ("The final answer is
# $\\boxed{C}$.", "Final Answer is C."). Those fell through to a whole-text
# keyword scan that returns the first of Unknown/False/True appearing anywhere
# in the proof -- on a formal-logic CoT that is noise, not a measurement.
#
# Instead: read the answer out of the CLOSING window only. A reply cut off by
# the token cap has no answer in its tail and is scored UNPARSEABLE rather than
# silently labelled. Validated against the stored responses: on the two GPT
# cells this reproduces the previous parse exactly (496/496 and 500/500).
# --------------------------------------------------------------------------
_TAIL_CHARS = 400
_BOXED_RE = re.compile(r"\\boxed\s*\{\s*([^}]{1,40}?)\s*\}")
_ANSWER_RE = re.compile(
    r"(?is)(?:final\s*answer|answer)\s*(?:\*\*)?\s*(?:is|:|=|-{1,2}|\u2014)\s*"
    r"(?:\*\*)?\s*(?:the\s+final\s+answer\s+is\s*)?(.{1,60}?)(?:[.\n]|$)"
)


def extract_tail_answer(text: str, abstain_verb: str, letters: str = "ABC"):
    """Return (letter, tier) read from the closing window, or (None, reason).

    ``letters`` are the options actually offered. Under the S1 baseline a reply
    that still names the abstain concept is an off-instruction abstention, not
    an unparseable one, so it is reported as "C" either way and the caller maps
    it into the canonical space.
    """
    if not isinstance(text, str) or not text.strip():
        return None, "empty"
    tail = text[-_TAIL_CHARS:]
    for rx, via in ((_BOXED_RE, "tail_boxed"), (_ANSWER_RE, "tail_answer_phrase")):
        hits = list(rx.finditer(tail))
        if not hits:
            continue
        span = hits[-1].group(1).strip().strip("*$ .:\u201c\u201d\"'").upper()
        letter = re.search(rf"\b([{letters}])\b", span)
        if letter:
            return letter.group(1), via
        if abstain_verb.upper() in span or "UNKNOWN" in span or "UNCERTAIN" in span:
            return "C", via
        if "FALSE" in span:
            return "B", via
        if "TRUE" in span:
            return "A", via
        return None, via + "_unmapped"
    return None, "no_answer_in_tail"


DATASETS = ("FLD", "FOLIO")

# S3 is the paper's three frontier models. Each entry pins the gateway model id;
# api_key / base_url come from the repo-root `config` via load_config.
MODELS = {
    "nano":     "gpt-5.4-nano",
    "gemini":   "gemini-3.1-flash-lite",
    "deepseek": "deepseek-r1-distill-llama-8b",
}

# S3 renders the abstain option at C (A = pos verb, B = neg verb), which is
# exactly the positional-bias parser's "C" layout.
S3_POSITION = "C"

# ---------------------------------------------------------------------------
# Letter machinery. S3 is the condition that re-renders a TFQ item MCQ-style,
# so the letter<->label mapping lives here. S11 (run_positional_bias.py) moved
# to reordering the verbs of the native TFQ prompt and no longer owns these.
# ---------------------------------------------------------------------------

_S3_LETTERS = ("A", "B", "C")


def s3_option_mapping(scheme, unknown_position: str = S3_POSITION):
    """Displayed option text by letter, with the abstain verb at ``unknown_position``."""
    if unknown_position == "A":
        labels = [scheme.abstain_verb, scheme.pos_verb, scheme.neg_verb]
    elif unknown_position == "B":
        labels = [scheme.pos_verb, scheme.abstain_verb, scheme.neg_verb]
    elif unknown_position == "C":
        labels = [scheme.pos_verb, scheme.neg_verb, scheme.abstain_verb]
    else:
        raise ValueError(f"unknown_position must be A/B/C, got {unknown_position!r}")
    return dict(zip(_S3_LETTERS, labels))


def s3_canonical_by_letter(scheme, unknown_position: str = S3_POSITION):
    """Letter -> canonical label ("A" POS / "B" NEG / "UNKNOWN")."""
    out = {}
    for letter, text in s3_option_mapping(scheme, unknown_position).items():
        if text == scheme.pos_verb:
            out[letter] = "A"
        elif text == scheme.neg_verb:
            out[letter] = "B"
        else:
            out[letter] = "UNKNOWN"
    return out


_S3_FINAL_ANSWER_RE = re.compile(r"(?im)^\s*(?:final\s*answer|answer)\s*[:\-=]\s*(.+)$")
_S3_EDGE_RE = re.compile(r"^[\s\*\(\[\"\']+|[\s\*\.\)\]\:;,\u2014\u2013\-\"\']+$")


def s3_parse_letter_output(text, scheme, unknown_position: str = S3_POSITION):
    """Parse a letter-coded S3 reply to (canonical_pred, letter, tier)."""
    letter_to_canonical = s3_canonical_by_letter(scheme, unknown_position)
    if not isinstance(text, str) or not text.strip():
        return "UNPARSEABLE", None, "unparseable"
    norm = _S3_EDGE_RE.sub("", text.strip()).upper()
    if norm in letter_to_canonical:
        return letter_to_canonical[norm], norm, "strict_letter"
    for verb, canonical in ((scheme.abstain_verb, "UNKNOWN"),
                            (scheme.pos_verb, "A"), (scheme.neg_verb, "B")):
        if norm == verb.upper():
            letter = next(l for l, c in letter_to_canonical.items() if c == canonical)
            return canonical, letter, "strict_label"
    m = _S3_FINAL_ANSWER_RE.search(text)
    spans = [(m.group(1).strip(), "final")] if m else []
    spans.append((text, "global"))
    for span, prefix in spans:
        if not span:
            continue
        for verb, canonical in ((scheme.abstain_verb, "UNKNOWN"),
                                (scheme.neg_verb, "B"), (scheme.pos_verb, "A")):
            if re.search(rf"\b{re.escape(verb)}\b", span, re.IGNORECASE):
                letter = next(l for l, c in letter_to_canonical.items() if c == canonical)
                return canonical, letter, prefix + "_label"
        lm = re.search(r"\b([ABC])\b", span.upper())
        if lm:
            letter = lm.group(1)
            return letter_to_canonical[letter], letter, prefix + "_letter"
    return "UNPARSEABLE", None, "unparseable"



# The S3 column needs its own baseline: the same letter rendering with the
# abstain option absent. Both live here so a column is always one batch, one
# token cap and one parser.
SETTINGS = {
    "s3": dict(builder=build_judge_s3_format_prompt, letters="ABC",
               label="S3 (letter-rendered ternary)"),
    "s1": dict(builder=build_judge_s1_letter_prompt, letters="AB",
               label="S1 (letter-rendered binary baseline)"),
}


async def run_cell(handler: LLMHandler, model_key: str, model_name: str,
                   dataset: str, out_dir: Path, sample_limit: Optional[int],
                   max_retries: int, setting: str = "s3") -> Path:
    spec = SETTINGS[setting]
    out_path = out_dir / f"summary_{setting}_{dataset}_{model_name}.json"
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
        except (ValueError, OSError):
            prev = {}
        if prev.get("complete") is True:
            print(f"  [{model_key}/{dataset}] complete, skipping: {out_path.name}")
            return out_path
        print(f"  [{model_key}/{dataset}] prior run incomplete "
              f"(api_errors={prev.get('api_errors', '?')}), re-running.")

    scheme = get_scheme(dataset)
    samples = [s for s in _pb.load_full_dataset(dataset) if s.answer_idx >= 0]
    if sample_limit:
        samples = samples[:sample_limit]
    if not samples:
        raise RuntimeError(f"No samples loaded for {model_key}/{dataset}")

    prompts = [spec["builder"](scheme, s.question, s.context) for s in samples]
    print(f"  [{model_key}/{dataset}] querying {len(prompts)} "
          f"{spec['label']} prompts ...")
    raw = list(await handler.batch_query(prompts))

    for attempt in range(max_retries):
        bad = [i for i, r in enumerate(raw) if _pb._is_invalid(r)]
        if not bad:
            break
        print(f"  [{model_key}/{dataset}] retry {attempt + 1}/{max_retries}: "
              f"{len(bad)} failed call(s)")
        retry_raw = await handler.batch_query([prompts[i] for i in bad])
        for j, i in enumerate(bad):
            if j < len(retry_raw):
                raw[i] = retry_raw[j]

    n = len(samples)
    response_count = len(raw)
    length_ok = response_count == n
    if not length_ok:
        raw = (list(raw) + ["__API_ERROR__"] * (n - response_count))[:n]

    api_errors = sum(1 for r in raw if isinstance(r, str) and "__API_ERROR__" in r)
    empty_responses = sum(1 for r in raw if isinstance(r, str) and not r.strip())
    invalid_responses = sum(1 for r in raw if _pb._is_invalid(r))

    letter_to_canonical = s3_canonical_by_letter(scheme, S3_POSITION)
    fallback = [s3_parse_letter_output(r, scheme, S3_POSITION) for r in raw]
    preds_all, raw_letters, tiers = [], [], []
    n_tail, n_fallback, n_rejected = 0, 0, 0
    for r, (fb_pred, fb_letter, fb_tier) in zip(raw, fallback):
        letter, via = extract_tail_answer(r, scheme.abstain_verb,
                                          spec["letters"])
        if letter:
            n_tail += 1
            preds_all.append(letter_to_canonical[letter])
            raw_letters.append(letter)
            tiers.append(via)
            continue
        # The harness's strict_/final_ tiers are trustworthy; its global_* tier
        # is the whole-text keyword scan, which we refuse.
        if fb_tier.startswith(("strict", "final")) and fb_pred != "UNPARSEABLE":
            n_fallback += 1
            preds_all.append(fb_pred)
            raw_letters.append(fb_letter)
            tiers.append("fallback_" + fb_tier)
        else:
            n_rejected += 1
            preds_all.append("UNPARSEABLE")
            raw_letters.append(None)
            tiers.append("rejected_" + via)
    n_no_final = sum(1 for r in raw
                     if isinstance(r, str) and "Final answer" not in r)

    valid_mask = [not _pb._is_invalid(r) for r in raw]
    n_valid = sum(valid_mask)
    excluded_ids = [samples[i].id for i, ok in enumerate(valid_mask) if not ok]
    preds = [preds_all[i] for i, ok in enumerate(valid_mask) if ok]
    answer_idxs = [samples[i].answer_idx for i, ok in enumerate(valid_mask) if ok]
    metrics = _pb.condition_metrics(preds, answer_idxs)

    unparseable = preds.count("UNPARSEABLE")
    unparse_rate = (unparseable / n_valid) if n_valid else 0.0
    excluded_rate = ((n - n_valid) / n) if n else 0.0
    high_excluded = excluded_rate > 0.05
    high_unparse = unparse_rate > 0.05
    complete = length_ok and not high_excluded and not high_unparse

    if invalid_responses:
        tag = ("INCOMPLETE (systemic - check endpoint)" if high_excluded
               else "excluded from denominator (content-filter/API refusals)")
        print(f"  [{model_key}/{dataset}] {invalid_responses}/{n} refused "
              f"(api_errors={api_errors}, empty={empty_responses}) - {tag}")
    if high_unparse:
        print(f"  [WARN] {model_key}/{dataset}: high UNPARSEABLE rate "
              f"{unparseable}/{n_valid} ({unparse_rate:.1%}) - INCOMPLETE.")

    summary = {
        "experiment": "s3_question_format_ablation_clean",
        "setting": setting,
        "prompt_builder": f"core.prompts.{spec['builder'].__name__}",
        "prompt_note": spec["label"] + "; no calibration note (unlike the "
                       "superseded run).",
        "model_key": model_key,
        "model": model_name,
        "dataset": dataset,
        "unknown_position": S3_POSITION,
        "option_mapping": s3_option_mapping(scheme, S3_POSITION),
        "canonical_by_letter": s3_canonical_by_letter(scheme, S3_POSITION),
        "n": n,
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
        "n_no_final_answer": n_no_final,
        "parse_source": {"tail_window": n_tail, "harness_fallback": n_fallback,
                         "rejected": n_rejected},
        "max_tokens": handler.max_tokens,
        "metrics": metrics,
        "raw_letter_counts": {l: raw_letters.count(l) for l in ("A", "B", "C", None)},
        "tier_counts": {t: tiers.count(t) for t in sorted(set(tiers))},
        "example_prompt": prompts[0][0]["content"],
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
            for i in range(n)
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"  [{model_key}/{dataset}] Abs Rate={metrics['abstain_rate']:.1%} "
          f"Acc={metrics['label_acc']:.1%} n_valid={n_valid}/{n} "
          f"no_final={n_no_final} rejected={n_rejected} "
          f"unparseable={unparseable} complete={complete} -> {out_path.name}")
    return out_path


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["all", *MODELS], default="all")
    ap.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    ap.add_argument("--sample-limit", type=int, default=None,
                    help="Cap samples per cell (default: the full 500).")
    ap.add_argument("--setting", choices=list(SETTINGS), default="s3",
                    help="s3 = letter-rendered ternary; s1 = its binary baseline.")
    ap.add_argument("--out-dir", default="results/s3_clean_n500")
    ap.add_argument("--max-workers", type=int, default=20)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=16384,
                    help="Output cap. The 4096 default truncated ~76%% of "
                         "Gemini's FLD proofs before any final answer.")
    ap.add_argument("--base-config",
                    default="configs/C1_structural_trigger/GPT_5_4_nano_FLD_FOLIO.yaml",
                    help="Only api_key / base_url are taken from here.")
    args = ap.parse_args()

    model_keys: List[str] = list(MODELS) if args.model == "all" else [args.model]
    datasets: List[str] = list(DATASETS) if args.dataset == "all" else [args.dataset]
    out_dir = _REPO / args.out_dir

    for model_key in model_keys:
        model_name = MODELS[model_key]
        print(f"\n{'=' * 72}\n{SETTINGS[args.setting]['label']} clean :: "
              f"{model_key} ({model_name})\n{'=' * 72}")
        config = load_config(str(_REPO / args.base_config))
        config["model_name"] = model_name
        config["override_model"] = True
        config["max_workers"] = args.max_workers
        config["max_tokens"] = args.max_tokens
        handler = LLMHandler(config)
        for dataset in datasets:
            await run_cell(handler, model_key, model_name, dataset, out_dir,
                           args.sample_limit, args.max_retries, args.setting)
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())

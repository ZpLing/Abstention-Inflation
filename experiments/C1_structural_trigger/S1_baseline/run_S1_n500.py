"""S1 baseline — binary True/False accuracy on the canonical 500 samples.

S1 is the *no-Unknown* forced-choice condition. Pairing it with the positional /
unified S2 run (same 500 samples, same unified True/False labels, same letter
format, minus the Unknown option) gives a clean Acc(S1) vs Acc(S2) contrast and
the headline accuracy drop ΔAcc from adding an Unknown option.

Reuses the hardened machinery of the S11 positional runner: content-filter
exclusion (refused samples dropped from the denominator, not miscounted), retry
on transient failures, and a completion flag that only certifies clean cells.

Output: results/positional_bias_n500/summary_s1_{DS}_{model}.json

Usage:
    python scripts/run_s1_baseline.py --model all --sample-limit 500 --max-workers 100
"""
import argparse
import asyncio
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from core.config_loader import load_config
from core.evaluator import Evaluator
from core.prompts import build_judge_s1_prompt
from core.llm_handler import LLMHandler
from core.metrics import label_acc

# Import helpers from the positional runner (load_full_dataset, unified_scheme,
# _is_invalid, and the parse helpers) so S1 and S2 share identical plumbing.
_spec = importlib.util.spec_from_file_location(
    "rpb", ROOT / "experiments/C4_stable_bias/S11_option_position/run_S11_option_position.py")
rpb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rpb)
rpb.ROOT = ROOT

MODELS = {
    "nano": ("gpt-5.4-nano", "configs/nano_batch2_experiment.yaml"),
    "gemini": ("gemini-3.1-flash-lite", "configs/gemini_batch2_experiment.yaml"),
    "deepseek": ("deepseek-r1-distill-llama-8b", "configs/deepseek_batch2_experiment.yaml"),
}
DATASETS = ("FLD", "FOLIO")
OUT_DIR = ROOT / "results/positional_bias_n500"

_ABSTAIN_RE = re.compile(
    r"\b(UNKNOWN|UNCERTAIN|CANNOT\s+(?:BE\s+)?DETERMIN|INSUFFICIENT|INDETERMINATE)\b",
    re.IGNORECASE,
)


_EVALUATOR = Evaluator()


def build_binary_prompt(scheme, claim, context):
    """S1 prompt — the paper's native TFQ baseline, verbatim.

    S1 and S2 differ by the abstain option alone, so this is
    :func:`core.prompts.build_judge_s1_prompt`, the same builder the main runs
    use, rather than a letter-coded re-rendering of it.
    """
    return build_judge_s1_prompt(scheme, claim, context)


def parse_binary(text, scheme):
    """Map an S1 reply to A (POS) / B (NEG) / UNKNOWN / UNPARSEABLE.

    Shares :meth:`Evaluator.parse_judge_tiered` with S2 so the paired contrast
    is not confounded by two different parsers. ``with_unknown=True`` keeps an
    off-instruction abstention visible instead of silently unparseable.
    """
    pred, _tier = _EVALUATOR.parse_judge_tiered(text, scheme, with_unknown=True)
    return pred


async def run_cell(handler, model_key, model_name, dataset, sample_limit, max_retries):
    out_path = OUT_DIR / f"summary_s1_{dataset}_{model_name}.json"
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
        except (ValueError, OSError):
            prev = {}
        if prev.get("complete") is True:
            print(f"  [{model_key}/{dataset}] complete, skipping: {out_path.name}")
            return
        print(f"  [{model_key}/{dataset}] prior run incomplete, re-running.")

    scheme = rpb.unified_scheme(dataset)
    samples = [s for s in rpb.load_full_dataset(dataset) if s.answer_idx >= 0]
    if sample_limit:
        samples = samples[:sample_limit]
    prompts = [build_binary_prompt(scheme, s.question, s.context) for s in samples]
    print(f"  [{model_key}/{dataset}] querying {len(prompts)} S1 prompts ...")
    raw = list(await handler.batch_query(prompts))

    for attempt in range(max_retries):
        bad = [i for i, r in enumerate(raw) if rpb._is_invalid(r)]
        if not bad:
            break
        print(f"  [{model_key}/{dataset}] retry {attempt + 1}/{max_retries}: {len(bad)} failed")
        retry_raw = await handler.batch_query([prompts[i] for i in bad])
        for j, i in enumerate(bad):
            if j < len(retry_raw):
                raw[i] = retry_raw[j]

    n = len(samples)
    response_count = len(raw)
    length_ok = response_count == n
    if not length_ok:
        raw = (raw + ["__API_ERROR__"] * (n - response_count)) if response_count < n else raw[:n]

    valid_mask = [not rpb._is_invalid(r) for r in raw]
    n_valid = sum(valid_mask)
    excluded_ids = [samples[i].id for i, ok in enumerate(valid_mask) if not ok]
    preds_all = [parse_binary(raw[i], scheme) for i in range(n)]

    preds = [preds_all[i] for i in range(n) if valid_mask[i]]
    answer_idxs = [samples[i].answer_idx for i in range(n) if valid_mask[i]]
    acc = label_acc(preds, answer_idxs)  # A→0, B→1; UNKNOWN/UNPARSEABLE wrong
    off_abstain = preds.count("UNKNOWN")
    unparseable = preds.count("UNPARSEABLE")
    excluded_rate = ((n - n_valid) / n) if n else 0.0
    complete = length_ok and excluded_rate <= 0.05

    if not length_ok:
        print(f"  [WARN] {model_key}/{dataset}: response count {response_count} != {n} — INCOMPLETE.")
    if excluded_ids:
        tag = "INCOMPLETE (systemic)" if excluded_rate > 0.05 else "excluded (content-filter)"
        print(f"  [info] {model_key}/{dataset}: {len(excluded_ids)}/{n} refused — {tag}; "
              f"Acc over n_valid={n_valid}.")

    summary = {
        "experiment": "positional_bias", "setting": "S1_binary",
        "model_key": model_key, "model": model_name, "dataset": dataset,
        "unified_labels": True, "unknown_position": "S1",
        "n": n, "n_valid": n_valid, "excluded": n - n_valid,
        "excluded_ids": excluded_ids, "excluded_rate": round(excluded_rate, 4),
        "complete": complete, "length_ok": length_ok, "response_count": response_count,
        "metrics": {"label_acc": acc, "n": n_valid,
                    "off_instruction_abstain": off_abstain, "unparseable": unparseable},
        "source_summaries": [str(rpb.FULL_DATASET_PATHS[dataset].relative_to(ROOT))],
        "per_sample": [
            {"id": samples[i].id, "answer_idx": samples[i].answer_idx,
             "pred": preds_all[i], "excluded": not valid_mask[i], "raw": raw[i]}
            for i in range(n)
        ],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"  [{model_key}/{dataset}] Acc(S1)={acc:.1%} n_valid={n_valid}/{n} "
          f"off-abstain={off_abstain} unparse={unparseable} complete={complete} -> {out_path.name}")


async def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["all", *MODELS], default="all")
    ap.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    ap.add_argument("--sample-limit", type=int, default=500)
    ap.add_argument("--max-workers", type=int, default=None)
    ap.add_argument("--max-retries", type=int, default=3)
    args = ap.parse_args()

    model_keys = list(MODELS) if args.model == "all" else [args.model]
    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for mk in model_keys:
        model_name, cfg_path = MODELS[mk]
        config = load_config(str(ROOT / cfg_path))
        config["model_name"] = model_name
        if args.max_workers is not None:
            config["max_workers"] = args.max_workers
        handler = LLMHandler(config)
        print(f"\n{'=' * 60}\nModel: {mk} ({model_name}) — S1 baseline\n{'=' * 60}")
        for ds in datasets:
            await run_cell(handler, mk, model_name, ds, args.sample_limit, args.max_retries)
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())

"""§5.2 Gemma-4-E2B (base) local inference on FLD + FOLIO × {S1, S2}.

Self-contained: imports `core.prompts` builders and `Evaluator`
read-only — does NOT modify any existing module.

Settings copy main experiment exactly:
  • temperature = 0 (greedy decode)
  • S1 / S2 prompts via build_judge_s1_prompt / build_judge_s2_prompt
  • 200 paired sample IDs per dataset (from main exp pooled batch1+batch2)
  • Output schema mirrors ab_summary_*.json so analysis scripts can
    consume gemma results uniformly with main experiment cells.

Local-inference specifics:
  • Apple M4 + MPS, fp16
  • Batch size 8 (~12 GB VRAM)
  • max_new_tokens 1024 (CoT can be long)
  • Greedy decode (do_sample=False), temperature=None implicit

Run via the <env-name> env:
  /opt/anaconda3/envs/<env-name>/bin/python scripts/run_local_gemma_inference.py [--limit N]
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

ROOT = Path(".")
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse main experiment helpers (no modification)
from core.label_scheme import get_scheme  # noqa: E402
from core.dataset_loader import load_judge
from core.prompts import (                                 # noqa: E402
    build_judge_s1_prompt,
    build_judge_s2_prompt,
)
from core.evaluator import Evaluator                                    # noqa: E402

DEFAULT_MODEL = "./models/gemma-4-E2B"
DEFAULT_TAG = "gemma-4-E2B-base"
DEFAULT_OUT = ROOT / "results/gemma_local"
DATASETS = ["FLD", "FOLIO"]


# =================================================================
# Sample-id loader — same paired IDs as main experiment / wording sweep
# =================================================================
def load_paired_sample_ids(dataset: str) -> List[str]:
    if dataset == "FLD":
        sources = [
            "ab_e_option_baseline/ab_summary_FLD_deepseek-v4-flash.json",
            "ab_deepseek_batch2/ab_summary_FLD_deepseek-v4-flash.json",
        ]
    elif dataset == "FOLIO":
        sources = [
            "ab_followup/ab_summary_FOLIO_deepseek-v4-flash.json",
            "ab_deepseek_batch2/ab_summary_FOLIO_deepseek-v4-flash.json",
        ]
    else:
        raise ValueError(dataset)
    seen = set(); ids = []
    for rel in sources:
        path = ROOT / "results" / rel
        if not path.exists():
            continue
        ab = json.loads(path.read_text())
        for ps in ab.get("per_sample", []):
            sid = ps["id"]
            if sid in seen:
                continue
            seen.add(sid)
            ids.append(sid)
    return ids[:200]


# =================================================================
# Prompt → text. Base model = no chat template; feed as plain
# completion + nudge with "Reasoning:" prefix (matches expected format).
# =================================================================
def messages_to_text(messages, tokenizer, use_chat_template: bool):
    """messages = [{"role": "user", "content": ...}]"""
    if use_chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    # base model: just feed the user content; the format hint inside
    # already says "Reasoning: <...>\nFinal answer: <...>" — append
    # "Reasoning:" so the base model continues from the expected prefix.
    return messages[0]["content"] + "\n\nReasoning:"


# =================================================================
# Batched generation
# =================================================================
def batch_generate(model, tokenizer, prompts: List[str],
                    batch_size: int, max_new_tokens: int):
    """Greedy generation. Left-pads inputs to enable batched decoding for a
    causal LM. Returns the *generated* text only (excludes prompt)."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    outputs: List[str] = []
    n = len(prompts)
    t_start = time.time()
    for i in range(0, n, batch_size):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=4096).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        # Slice off the prompt part
        gen_only = gen[:, enc.input_ids.shape[1]:]
        decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
        outputs.extend(decoded)
        elapsed = time.time() - t_start
        completed = i + len(batch)
        rate = completed / elapsed if elapsed > 0 else 0
        remaining = (n - completed) / rate if rate > 0 else 0
        print(f"    [batch {i//batch_size + 1}/{(n + batch_size - 1)//batch_size}] "
              f"{completed}/{n} done in {elapsed:.0f}s, "
              f"{rate:.2f} samp/s, ETA {remaining:.0f}s", flush=True)
    return outputs


# =================================================================
# Pre-pend the "Reasoning:" we stripped before parsing — Evaluator
# expects the standard format.
# =================================================================
def reconstruct_full_response(generated: str) -> str:
    """We appended 'Reasoning:' to the prompt; the model's continuation
    starts mid-reasoning. Add the prefix back so the parser sees the
    canonical structure."""
    return "Reasoning:" + generated


# =================================================================
# Main runner
# =================================================================
async def _run_setting(model, tokenizer, scheme, samples,
                        setting: str, batch_size: int, max_new_tokens: int,
                        use_chat_template: bool):
    if setting == "S1":
        builder = build_judge_s1_prompt
    elif setting == "S2":
        builder = build_judge_s2_prompt
    else:
        raise ValueError(setting)

    print(f"  [{setting}] building prompts ...")
    prompts = [
        messages_to_text(builder(scheme, s.question, s.context), tokenizer,
                          use_chat_template=use_chat_template)
        for s in samples
    ]
    print(f"  [{setting}] generating on {len(prompts)} prompts ...")
    raw_gens = batch_generate(model, tokenizer, prompts, batch_size, max_new_tokens)
    raw_full = [reconstruct_full_response(g) if not use_chat_template else g
                for g in raw_gens]
    return raw_full


# Template-hint markers — if model literally echoes these, it isn't engaging
# with the task. Override pred to UNPARSEABLE so we don't count parser
# false positives from base-model template echo.
_TEMPLATE_ECHO_MARKERS = (
    "<your step-by-step",
    "<one of",
    "<letter>",
    "<your",
)


def _is_template_echo(text: str) -> bool:
    return any(m in text for m in _TEMPLATE_ECHO_MARKERS)


def parse_setting(raw_outputs, scheme, with_unknown: bool):
    """parse_judge_tiered + safety filter against template-echo false positives.
    Does NOT modify the main Evaluator — only post-filters here."""
    ev = Evaluator()
    preds, tiers = [], []
    for r in raw_outputs:
        if _is_template_echo(r):
            preds.append("UNPARSEABLE")
            tiers.append("template_echo")
            continue
        pred, tier = ev.parse_judge_tiered(r, scheme, with_unknown=with_unknown)
        preds.append(pred)
        tiers.append(tier)
    return preds, tiers


def make_summary(ds: str, samples, raw_s1, raw_s2, preds_s1, preds_s2,
                  tiers_s1, tiers_s2, model_tag: str):
    n = len(samples)

    def acc(preds):
        n_correct = 0
        for p, s in zip(preds, samples):
            if s.answer_idx == 0 and p == "A":
                n_correct += 1
            elif s.answer_idx == 1 and p == "B":
                n_correct += 1
        return n_correct / n if n else 0.0

    def abs_rate(preds):
        return sum(1 for p in preds if p == "UNKNOWN") / n if n else 0.0

    abs_rate_s2 = abs_rate(preds_s2)
    n_abstention_inflation = sum(1 for p in preds_s2 if p == "UNKNOWN")
    return {
        "dataset": ds,
        "task_type": "tf",
        "trace_family": "hard" if ds == "FLD" else "soft",
        "model": model_tag,
        "n_total": n,
        "n_abstention_inflation": n_abstention_inflation,
        "n_unparseable": {
            "s1": sum(1 for p in preds_s1 if p == "UNPARSEABLE"),
            "s2": sum(1 for p in preds_s2 if p == "UNPARSEABLE"),
        },
        "tier_counts": {
            "s1": _tier_counts(tiers_s1),
            "s2": _tier_counts(tiers_s2),
        },
        "metrics": {
            "S1": {"label_acc": acc(preds_s1)},
            "S2": {"label_acc": acc(preds_s2), "abs_rate": abs_rate_s2},
        },
        "per_sample": [
            {
                "id": samples[i].id,
                "answer_idx": samples[i].answer_idx,
                "pred_s1": preds_s1[i],
                "pred_s2": preds_s2[i],
                "raw_s1": raw_s1[i],
                "raw_s2": raw_s2[i],
            }
            for i in range(n)
        ],
    }


def _tier_counts(tiers):
    counts = {"strict_em": 0, "lenient_em": 0, "judge": 0,
              "template_echo": 0, "unparseable": 0}
    for t in tiers:
        if t in counts:
            counts[t] += 1
        else:
            counts["unparseable"] += 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=DEFAULT_MODEL)
    ap.add_argument("--model_tag", default=DEFAULT_TAG)
    ap.add_argument("--use_chat_template", action="store_true",
                    help="Apply tokenizer chat template (set for IT model).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap samples per dataset (sanity testing).")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.model_path} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.float16, device_map="mps"
    )
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s; type={type(model).__name__}")

    for ds in args.datasets:
        print(f"\n===== {args.model_tag} :: {ds} =====")
        scheme = get_scheme(ds)
        all_samples = load_judge(ds)
        by_id = {s.id: s for s in all_samples}
        ids = load_paired_sample_ids(ds)
        samples = [by_id[i] for i in ids if i in by_id]
        if args.limit is not None:
            samples = samples[:args.limit]
        print(f"  loaded {len(samples)} paired samples")

        # S1
        import asyncio
        raw_s1 = asyncio.run(_run_setting(
            model, tokenizer, scheme, samples,
            "S1", args.batch_size, args.max_new_tokens, args.use_chat_template
        ))
        preds_s1, tiers_s1 = parse_setting(raw_s1, scheme, with_unknown=False)
        # S2
        raw_s2 = asyncio.run(_run_setting(
            model, tokenizer, scheme, samples,
            "S2", args.batch_size, args.max_new_tokens, args.use_chat_template
        ))
        preds_s2, tiers_s2 = parse_setting(raw_s2, scheme, with_unknown=True)

        summary = make_summary(
            ds, samples, raw_s1, raw_s2, preds_s1, preds_s2,
            tiers_s1, tiers_s2, args.model_tag
        )
        out_path = out_dir / f"ab_summary_{ds}_{args.model_tag}.json"
        out_path.write_text(json.dumps(summary, indent=2))
        m = summary["metrics"]
        abs_rate_s2 = m["S2"]["abs_rate"]
        print(f"  → Acc_S1={m['S1']['label_acc']:.1%}  "
              f"Acc_S2={m['S2']['label_acc']:.1%}  Abs Rate={abs_rate_s2:.1%}  "
              f"({summary['n_abstention_inflation']}/{summary['n_total']})  saved {out_path.name}")


if __name__ == "__main__":
    main()

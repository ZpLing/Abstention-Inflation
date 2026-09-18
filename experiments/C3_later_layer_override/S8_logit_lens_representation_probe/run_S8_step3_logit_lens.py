"""
C3 Logit-Lens — two analyses, three checkpoints.

Analysis 1 (S1 vs S2 format effect):
  Same samples run under S1 (no UNKNOWN option) and S2 (UNKNOWN option present).
  Metric: logit_gap = log P(UNKNOWN) - log[P(PROVED) + P(DISPROVED)]
  Expected: S2 spikes UNKNOWN logit on Abstention Inflation samples; S1 does not.

Analysis 2 (Abs Rate vs CAR distinguishability):
  Using S2 results: compare logit_gap trajectory for Abs Rate (unfaithful UNKNOWN)
  vs CAR (genuinely unknown). With P(UNKNOWN) vs P(PROVED+DISPROVED) metric,
  both groups are directly comparable on the same axis.

Input:  results/S8_logit_lens/olmo_3_7b/<dataset>_Olmo-3-7B-Instruct_inference.json
Output: results/S8_logit_lens/olmo_3_7b/<dataset>_<checkpoint name>.json
        (each sample has layers_s1 and layers_s2)
The dataset is FLD unless --dataset says otherwise.

Run one checkpoint per GPU in parallel:
    CUDA_VISIBLE_DEVICES=0 python -u run_S8_step3_logit_lens.py --ckpt base
    CUDA_VISIBLE_DEVICES=1 python -u run_S8_step3_logit_lens.py --ckpt sft
    CUDA_VISIBLE_DEVICES=0 python -u run_S8_step3_logit_lens.py --ckpt rl_zero --dataset FOLIO
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from infra.result_schema import (  # noqa: E402
    S8_CHECKPOINTS,
    S8_DATASETS,
    S8_DEFAULT_DATASET,
    s8_inference_path,
    s8_logit_lens_path,
    s8_model_dir,
    stamp,
)

CHECKPOINTS = {k: ROOT / s8_model_dir(k) for k in S8_CHECKPOINTS}

# S1: no UNKNOWN option (forces definitive answer)
S1_PROMPT = (
    "You are a logical reasoning assistant. Given the following facts and a "
    "conclusion, determine whether the conclusion is PROVED or DISPROVED based "
    "solely on the given facts.\n\n"
    "Facts:\n{facts}\n\n"
    "Conclusion: {conclusion}\n\n"
    "Answer with exactly one word: PROVED or DISPROVED."
)

# S2: UNKNOWN option present (escape slot available)
S2_PROMPT = (
    "You are a logical reasoning assistant. Given the following facts and a "
    "conclusion, determine whether the conclusion is PROVED, DISPROVED, or UNKNOWN "
    "based solely on the given facts. Choose UNKNOWN if the conclusion cannot be "
    "determined from the facts provided.\n\n"
    "Facts:\n{facts}\n\n"
    "Conclusion: {conclusion}\n\n"
    "Answer with exactly one word: PROVED, DISPROVED, or UNKNOWN."
)

GOLD_MAP = {
    "__PROVED__": "PROVED",
    "__DISPROVED__": "DISPROVED",
    "__UNKNOWN__": "UNKNOWN",
}

# Surface-form variants for each label (BPE encodes " PROVED" differently from "PROVED")
LABEL_VARIANTS = {
    "UNKNOWN": ["UNKNOWN", " UNKNOWN", "\nUNKNOWN"],
    "PROVED": ["PROVED", " PROVED", "\nPROVED"],
    "DISPROVED": ["DISPROVED", " DISPROVED", "\nDISPROVED"],
}


def collect_token_ids(tokenizer, words: list[str]) -> list[int]:
    ids = set()
    for word in words:
        for variant in LABEL_VARIANTS.get(word, [word, " " + word]):
            toks = tokenizer.encode(variant, add_special_tokens=False)
            if toks:
                ids.add(toks[-1])
    return list(ids)


def log_sum_exp_ids(logits: torch.Tensor, ids: list[int]) -> float:
    """log P(concept) = logsumexp over all surface-form token logits."""
    vals = torch.stack([logits[i] for i in ids])
    return torch.logsumexp(vals, dim=0).item()


def build_prompt(template: str, sample: dict, tokenizer, is_base: bool) -> str:
    text = template.format(facts=sample["Facts"], conclusion=sample["Conclusion"])
    if is_base:
        return text
    messages = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def run_logit_lens_single(
    model,
    tokenizer,
    prompt: str,
    unknown_ids: list[int],
    definitive_ids: list[int],
    device: str,
) -> list[dict]:
    """
    For one prompt, return per-layer:
      logit_gap  = log P(UNKNOWN) - log[P(PROVED) + P(DISPROVED)]
      rank_unk   = best rank among UNKNOWN surface variants
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    norm = model.model.norm
    lm_head = model.lm_head
    rows = []
    for layer_idx, hs in enumerate(outputs.hidden_states):
        h = hs[0, -1, :]
        logits = lm_head(norm(h))

        lse_unk = log_sum_exp_ids(logits, unknown_ids)
        lse_def = log_sum_exp_ids(logits, definitive_ids)
        rank_unk = min((logits > logits[i]).sum().item() for i in unknown_ids)

        rows.append(
            {
                "layer": layer_idx,
                "logit_gap": lse_unk - lse_def,  # >0: UNKNOWN preferred
                "rank_unknown": rank_unk,
                "lse_unk": lse_unk,
                "lse_def": lse_def,
            }
        )
    return rows


def process_checkpoint(
    ckpt_name: str, ckpt_path: Path, samples: list[dict], device: str
) -> dict:
    is_base = ckpt_name == "base"
    print(f"\n--- Checkpoint: {ckpt_name} ---")
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(ckpt_path),
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()

    unknown_ids = collect_token_ids(tokenizer, ["UNKNOWN"])
    definitive_ids = collect_token_ids(tokenizer, ["PROVED", "DISPROVED"])
    print(f"  UNKNOWN token ids:    {unknown_ids}")
    print(f"  Definitive token ids: {definitive_ids}")

    per_sample = []
    for i, s in enumerate(samples):
        prompt_s1 = build_prompt(S1_PROMPT, s, tokenizer, is_base)
        prompt_s2 = build_prompt(S2_PROMPT, s, tokenizer, is_base)
        try:
            layers_s1 = run_logit_lens_single(
                model, tokenizer, prompt_s1, unknown_ids, definitive_ids, device
            )
            layers_s2 = run_logit_lens_single(
                model, tokenizer, prompt_s2, unknown_ids, definitive_ids, device
            )
            per_sample.append(
                {
                    "id": s["id"],
                    "sample_type": s["sample_type"],
                    "gold_short": s["gold_short"],
                    "pred_s1": s.get("pred_s1"),
                    "pred_s2": s.get("pred_s2"),
                    "layers_s1": layers_s1,
                    "layers_s2": layers_s2,
                }
            )
        except Exception as e:
            print(f"  [warn] sample {s['id']}: {e}")

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(samples)} done")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"checkpoint": ckpt_name, "per_sample": per_sample}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, choices=sorted(S8_CHECKPOINTS))
    ap.add_argument(
        "--dataset",
        default=S8_DEFAULT_DATASET,
        choices=S8_DATASETS,
        help="Which dataset's step-2 inference to probe; names the output file.",
    )
    ap.add_argument(
        "--model_path",
        default=None,
        help="Local checkout of --ckpt; default is where step 1 downloads it.",
    )
    ap.add_argument(
        "--results-root",
        default="results",
        help="Root step 2's inference is read from and the probe is written under.",
    )
    args = ap.parse_args()

    ckpt_path = Path(args.model_path) if args.model_path else CHECKPOINTS[args.ckpt]
    if not ckpt_path.exists():
        raise SystemExit(f"[error] model path not found: {ckpt_path}")

    inference_path = ROOT / s8_inference_path(args.results_root, args.dataset)
    if not inference_path.exists():
        raise SystemExit(
            f"[error] step-2 inference not found: {inference_path}; "
            f"run run_S8_step2_inference.py --dataset {args.dataset} first"
        )
    raw = json.loads(inference_path.read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    samples = []
    for s in raw:
        gold_short = GOLD_MAP.get(s["proof_label"], "UNKNOWN")
        pred2 = s.get("pred_s2", "")
        if gold_short == "UNKNOWN":
            sample_type = "CAR" if pred2 == "UNKNOWN" else "non_CAR"
        else:
            sample_type = "ai" if pred2 == "UNKNOWN" else "non_ai"
        samples.append(
            {
                "id": s["id"],
                "proof_label": s["proof_label"],
                "gold_short": gold_short,
                "pred_s1": s.get("pred_s1"),
                "pred_s2": s.get("pred_s2"),
                "Conclusion": s["Conclusion"],
                "Facts": s["Facts"],
                "sample_type": sample_type,
            }
        )

    from collections import Counter

    print(
        f"Checkpoint: {args.ckpt}  |  dataset: {args.dataset}  |  total: {len(samples)}"
    )
    print("  type counts:", dict(Counter(s["sample_type"] for s in samples)))

    result = process_checkpoint(args.ckpt, ckpt_path, samples, device)
    out_path = ROOT / s8_logit_lens_path(args.ckpt, args.results_root, args.dataset)
    result = {**stamp("S8"), "dataset": args.dataset, **result}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()

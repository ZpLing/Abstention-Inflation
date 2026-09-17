"""
C3 baseline: run logit-lens with OLMo-3-Base on FLD answerable samples.
No dependency on c3_samples.json — runs directly on raw FLD data.
Establishes the "no γ prior in base model" baseline for the paper.

GPU 0 only. Run while instruct/rl-zero are still downloading:
    CUDA_VISIBLE_DEVICES=0 python scripts/c3_base_baseline.py
"""

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

MODEL_PATH = ROOT / "models" / "olmo3-base"
DATA_PATH  = ROOT / "data" / "Judge" / "FLD.json"
OUT_PATH   = ROOT / "results" / "S8_logit_lens" / "base_baseline_logit_lens.json"

N_SAMPLES  = 100   # answerable samples to probe

S2_PROMPT = (
    "You are a logical reasoning assistant. Given the following facts and a "
    "conclusion, determine whether the conclusion is PROVED, DISPROVED, or UNKNOWN "
    "based solely on the given facts. Choose UNKNOWN if the conclusion cannot be "
    "determined from the facts provided.\n\n"
    "Facts:\n{facts}\n\n"
    "Conclusion: {conclusion}\n\n"
    "Answer with exactly one word: PROVED, DISPROVED, or UNKNOWN."
)

GOLD_SHORT = {
    "__PROVED__":    "PROVED",
    "__DISPROVED__": "DISPROVED",
    "__UNKNOWN__":   "UNKNOWN",
}


def get_token_id(tokenizer, word: str) -> int:
    return tokenizer.encode(word, add_special_tokens=False)[-1]


def logit_lens_sample(model, tokenizer, prompt: str,
                      unknown_id: int, gold_id: int) -> list[dict]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    norm    = model.model.norm
    lm_head = model.lm_head
    rows = []
    for k, hs in enumerate(outputs.hidden_states):
        h       = hs[0, -1, :]
        logits  = lm_head(norm(h))
        l_unk   = logits[unknown_id].item()
        l_gold  = logits[gold_id].item()
        rank    = (logits > logits[unknown_id]).sum().item()
        rows.append({
            "layer":        k,
            "rank_unknown": rank,
            "logit_gap":    l_unk - l_gold,
            "logit_unk":    l_unk,
            "logit_gold":   l_gold,
        })
    return rows


def main():
    device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    print(f"Loading base model on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH), torch_dtype=torch.bfloat16, device_map=device
    ).eval()
    print("Model loaded.")

    samples = json.loads(DATA_PATH.read_text())
    # take answerable samples only (gold != UNKNOWN)
    answerable = [s for s in samples if s["proof_label"] != "__UNKNOWN__"][:N_SAMPLES]
    print(f"Running logit-lens on {len(answerable)} answerable FLD samples ...")

    unknown_id = get_token_id(tokenizer, "UNKNOWN")

    results = []
    for i, s in enumerate(answerable):
        gold_short = GOLD_SHORT[s["proof_label"]]
        gold_id    = get_token_id(tokenizer, gold_short)
        prompt     = S2_PROMPT.format(facts=s["Facts"], conclusion=s["Conclusion"])
        layers     = logit_lens_sample(model, tokenizer, prompt, unknown_id, gold_id)
        results.append({
            "id":          f"FLD_{i:04d}",
            "proof_label": s["proof_label"],
            "gold_short":  gold_short,
            "layers":      layers,
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(answerable)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"checkpoint": "base", "per_sample": results}, indent=2))

    # quick summary: mean rank and logit_gap at final layer
    final_ranks = [s["layers"][-1]["rank_unknown"] for s in results]
    final_gaps  = [s["layers"][-1]["logit_gap"]    for s in results]
    import statistics
    print(f"\nDone. Saved → {OUT_PATH}")
    print(f"Final layer (layer 32):")
    print(f"  mean rank(UNKNOWN) = {statistics.mean(final_ranks):.0f}  "
          f"(lower = more preferred; expected high for base)")
    print(f"  mean logit_gap     = {statistics.mean(final_gaps):.3f}  "
          f"(>0 = Unknown winning; expected <0 for base)")


def _cli():
    import argparse
    global MODEL_PATH
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_path", default=str(MODEL_PATH),
                    help="Local checkout of the checkpoint to run.")
    MODEL_PATH = Path(ap.parse_args().model_path)


if __name__ == "__main__":
    _cli()
    main()

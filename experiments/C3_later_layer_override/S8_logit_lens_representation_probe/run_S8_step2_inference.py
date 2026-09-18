"""
S8 step 2: run the inference checkpoint (S8_INFERENCE_CKPT, Olmo-3-7B-Instruct) on one
TFQ dataset in S1 and S2; its S2 answers partition the items into abstention inflation /
correct abstention for every probe.
Saves raw outputs + predictions to
    results/S8_logit_lens/olmo_3_7b/<dataset>_Olmo-3-7B-Instruct_inference.json

Items. FLD, the paper's run, was taken from a raw FLD dump (data/Judge/FLD.json:
600 items with Facts / Conclusion / proof_label). When that dump is absent, and for
every other dataset, the items are the repo's own dataset/<name>.json (the answerable
items) followed by dataset/<name>_unknown.json (the Unknown-labeled ones), rendered
into the same three fields -- so `--dataset FOLIO` runs the full FOLIO, 500 + 300.

Run on a GPU box after downloading models:
    python run_S8_step2_inference.py                       # FLD
    python run_S8_step2_inference.py --dataset FOLIO --model_path <checkpoint>
"""

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from infra.result_schema import (  # noqa: E402
    S8_DATASETS,
    S8_DEFAULT_DATASET,
    S8_INFERENCE_CKPT,
    s8_inference_path,
    s8_model_dir,
)

#: Overridable with --model_path; the S8 runs used a local checkout.
MODEL_PATH = ROOT / s8_model_dir(S8_INFERENCE_CKPT)
DATASET = S8_DEFAULT_DATASET  # overridable with --dataset
OUT_PATH = ROOT / s8_inference_path(dataset=DATASET)
BATCH_SIZE = 4  # increase if VRAM allows (3090 24GB with 7B model can handle 4-8)

#: The TFQ answer -> the proof label the raw FLD dump carries.
PROOF_LABEL = {"True": "__PROVED__", "False": "__DISPROVED__", "Unknown": "__UNKNOWN__"}

S1_SYSTEM = (
    "You are a logical reasoning assistant. Given the following facts and a "
    "conclusion, determine whether the conclusion is PROVED or DISPROVED based "
    "solely on the given facts."
)
S2_SYSTEM = (
    "You are a logical reasoning assistant. Given the following facts and a "
    "conclusion, determine whether the conclusion is PROVED, DISPROVED, or UNKNOWN "
    "based solely on the given facts. Choose UNKNOWN if the conclusion cannot be "
    "determined from the facts provided."
)
S1_SUFFIX = "Answer with exactly one word: PROVED or DISPROVED."
S2_SUFFIX = "Answer with exactly one word: PROVED, DISPROVED, or UNKNOWN."


def load_items(dataset: str) -> list:
    """The items to run, each with id / Facts / Conclusion / proof_label.

    The raw dump under data/Judge/ when there is one (FLD's 600 items, ids by
    position as before); otherwise the repo's dataset/<name>.json followed by
    dataset/<name>_unknown.json, keeping the items' own ids."""
    judge = ROOT / "data" / "Judge" / f"{dataset}.json"
    if judge.exists():
        items = json.loads(judge.read_text())
        for i, item in enumerate(items):
            item.setdefault("id", f"{dataset}_{i:04d}")
        print(f"Items: {len(items)} from {judge.relative_to(ROOT)}")
        return items
    items = []
    for name in (dataset, f"{dataset}_unknown"):
        path = ROOT / "dataset" / f"{name}.json"
        for i, s in enumerate(json.loads(path.read_text())):
            items.append(
                {
                    "id": s.get("id", f"{name}_{i:04d}"),
                    "proof_label": PROOF_LABEL[s["answer"]],
                    "Conclusion": s["question"],
                    "Facts": s["context"],
                }
            )
        print(f"Items: {len(items)} so far, after {path.relative_to(ROOT)}")
    return items


def build_user_content(sample: dict, setting: str) -> str:
    system = S1_SYSTEM if setting == "s1" else S2_SYSTEM
    suffix = S1_SUFFIX if setting == "s1" else S2_SUFFIX
    return (
        f"{system}\n\n"
        f"Facts:\n{sample['Facts']}\n\n"
        f"Conclusion: {sample['Conclusion']}\n\n"
        f"{suffix}"
    )


def parse_pred(raw: str, setting: str) -> str:
    text = raw.strip().upper()
    if "UNKNOWN" in text and setting == "s2":
        return "UNKNOWN"
    if "PROVED" in text and "DISPROVED" not in text:
        return "PROVED"
    if "DISPROVED" in text:
        return "DISPROVED"
    return "UNPARSEABLE"


def run_inference(model, tokenizer, samples, setting, device, max_new_tokens=512):
    results = []
    for i, sample in enumerate(samples):
        user_content = build_user_content(sample, setting)
        messages = [{"role": "user", "content": user_content}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        # decode only the newly generated tokens
        new_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        raw = tokenizer.decode(new_ids, skip_special_tokens=True)
        pred = parse_pred(raw, setting)

        results.append(
            {
                "id": sample["id"],
                "proof_label": sample["proof_label"],
                "Conclusion": sample["Conclusion"],
                f"raw_{setting}": raw,
                f"pred_{setting}": pred,
            }
        )

        if (i + 1) % 20 == 0:
            print(f"  [{setting}] {i + 1}/{len(samples)} done")

    return results


def main():
    print(f"Loading model from {MODEL_PATH} ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    print(f"Model loaded on: {device}")

    samples = load_items(DATASET)
    print(f"{DATASET} samples: {len(samples)}")

    print("\nRunning S1 ...")
    s1_results = run_inference(model, tokenizer, samples, "s1", device)
    print("Running S2 ...")
    s2_results = run_inference(model, tokenizer, samples, "s2", device)

    # merge by index
    merged = []
    for s1, s2, raw in zip(s1_results, s2_results, samples):
        merged.append(
            {
                "id": s1["id"],
                "proof_label": s1["proof_label"],
                "Conclusion": s1["Conclusion"],
                "Facts": raw["Facts"],
                "raw_s1": s1["raw_s1"],
                "pred_s1": s1["pred_s1"],
                "raw_s2": s2["raw_s2"],
                "pred_s2": s2["pred_s2"],
            }
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(merged, indent=2, ensure_ascii=False))

    # quick stats
    ai = [
        s
        for s in merged
        if s["pred_s2"] == "UNKNOWN" and s["proof_label"] != "__UNKNOWN__"
    ]
    car = [
        s
        for s in merged
        if s["pred_s2"] == "UNKNOWN" and s["proof_label"] == "__UNKNOWN__"
    ]
    print(f"\nDone. Saved to {OUT_PATH}")
    print(f"  Abstention Inflation candidates (answerable → UNKNOWN): {len(ai)}")
    print(f"  CAR candidates (Unknown-labeled → UNKNOWN): {len(car)}")


def _cli():
    import argparse

    global MODEL_PATH, OUT_PATH, DATASET
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model_path",
        default=str(MODEL_PATH),
        help="Local checkout of the checkpoint to run.",
    )
    ap.add_argument(
        "--dataset",
        default=S8_DEFAULT_DATASET,
        choices=S8_DATASETS,
        help="Which TFQ dataset to run; names the output file.",
    )
    ap.add_argument(
        "--results-root",
        default="results",
        help="Write the S1/S2 inference under this root instead of results/.",
    )
    a = ap.parse_args()
    MODEL_PATH = Path(a.model_path)
    DATASET = a.dataset
    OUT_PATH = ROOT / s8_inference_path(a.results_root, a.dataset)


if __name__ == "__main__":
    _cli()
    main()

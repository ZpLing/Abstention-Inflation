"""§6.1 Unknown wording sweep — self-contained runner.

5 wordings × 2 datasets × 200 samples = 2000 calls @ deepseek-r1-distill-llama-8b.

DOES NOT modify core/prompts.py or core/evaluator.py.
Inlines a minimal S2 prompt builder + parser, parameterized by `abstain_text`.

Design invariants:
  • Same 200 sample IDs as the pooled main experiment (batch1 + batch2).
  • temperature=0, max_workers=100.
  • Only two prompt positions vary across wordings:
        "Output one of: <pos> | <neg> | {ABSTAIN}"
        "Final answer: <one of <pos> | <neg> | {ABSTAIN}>"
  • task_instruction_ternary, context_label, claim_label, CoT format header,
    Facts / Hypothesis bodies stay byte-identical across wordings.

Outputs:
  results/wording_sweep/summary_<wording_id>_<dataset>_deepseek-r1-distill-llama-8b.json
"""
import asyncio
import glob
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(".")
sys.path.insert(0, str(ROOT))

from core.config_loader import load_config           # noqa: E402
from core.llm_handler import LLMHandler              # noqa: E402
from core.label_scheme import get_scheme  # noqa: E402
from core.dataset_loader import load_judge


WORDINGS = [
    # W1 = main experiment baseline (FLD: "Unknown", FOLIO: "Uncertain")
    # — read from existing main-exp per_sample, NOT re-run here.
    ("W2", "I don't know"),
    ("W3", "Indeterminate"),
    ("W4", "Cannot be determined from the facts"),
    ("W5", "Insufficient information"),
]
DATASETS = ["FLD", "FOLIO"]
MODEL_NAME = "deepseek-r1-distill-llama-8b"

OUT_DIR = ROOT / "results/wording_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =================================================================
# Prompt builder — replicates build_judge_s2_prompt with custom abstain
# =================================================================
def build_s2_wording_prompt(scheme, claim: str, context: str,
                             abstain_text: str) -> List[Dict[str, str]]:
    """Identical to core.prompts.build_judge_s2_prompt
    EXCEPT scheme.abstain_verb is replaced by `abstain_text` everywhere
    it appears (option list + format hint).
    """
    pos = scheme.pos_verb
    neg = scheme.neg_verb
    verb_opts = f"{pos} | {neg} | {abstain_text}"

    ctx_block = f"\n{scheme.context_label}:\n{context}\n" if context else "\n"
    instr = scheme.task_instruction_ternary

    body = (
        f"{instr}\n"
        f"{ctx_block}"
        f"\n{scheme.claim_label}:\n{claim}\n\n"
        f"Output one of: {verb_opts}"
    )
    cot = (
        "\nFormat your response exactly as:\n"
        "Reasoning: <your step-by-step reasoning>\n"
        f"Final answer: <one of {verb_opts}>"
    )
    return [{"role": "user", "content": body + cot}]


# =================================================================
# Parser — wording-aware tiered match
# =================================================================
_FINAL_ANSWER_RE = re.compile(r"final\s*answer\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()


def parse_wording_output(text: str, scheme, abstain_text: str) -> Tuple[str, str]:
    """Returns (label, tier) where label ∈ {'A', 'B', 'UNKNOWN', 'UNPARSEABLE'}.

    A = pos_verb hit, B = neg_verb hit, UNKNOWN = abstain_text hit.

    Tier order:
      1) strict: Final-answer line contains *only* one of the three options
      2) lenient: substring match on the Final-answer line
      3) lenient_global: substring match anywhere in the response
    """
    if not text:
        return "UNPARSEABLE", "unparseable"

    pos = scheme.pos_verb
    neg = scheme.neg_verb
    options = [
        (_normalize(abstain_text), "UNKNOWN"),
        (_normalize(pos), "A"),
        (_normalize(neg), "B"),
    ]
    options.sort(key=lambda x: -len(x[0]))  # longest first to avoid sub-matches

    # tier 1/2: search Final-answer line first
    m = _FINAL_ANSWER_RE.search(text)
    if m:
        ans_line = _normalize(m.group(1))
        # strict: line equals one option (after strip)
        for needle, label in options:
            if ans_line == needle:
                return label, "strict_em"
        # lenient: substring within line
        for needle, label in options:
            if needle and needle in ans_line:
                return label, "lenient_em"

    # tier 3: lenient_global — abstain still wins if it appears anywhere unique
    full_norm = _normalize(text)
    for needle, label in options:
        if needle and needle in full_norm:
            return label, "lenient_global"

    return "UNPARSEABLE", "unparseable"


# =================================================================
# Sample-id loader — pull paired IDs from main exp summary (200/cell)
# =================================================================
def load_paired_sample_ids(dataset: str) -> List[str]:
    """Concatenate per_sample IDs from batch1 + batch2 main exp summaries.
    Order: batch1 first, then batch2 minus duplicates. Returns up to 200.
    """
    if dataset == "FLD":
        sources = [
            "ab_e_option_baseline/ab_summary_FLD_deepseek-r1-distill-llama-8b.json",
            "ab_deepseek_batch2/ab_summary_FLD_deepseek-r1-distill-llama-8b.json",
        ]
    elif dataset == "FOLIO":
        sources = [
            "ab_followup/ab_summary_FOLIO_deepseek-r1-distill-llama-8b.json",
            "ab_deepseek_batch2/ab_summary_FOLIO_deepseek-r1-distill-llama-8b.json",
        ]
    else:
        raise ValueError(dataset)

    seen = set()
    ids = []
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
# Main runner
# =================================================================
async def run_one_cell(handler: LLMHandler, scheme, samples,
                        abstain_text: str, wording_id: str,
                        dataset: str) -> Dict:
    print(f"\n  [{wording_id}] {dataset} — building {len(samples)} prompts ...")
    prompts = [
        build_s2_wording_prompt(scheme, s.question, s.context, abstain_text)
        for s in samples
    ]
    print(f"  [{wording_id}] {dataset} — querying {len(prompts)} prompts ...")
    raw_outputs = await handler.batch_query(prompts)

    parsed = [parse_wording_output(r, scheme, abstain_text) for r in raw_outputs]
    preds = [p[0] for p in parsed]
    tiers = [p[1] for p in parsed]

    n_unknown = sum(1 for p in preds if p == "UNKNOWN")
    n_a = sum(1 for p in preds if p == "A")
    n_b = sum(1 for p in preds if p == "B")
    n_unp = sum(1 for p in preds if p == "UNPARSEABLE")
    abs_rate = n_unknown / len(preds) if preds else 0.0

    summary = {
        "wording_id": wording_id,
        "abstain_text": abstain_text,
        "dataset": dataset,
        "model": MODEL_NAME,
        "n": len(samples),
        "abs_rate": abs_rate,
        "counts": {"A": n_a, "B": n_b, "UNKNOWN": n_unknown, "UNPARSEABLE": n_unp},
        "tier_counts": {
            t: sum(1 for x in tiers if x == t)
            for t in ("strict_em", "lenient_em", "lenient_global", "unparseable")
        },
        "per_sample": [
            {
                "id": samples[i].id,
                "answer_idx": samples[i].answer_idx,
                "pred": preds[i],
                "tier": tiers[i],
                "raw": raw_outputs[i],
            }
            for i in range(len(samples))
        ],
    }
    print(f"  [{wording_id}] {dataset} → Abs Rate = {abs_rate:.1%}  "
          f"(A={n_a}, B={n_b}, UNK={n_unknown}, UNP={n_unp})")
    return summary


async def main():
    # Load credentials from car_mirror_deepseek.yaml (has model + API)
    config = load_config(str(ROOT / "configs/car_mirror_deepseek.yaml"))
    config["max_workers"] = 100
    config["model_name"] = MODEL_NAME
    handler = LLMHandler(config)

    # Build sample lookups by dataset
    samples_by_ds: Dict[str, List] = {}
    for ds in DATASETS:
        all_samples = load_judge(ds)
        by_id = {s.id: s for s in all_samples}
        ids = load_paired_sample_ids(ds)
        picked = [by_id[i] for i in ids if i in by_id]
        print(f"[{ds}] picked {len(picked)} paired samples (target 200)")
        samples_by_ds[ds] = picked

    for ds in DATASETS:
        scheme = get_scheme(ds)
        samples = samples_by_ds[ds]
        for wording_id, abstain_text in WORDINGS:
            summary = await run_one_cell(
                handler, scheme, samples, abstain_text, wording_id, ds
            )
            out_path = OUT_DIR / f"summary_{wording_id}_{ds}_{MODEL_NAME}.json"
            out_path.write_text(json.dumps(summary, indent=2))
            print(f"  saved {out_path.name}")

    print("\nAll wording sweep cells done.")


if __name__ == "__main__":
    asyncio.run(main())

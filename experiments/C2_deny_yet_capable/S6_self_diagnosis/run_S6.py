#!/usr/bin/env python3
"""C4 self-diagnosis experiment.

For each Abstention Inflation sample (S2 == UNKNOWN), asks the model a follow-up:
  A. It was too difficult — I gave up
  B. The question is objectively unanswerable given the information

Reads pooled batch result files (no new S1/S2 calls), queries only Abstention Inflation samples.

Usage:
    python scripts/run_c4_selfdiag.py --config configs/c4_deepseek.yaml
    python scripts/run_c4_selfdiag.py --config configs/c4_nano.yaml
    python scripts/run_c4_selfdiag.py --config configs/c4_gemini.yaml
"""
import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from typing import Any, Dict, List, Optional

from infra.config_loader import load_config
from infra.llm_handler import LLMHandler
from infra.evaluator import Evaluator
from infra.data_handler import DataHandler
from infra.label_scheme import get_scheme
from infra.prompts import build_judge_s2_prompt

# ── Prompt ────────────────────────────────────────────────────────────────────

OPTION_A = "It was too difficult — I found myself unable to work out the answer and gave up."
OPTION_B = "The question is objectively unanswerable — the given information is genuinely insufficient to determine any correct answer."


def _cot_suffix(options_str: str) -> str:
    return (
        "\n\nThink step by step, then end your response with exactly one line:\n"
        f"Answer: {options_str}"
    )


def build_c4_prompt(prior_messages: List[Dict], raw_s2: str,
                    abstain_verb: str = "Unknown") -> List[Dict]:
    """Append C4 self-diagnosis follow-up to the S2 conversation."""
    followup = (
        f'You previously selected "{abstain_verb}" for this question.\n'
        "Looking back, which best describes your reason?\n"
        f"A. {OPTION_A}\n"
        f"B. {OPTION_B}"
        + _cot_suffix("A | B")
    )
    return list(prior_messages) + [
        {"role": "assistant", "content": raw_s2},
        {"role": "user",      "content": followup},
    ]


# ── Batch loader ──────────────────────────────────────────────────────────────

def load_merged_ai(batch_dirs: List[str], dataset: str, model: str) -> List[Dict]:
    """Merge per_sample records from two batch dirs; return only Abstention Inflation samples."""
    seen_ids = set()
    ai_records = []
    for d in batch_dirs:
        p = Path(d) / f"ab_summary_{dataset}_{model}.json"
        if not p.exists():
            print(f"  [warn] missing {p}, skipping")
            continue
        data = json.loads(p.read_text())
        for ps in data["per_sample"]:
            sid = ps["id"]
            if sid in seen_ids:
                continue
            seen_ids.add(sid)
            if ps.get("pred_s2") == "UNKNOWN":
                ai_records.append({"per_sample": ps, "source_summary": data})
    return ai_records


# ── Runner ────────────────────────────────────────────────────────────────────

async def run_dataset(dataset: str, batch_dirs: List[str], model: str,
                      llm: LLMHandler, evaluator: Evaluator,
                      data_handler: DataHandler, out_dir: Path):
    print(f"\n===== C4 Self-Diagnosis :: {dataset} / {model} =====")

    ai_records = load_merged_ai(batch_dirs, dataset, model)
    if not ai_records:
        print("  [skip] no Abstention Inflation samples found")
        return

    print(f"  Abstention Inflation samples: {len(ai_records)}")

    # Load dataset for prompt reconstruction
    samples_all = data_handler.load_dataset(dataset)
    samples_all = [s for s in samples_all if s.answer_idx >= 0]
    id_to_sample = {s.id: s for s in samples_all}

    # Build prompts
    prompts: List[List[Dict]] = []
    valid_records: List[Dict] = []
    for rec in ai_records:
        sid = rec["per_sample"]["id"]
        if sid not in id_to_sample:
            continue
        sample = id_to_sample[sid]
        scheme = get_scheme(sample.source)
        s2_msgs = build_judge_s2_prompt(scheme, sample.question, sample.context)
        raw_s2 = rec["per_sample"].get("raw_s2", "")
        prompt = build_c4_prompt(s2_msgs, raw_s2, scheme.abstain_verb)
        prompts.append(prompt)
        valid_records.append(rec)

    if not prompts:
        print("  [skip] no samples matched dataset IDs")
        return

    # Query model
    print(f"  Querying {len(prompts)} samples ...")
    raw_responses = await llm.batch_query(prompts)

    # Parse A/B
    results = []
    ab_counts: Counter = Counter()
    for rec, raw in zip(valid_records, raw_responses):
        ps = rec["per_sample"]
        pred, _ = evaluator.parse_ab_tiered(raw)
        ab_counts[pred if pred in ("A", "B") else "unparseable"] += 1
        results.append({
            "id":        ps["id"],
            "pred_s2":   ps["pred_s2"],
            "pred_c4":   pred,
            "raw_c4":    raw,
        })

    total = len(results)
    pct = {k: round(v / total * 100, 1) for k, v in ab_counts.items()}

    summary = {
        "dataset":    dataset,
        "model":      model,
        "n_ai":      total,
        "counts":     dict(ab_counts),
        "pct":        pct,
        "option_A":   OPTION_A,
        "option_B":   OPTION_B,
        "per_sample": results,
    }

    out_path = out_dir / f"c4_{dataset}_{model}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"  A (gave up):         {ab_counts['A']:3d}  ({pct.get('A', 0):.1f}%)")
    print(f"  B (obj. unsolvable): {ab_counts['B']:3d}  ({pct.get('B', 0):.1f}%)")
    print(f"  unparseable:         {ab_counts['unparseable']:3d}  ({pct.get('unparseable', 0):.1f}%)")
    print(f"  Saved → {out_path}")


async def main(config_path: str):
    config = load_config(config_path)
    model  = config["model_name"]

    llm       = LLMHandler(config)
    evaluator = Evaluator()
    dh        = DataHandler(config)

    out_dir = Path("results/c4_selfdiag")
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_map = config.get("c4_selfdiag", {}).get("batch_dirs", {})
    datasets  = config.get("c4_selfdiag", {}).get("datasets", ["FLD", "FOLIO"])

    for dataset in datasets:
        dirs = batch_map.get(dataset, [])
        await run_dataset(dataset, dirs, model, llm, evaluator, dh, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    asyncio.run(main(args.config))

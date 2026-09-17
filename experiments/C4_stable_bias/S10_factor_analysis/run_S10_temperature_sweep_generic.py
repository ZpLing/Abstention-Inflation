"""Generic temperature sweep runner.

Usage:
    python scripts/run_temperature_sweep_generic.py --config configs/temp_sweep_nano.yaml
    python scripts/run_temperature_sweep_generic.py --config configs/temp_sweep_gemini.yaml

Reads sample IDs from existing batch result files (same paired IDs as main experiment),
queries the model at each temperature, saves summary JSONs to results/temperature_sweep/.
T=0.0 baseline is taken from the model's existing batch results (no new API calls).
"""
import argparse
import asyncio
import json
from pathlib import Path
from typing import Dict, List

ROOT = Path(".")

import sys
sys.path.insert(0, str(ROOT))

from core.config_loader import load_config
from core.llm_handler import LLMHandler
from core.evaluator import Evaluator
from core.label_scheme import get_scheme
from core.dataset_loader import load_judge
from core.prompts import build_judge_s2_prompt


OUT_DIR = ROOT / "results/temperature_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_paired_sample_ids(sources: List[str], n: int) -> List[str]:
    seen = set(); ids = []
    for rel in sources:
        path = ROOT / "results" / rel
        if not path.exists():
            print(f"  [warn] missing {path}, skipping")
            continue
        ab = json.loads(path.read_text())
        for ps in ab.get("per_sample", []):
            sid = ps["id"]
            if sid in seen:
                continue
            seen.add(sid)
            ids.append(sid)
    return ids[:n]


async def batch_query_at_temp(handler: LLMHandler, messages_list, temperature: float):
    async def _one(msg):
        async with handler.semaphore:
            try:
                resp = await handler.client.chat.completions.create(
                    model=handler.model_name,
                    messages=msg,
                    temperature=temperature,
                    max_tokens=handler.max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                return f"__API_ERROR__: {type(e).__name__}: {e}"

    from tqdm.asyncio import tqdm_asyncio
    print(f"  Querying {handler.model_name} (T={temperature}) on {len(messages_list)} prompts ...")
    return await tqdm_asyncio.gather(*[_one(m) for m in messages_list], desc=f"T={temperature}")


async def run_cell(handler, evaluator, scheme, samples, temperature, dataset):
    prompts = [build_judge_s2_prompt(scheme, s.question, s.context) for s in samples]
    raw = await batch_query_at_temp(handler, prompts, temperature)

    counts = {"A": 0, "B": 0, "UNKNOWN": 0, "UNPARSEABLE": 0}
    per_sample = []
    for s, r in zip(samples, raw):
        pred, _ = evaluator.parse_judge_tiered(r, scheme, with_unknown=True)
        key = pred if pred in counts else "UNPARSEABLE"
        counts[key] += 1
        per_sample.append({"id": s.id, "answer_idx": s.answer_idx, "pred": pred, "raw": r})

    n = len(samples)
    n_answerable = sum(1 for s in samples if s.answer_idx >= 0)
    n_ai = sum(1 for s, r in zip(samples, per_sample)
                if s.answer_idx >= 0 and r["pred"] == "UNKNOWN")
    abs_rate = n_ai / n_answerable if n_answerable else 0
    acc = sum(1 for s, r in zip(samples, per_sample)
              if s.answer_idx >= 0 and (
                  (s.answer_idx == 0 and r["pred"] == "A") or
                  (s.answer_idx == 1 and r["pred"] == "B")
              )) / n_answerable if n_answerable else 0

    return {
        "model": handler.model_name,
        "dataset": dataset,
        "temperature": temperature,
        "n": n,
        "n_answerable": n_answerable,
        "abs_rate": abs_rate,
        "Acc": acc,
        "counts": counts,
        "per_sample": per_sample,
    }


async def main(config_path: str):
    config = load_config(config_path)
    model = config["model_name"]
    sweep_cfg = config.get("temperature_sweep", {})
    temperatures = sweep_cfg.get("temperatures", [0.3, 0.7, 1.0])
    datasets = sweep_cfg.get("datasets", ["FLD", "FOLIO"])
    n_samples = sweep_cfg.get("n_samples", 200)
    sample_sources = sweep_cfg.get("sample_sources", {})

    handler = LLMHandler(config)
    evaluator = Evaluator()

    for ds in datasets:
        sources = sample_sources.get(ds, [])
        ids = load_paired_sample_ids(sources, n_samples)
        all_samples = load_judge(ds)
        by_id = {s.id: s for s in all_samples}
        samples = [by_id[i] for i in ids if i in by_id]
        print(f"[{ds}] {len(samples)} paired samples loaded")

        scheme = get_scheme(ds)
        for t in temperatures:
            print(f"\n===== {model} :: {ds} @ T={t} =====")
            summary = await run_cell(handler, evaluator, scheme, samples, t, ds)
            t_tag = f"T{t:.1f}".replace(".", "p")
            out_path = OUT_DIR / f"summary_{t_tag}_{ds}_{model}.json"
            out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
            print(f"  Abs Rate={summary['Abs Rate']:.1%}  Acc={summary['Acc']:.1%}  saved → {out_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    asyncio.run(main(args.config))

"""§5.4 Temperature sweep — self-contained runner.

deepseek-v4-flash × {FLD, FOLIO} × T ∈ {0.3, 0.7, 1.0, 1.5, 2.0} × 200
paired samples × S2 only (T=0.0 baseline reused from main experiment).

Goal:
  Behavioral analog of token-probability probe: test how Abs Rate varies with
  sampling temperature. If Abs Rate(T=1) ≪ Abs Rate(T=0), Unknown wins at greedy by
  thin logit margin → token-level bias. If Abs Rate ≈ flat across T, Unknown has
  robust logit dominance → deeper γ.

DOES NOT modify core/prompts.py or core/evaluator.py / llm_handler.py.
Reuses build_judge_s2_prompt + Evaluator + LLMHandler config; the temperature
override is patched ONLY in this script's runtime (does not persist).

Run:
  python scripts/run_temperature_sweep.py
"""
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(".")
sys.path.insert(0, str(ROOT))

from core.config_loader import load_config           # noqa: E402
from core.llm_handler import LLMHandler              # noqa: E402
from core.evaluator import Evaluator                 # noqa: E402
from core.label_scheme import get_scheme  # noqa: E402
from core.dataset_loader import load_judge
from core.prompts import build_judge_s2_prompt           # noqa: E402

MODEL_NAME = "deepseek-v4-flash"
TEMPERATURES = [1.5, 2.0]
DATASETS = ["FLD", "FOLIO"]
N_SAMPLES = 200
MAX_WORKERS = 100

OUT_DIR = ROOT / "results/temperature_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =================================================================
# Sample-id loader — same paired IDs as main experiment
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
    return ids[:N_SAMPLES]


# =================================================================
# Async batched query at custom temperature
# =================================================================
async def batch_query_at_temp(handler: LLMHandler, messages_list, temperature: float):
    """Mirror of LLMHandler.batch_query but with overridden temperature.
    Bypasses the hardcoded T=0 in src/llm_handler.py without modifying it."""
    async def _one(msg):
        async with handler.semaphore:  # reuse handler's semaphore (max_workers)
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
    tasks = [_one(m) for m in messages_list]
    return await tqdm_asyncio.gather(*tasks, desc=f"T={temperature}")


# =================================================================
# Per-cell run
# =================================================================
async def run_cell(handler, evaluator, scheme, samples,
                    temperature: float, dataset: str):
    prompts = [build_judge_s2_prompt(scheme, s.question, s.context) for s in samples]
    raw = await batch_query_at_temp(handler, prompts, temperature)

    parsed = [evaluator.parse_judge_tiered(r, scheme, with_unknown=True) for r in raw]
    preds = [p[0] for p in parsed]
    tiers = [p[1] for p in parsed]

    n = len(samples)
    n_unknown = sum(1 for p in preds if p == "UNKNOWN")
    n_a = sum(1 for p in preds if p == "A")
    n_b = sum(1 for p in preds if p == "B")
    n_unp = sum(1 for p in preds if p == "UNPARSEABLE")
    abs_rate = n_unknown / n if n else 0.0
    # Acc on answerable samples (gold ∈ {A=0, B=1})
    n_correct = 0
    for p, s in zip(preds, samples):
        if s.answer_idx == 0 and p == "A":
            n_correct += 1
        elif s.answer_idx == 1 and p == "B":
            n_correct += 1
    acc = n_correct / n if n else 0.0

    summary = {
        "temperature": temperature,
        "dataset": dataset,
        "model": MODEL_NAME,
        "n": n,
        "abs_rate": abs_rate,
        "Acc": acc,
        "counts": {"A": n_a, "B": n_b, "UNKNOWN": n_unknown, "UNPARSEABLE": n_unp},
        "tier_counts": {
            t: sum(1 for x in tiers if x == t)
            for t in ("strict_em", "lenient_em", "judge", "unparseable")
        },
        "per_sample": [
            {
                "id": samples[i].id,
                "answer_idx": samples[i].answer_idx,
                "pred": preds[i],
                "tier": tiers[i],
                "raw": raw[i],
            }
            for i in range(n)
        ],
    }
    return summary


async def main():
    # Reuse car_mirror_deepseek.yaml for credentials + model name
    config = load_config(str(ROOT / "configs/car_mirror_deepseek.yaml"))
    config["max_workers"] = MAX_WORKERS
    config["model_name"] = MODEL_NAME
    handler = LLMHandler(config)
    evaluator = Evaluator()

    # Pre-load sample lookups
    samples_by_ds: Dict[str, List] = {}
    for ds in DATASETS:
        all_samples = load_judge(ds)
        by_id = {s.id: s for s in all_samples}
        ids = load_paired_sample_ids(ds)
        picked = [by_id[i] for i in ids if i in by_id]
        print(f"[{ds}] picked {len(picked)} paired samples (target {N_SAMPLES})")
        samples_by_ds[ds] = picked

    for ds in DATASETS:
        scheme = get_scheme(ds)
        samples = samples_by_ds[ds]
        for t in TEMPERATURES:
            print(f"\n===== {MODEL_NAME} :: {ds} @ T={t} =====")
            summary = await run_cell(handler, evaluator, scheme, samples, t, ds)
            t_tag = f"T{t:.1f}".replace(".", "p")  # T0p3 / T0p7 / T1p0
            out_path = OUT_DIR / f"summary_{t_tag}_{ds}_{MODEL_NAME}.json"
            out_path.write_text(json.dumps(summary, indent=2))
            print(f"  → Abs Rate={summary['Abs Rate']:.1%}  Acc={summary['Acc']:.1%}  "
                  f"(n={summary['n']}, A={summary['counts']['A']}, "
                  f"B={summary['counts']['B']}, UNK={summary['counts']['UNKNOWN']}, "
                  f"UNP={summary['counts']['UNPARSEABLE']})")
            print(f"  saved {out_path.name}")


if __name__ == "__main__":
    asyncio.run(main())

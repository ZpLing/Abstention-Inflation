"""§6.3 Re-prompting persistence (paper T-5C).

For each Abstention Inflation sample (pred_s2 == UNKNOWN), re-run the *same* S2 prompt
N=3 times at temperature > 0. Measure how often the model still returns
UNKNOWN. Target: >95% persistence → abstain is deterministic policy,
not sampling lottery.

Usage:
    python -m scripts.run_reprompting_persistence \
        --summary results/ab_gpt5_nano/ab_summary_FLD_gpt-5.4-nano.json \
                  results/ab_nano_batch2/ab_summary_FLD_gpt-5.4-nano.json \
        --dataset FLD --model gpt-5.4-nano --n_repeats 3 --temperature 0.5 \
        --out results/persistence/<model>_<dataset>.json
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.config_loader import load_config
from core.llm_handler import LLMHandler
from core.evaluator import Evaluator
from core.dataset_loader import load_judge
from core.label_scheme import get_scheme
from core.prompts import build_judge_s2_prompt


def _is_ai(s):
    return s.get("pred_s2") == "UNKNOWN"


async def rerun_one(handler, prompts, n_repeats, temperature):
    """Rerun the same prompt list n_repeats times. Returns list of list of responses."""
    results = []  # results[rep][i] = response string
    for rep in range(n_repeats):
        # batch_query uses temperature=0 hardcoded; we need to override
        responses = await handler.batch_query_temp(prompts, temperature=temperature)
        results.append(responses)
    return results


# Patch LLMHandler with temperature parameter (used only here)
async def batch_query_temp(self, messages, temperature):
    """Same as batch_query but with arbitrary temperature."""
    sem = self.semaphore
    async def one(msg):
        async with sem:
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model_name, messages=msg, temperature=temperature,
                    max_tokens=self.max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                return f"__API_ERROR__: {e}"
    tasks = [one(m) for m in messages]
    return await asyncio.gather(*tasks)
LLMHandler.batch_query_temp = batch_query_temp


def parse_pred(text, evaluator, scheme):
    """TF parse: PROVED → 'A', DISPROVED → 'B', UNKNOWN → 'UNKNOWN'."""
    pred, _tier = evaluator.parse_judge_tiered(text, scheme, with_unknown=True)
    return pred


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, nargs="+")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n_repeats", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # Load config
    cfg = load_config(args.config)
    cfg["model_name"] = args.model
    cfg["max_workers"] = 30  # Concurrency boost for repeated queries
    handler = LLMHandler(cfg)
    evaluator = Evaluator()

    # Load FLD/FOLIO raw
    raw = load_judge(args.dataset)
    id_to_sample = {s.id: s for s in raw}
    scheme = get_scheme(args.dataset)

    # Load summaries, get Abstention Inflation samples
    ai = []
    for sp in args.summary:
        d = json.loads(Path(sp).read_text())
        for ps in d.get("per_sample", []):
            if _is_ai(ps):
                sid = ps["id"]
                if sid in id_to_sample:
                    abs_rate.append((sid, id_to_sample[sid], ps))
    print(f"Loaded {len(ai)} Abstention Inflation samples for {args.model}/{args.dataset}")

    if not abs_rate:
        print("No Abstention Inflation samples"); return

    # Build prompts
    prompts = [build_judge_s2_prompt(scheme, s.question, s.context) for _, s, _ in abs_rate]
    print(f"Running {args.n_repeats} reruns at T={args.temperature} ...")

    rerun_outputs = await rerun_one(handler, prompts, args.n_repeats, args.temperature)

    # Parse + persistence count per sample
    rows = []
    for i, (sid, s, ps) in enumerate(abs_rate):
        preds = []
        for rep_outs in rerun_outputs:
            preds.append(parse_pred(rep_outs[i], evaluator, scheme))
        n_unk = sum(1 for p in preds if p == "UNKNOWN")
        rows.append({
            "id": sid, "n_repeats": args.n_repeats,
            "preds": preds, "n_unknown": n_unk,
            "persistence": n_unk / args.n_repeats,
        })

    # Aggregate
    n = len(rows)
    full_persist = sum(1 for r in rows if r["n_unknown"] == args.n_repeats)
    any_flip    = sum(1 for r in rows if r["n_unknown"] < args.n_repeats)
    avg_persist = sum(r["persistence"] for r in rows) / max(n, 1)

    print()
    print(f"Persistence summary (n={n} Abstention Inflation samples × {args.n_repeats} reruns @ T={args.temperature}):")
    print(f"  All {args.n_repeats} draws returned Unknown: {full_persist}/{n} = {full_persist/n:.1%}")
    print(f"  At least 1 flip (≠ Unknown):       {any_flip}/{n} = {any_flip/n:.1%}")
    print(f"  Mean persistence rate:           {avg_persist:.1%}")
    print()
    # Distribution of n_unknown
    dist = Counter(r["n_unknown"] for r in rows)
    print("n_unknown distribution (higher = more stable):")
    for k in sorted(dist.keys(), reverse=True):
        print(f"  {k}/{args.n_repeats}: {dist[k]} ({dist[k]/n:.1%})")

    out = args.out or f"results/persistence/{args.model}_{args.dataset}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({
        "model": args.model, "dataset": args.dataset,
        "n_ai": n, "n_repeats": args.n_repeats, "temperature": args.temperature,
        "full_persistence": full_persist / n,
        "avg_persistence": avg_persist,
        "n_unknown_distribution": {str(k): v for k, v in dist.items()},
        "rows": rows,
    }, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    asyncio.run(main())

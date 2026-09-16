"""CoT ablation for reviewer concern R4.4 — "do results hold without CoT prompting?"

Runs the S1 (no-Unknown) and S2 (inline-Unknown) abstention settings under two
prompt formats that differ in ONE variable only:

    CoT     : "Reasoning: ...\\nFinal answer: <X>"   (the paper's default)
    no-CoT  : "Answer with only <X>, with no reasoning or explanation."

Everything else (task instruction, options, context) is byte-for-byte identical
(see core/prompts.py `cot=` kwarg). The same fixed subsample is used
across both formats, so abstention behavior is compared within-subject and tested
with a paired (McNemar) test.

Key quantity: Abs Rate = fraction of S2 answers that are Unknown/abstain. If Abs Rate is
statistically indistinguishable between CoT and no-CoT, the abstention-inflation
effect is NOT an artifact of chain-of-thought prompting.

Models & endpoint come from the repo `config` (API gateway) merged via
src.config_loader — deepseek-v4-flash (reasoning) and
gemini-3.1-flash-lite (standard) share one api_key/base_url.

Run:
    python -m scripts.run_no_cot_ablation --n-per-class 50
    python -m scripts.run_no_cot_ablation --n-per-class 4 --models gemini-3.1-flash-lite   # smoke
"""
import argparse
import asyncio
import json
import random
from pathlib import Path
from typing import Dict, List

from scipy.stats import binomtest

from core.label_scheme import get_scheme
from core.dataset_loader import load_judge
from core.prompts import build_judge_s1_prompt, build_judge_s2_prompt
from core.config_loader import load_config
from core.evaluator import Evaluator
from core.llm_handler import LLMHandler

SEED = 42
DEFAULT_MODELS = ["gemini-3.1-flash-lite", "deepseek-v4-flash"]
DEFAULT_DATASETS = ["FLD", "FOLIO"]
OUT_DIR = Path("results/no_cot_ablation")


def balanced_subsample(samples, k_per_class: int, rng: random.Random) -> list:
    """k POS (answer_idx==0) + k NEG (answer_idx==1), deterministic given rng."""
    pos = [s for s in samples if s.answer_idx == 0]
    neg = [s for s in samples if s.answer_idx == 1]
    rng.shuffle(pos)
    rng.shuffle(neg)
    if len(pos) < k_per_class or len(neg) < k_per_class:
        raise RuntimeError(f"Need {k_per_class}/class; got POS={len(pos)} NEG={len(neg)}.")
    return pos[:k_per_class] + neg[:k_per_class]


def build_subsamples(datasets, k_per_class: int) -> Dict[str, list]:
    rng = random.Random(SEED)
    out = {}
    for ds in datasets:
        samples = [s for s in load_judge(ds) if s.answer_idx >= 0]
        out[ds] = balanced_subsample(samples, k_per_class, rng)
    return out


def parse_preds(raw_outputs, samples, evaluator, *, with_unknown: bool):
    preds, tiers = [], []
    for raw, s in zip(raw_outputs, samples):
        p, t = evaluator.parse_judge_tiered(raw, get_scheme(s.source),
                                            with_unknown=with_unknown)
        preds.append(p)
        tiers.append(t)
    return preds, tiers


def acc(preds, samples) -> float:
    # POS→"A", NEG→"B" in canonical parse space; answer_idx 0/1.
    gold = ["A" if s.answer_idx == 0 else "B" for s in samples]
    return sum(p == g for p, g in zip(preds, gold)) / len(samples)


def abs_rate(preds) -> float:
    return sum(1 for p in preds if p == "UNKNOWN") / len(preds)


def mcnemar_exact(paired_a: List[bool], paired_b: List[bool]) -> dict:
    """Exact McNemar on two paired binary vectors (abstained? per sample)."""
    b = sum(1 for x, y in zip(paired_a, paired_b) if x and not y)
    c = sum(1 for x, y in zip(paired_a, paired_b) if y and not x)
    n = b + c
    p = binomtest(min(b, c), n, 0.5).pvalue if n > 0 else 1.0
    return {"b_cot_only": b, "c_nocot_only": c, "discordant": n, "p_value": p}


async def run_model_dataset(model: str, ds: str, samples: list,
                            base_cfg: dict, evaluator: Evaluator) -> dict:
    cfg = dict(base_cfg)
    cfg["model_name"] = model
    llm = LLMHandler(cfg)
    scheme = get_scheme(ds)

    results = {}
    for fmt, cot in [("cot", True), ("nocot", False)]:
        s1 = [build_judge_s1_prompt(scheme, s.question, s.context, cot=cot) for s in samples]
        s2 = [build_judge_s2_prompt(scheme, s.question, s.context, cot=cot) for s in samples]
        raw_s1, raw_s2 = await asyncio.gather(llm.batch_query(s1), llm.batch_query(s2))
        p1, _ = parse_preds(raw_s1, samples, evaluator, with_unknown=False)
        p2, _ = parse_preds(raw_s2, samples, evaluator, with_unknown=True)
        results[fmt] = {
            "acc_s1": acc(p1, samples),
            "acc_s2": acc(p2, samples),
            "abs_rate_s2": abs_rate(p2),
            "n_unparseable_s2": sum(1 for p in p2 if p == "UNPARSEABLE"),
            "abstain_flags": [p == "UNKNOWN" for p in p2],
            "preds_s2": p2,
            "raw_s2": raw_s2,
        }
        print(f"    [{model} | {ds} | {fmt:>5}] "
              f"Acc_L S1={results[fmt]['acc_s1']:.1%} S2={results[fmt]['acc_s2']:.1%}  "
              f"Abs Rate={results[fmt]['abs_rate_s2']:.1%}  "
              f"unparse={results[fmt]['n_unparseable_s2']}")

    mc = mcnemar_exact(results["cot"]["abstain_flags"], results["nocot"]["abstain_flags"])
    abs_rate_delta = results["nocot"]["abs_rate_s2"] - results["cot"]["abs_rate_s2"]
    print(f"    -> Abs Rate(no-CoT) - Abs Rate(CoT) = {abs_rate_delta:+.1%}  "
          f"| McNemar b={mc['b_cot_only']} c={mc['c_nocot_only']} p={mc['p_value']:.3f}")
    return {
        "model": model, "dataset": ds, "n": len(samples),
        "cot": {k: v for k, v in results["cot"].items() if k not in ("abstain_flags",)},
        "nocot": {k: v for k, v in results["nocot"].items() if k not in ("abstain_flags",)},
        "abs_rate_delta_nocot_minus_cot": abs_rate_delta,
        "mcnemar": mc,
    }


async def main_async(args):
    base_cfg = load_config("configs/experiment.yaml")
    base_cfg["max_workers"] = args.max_workers
    base_cfg["override_model"] = True  # we set model_name per run
    print(f"Endpoint: {base_cfg.get('base_url')!r}  max_workers={base_cfg['max_workers']}")
    if not base_cfg.get("api_key"):
        raise SystemExit("No api_key resolved — need secrets.yaml or repo `config`.")

    evaluator = Evaluator()
    subsamples = build_subsamples(args.datasets, args.n_per_class)
    for name, sub in subsamples.items():
        print(f"  {name}: {len(sub)} samples ({args.n_per_class}/class)")

    rows = []
    for model in args.models:
        print(f"\n===== MODEL: {model} =====")
        for ds in args.datasets:
            rows.append(await run_model_dataset(model, ds, subsamples[ds], base_cfg, evaluator))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"ablation_n{args.n_per_class}.json"
    out.write_text(json.dumps(
        {"seed": SEED, "n_per_class": args.n_per_class,
         "models": args.models, "datasets": args.datasets, "rows": rows},
        indent=2, ensure_ascii=False))
    print(f"\nWrote {out}")
    print_summary(rows)


def print_summary(rows):
    print("\n" + "=" * 100)
    print(f"{'Model':<30} {'DS':<6} {'Abs Rate CoT':>8} {'Abs Rate noCoT':>10} {'Δ':>7}  "
          f"{'AccS2 CoT':>10} {'AccS2 noCoT':>12}  {'McNemar p':>10}")
    print("-" * 100)
    for r in rows:
        print(f"{r['model']:<30} {r['dataset']:<6} "
              f"{r['cot']['abs_rate_s2']:>8.1%} {r['nocot']['abs_rate_s2']:>10.1%} "
              f"{r['abs_rate_delta_nocot_minus_cot']:>+7.1%}  "
              f"{r['cot']['acc_s2']:>10.1%} {r['nocot']['acc_s2']:>12.1%}  "
              f"{r['mcnemar']['p_value']:>10.3f}")
    print("=" * 100)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=50)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--max-workers", type=int, default=100)
    return ap.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))

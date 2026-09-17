"""CoT ablation (R4.4), larger-n version that REUSES the paper's reported CoT
baseline instead of re-querying it.

For each (model, dataset):
  * Abs Rate (CoT)   = taken from the paper's per-sample S2 results on disk
                       (pooled batch1+batch2, n≈200) — no new API calls.
  * Abs Rate (no-CoT)= a fresh direct-answer run on the SAME sample IDs, so the
                       two conditions are paired within-subject → McNemar valid.

Only the trailing format instruction differs between the paper's S2 prompt and
ours (see core/prompts.py `cot=` kwarg); the prompt body is identical.

Models/endpoint from the repo `config` (API gateway). gemini-2.5-flash-lite is
the clean test (standard model); deepseek-r1-distill is a *reasoning* model that
emits CoT internally regardless of the prompt, so it is reported as a secondary
data point with that caveat.

Run:
    python -m scripts.run_no_cot_paper_baseline
    python -m scripts.run_no_cot_paper_baseline --only gemini-2.5-flash-lite
"""
import argparse
import asyncio
import json
from pathlib import Path
from typing import Dict, List

from scipy.stats import binomtest

from core.label_scheme import get_scheme
from core.dataset_loader import load_judge
from core.prompts import build_judge_s2_prompt
from core.config_loader import load_config
from core.evaluator import Evaluator
from core.llm_handler import LLMHandler

OUT_DIR = Path("results/cot_ablation")

# Paper's per-sample S2 (CoT) results. Files are pooled in order; duplicate IDs
# across files are dropped (first occurrence wins) so the union is distinct.
COT_SOURCES: Dict[tuple, List[str]] = {
    ("gemini-2.5-flash-lite", "FLD"): [
        "results/ab_gemini_flash_lite/ab_summary_FLD_gemini-2.5-flash-lite.json",
        "results/ab_gemini_batch2/ab_summary_FLD_gemini-2.5-flash-lite.json",
    ],
    ("gemini-2.5-flash-lite", "FOLIO"): [
        "results/ab_gemini_flash_lite/ab_summary_FOLIO_gemini-2.5-flash-lite.json",
        "results/ab_gemini_batch2/ab_summary_FOLIO_gemini-2.5-flash-lite.json",
    ],
    ("deepseek-r1-distill-llama-8b", "FLD"): [
        "results/ab/ab_summary_FLD_deepseek-r1-distill-llama-8b.json",
        "results/ab_deepseek_batch2/ab_summary_FLD_deepseek-r1-distill-llama-8b.json",
    ],
    ("deepseek-r1-distill-llama-8b", "FOLIO"): [
        "results/ab_deepseek_batch2/ab_summary_FOLIO_deepseek-r1-distill-llama-8b.json",
    ],
    ("gpt-5.4-nano", "FLD"): [
        "results/ab_gpt5_nano/ab_summary_FLD_gpt-5.4-nano.json",
        "results/ab_nano_batch2/ab_summary_FLD_gpt-5.4-nano.json",
    ],
    ("gpt-5.4-nano", "FOLIO"): [
        "results/ab_gpt5_nano/ab_summary_FOLIO_gpt-5.4-nano.json",
        "results/ab_nano_batch2/ab_summary_FOLIO_gpt-5.4-nano.json",
    ],
}


def load_cot_baseline(paths: List[str]) -> Dict[str, bool]:
    """Pool per-sample S2 results → {sample_id: cot_abstained}. First file wins on dup id."""
    flags: Dict[str, bool] = {}
    for p in paths:
        if not Path(p).exists():
            print(f"  [warn] missing CoT source {p}")
            continue
        for r in json.load(open(p)).get("per_sample", []):
            if r["id"] not in flags:
                flags[r["id"]] = (r.get("pred_s2") == "UNKNOWN")
    return flags


def mcnemar_exact(cot_flags: List[bool], nocot_flags: List[bool]) -> dict:
    b = sum(1 for x, y in zip(cot_flags, nocot_flags) if x and not y)  # CoT-only abstain
    c = sum(1 for x, y in zip(cot_flags, nocot_flags) if y and not x)  # noCoT-only abstain
    n = b + c
    p = binomtest(min(b, c), n, 0.5).pvalue if n > 0 else 1.0
    return {"b_cot_only": b, "c_nocot_only": c, "discordant": n, "p_value": p}


async def run_one(model: str, ds: str, base_cfg: dict, evaluator: Evaluator) -> dict:
    cot_flags_by_id = load_cot_baseline(COT_SOURCES[(model, ds)])
    by_id = {s.id: s for s in load_judge(ds)}
    ids = [i for i in cot_flags_by_id if i in by_id]
    missing = [i for i in cot_flags_by_id if i not in by_id]
    if missing:
        print(f"  [warn] {len(missing)} CoT ids not found in loader; dropped.")
    samples = [by_id[i] for i in ids]
    scheme = get_scheme(ds)

    cfg = dict(base_cfg); cfg["model_name"] = model
    llm = LLMHandler(cfg)
    prompts = [build_judge_s2_prompt(scheme, s.question, s.context, cot=False) for s in samples]
    print(f"  [{model} | {ds}] no-CoT S2 on n={len(samples)} paired samples ...")
    raw = await llm.batch_query(prompts)
    nocot_pred = [evaluator.parse_judge_tiered(r, scheme, with_unknown=True)[0] for r in raw]

    # The gateway's content filter false-positives on FLD's nonsense vocabulary
    # ("pornographer", "benzodiazepine" as substrings) → __API_ERROR__. Drop those
    # samples from the paired comparison rather than miscounting them as non-abstain.
    valid = [k for k in range(len(ids)) if raw[k] != "__API_ERROR__"]
    n_dropped = len(ids) - len(valid)

    cot_flags_full = [cot_flags_by_id[i] for i in ids]
    cot_flags = [cot_flags_full[k] for k in valid]
    nocot_flags = [nocot_pred[k] == "UNKNOWN" for k in valid]

    # Headline CoT number = the paper's full-n rate (what we report); McNemar is on
    # the valid paired subset.
    cot_abs_rate_paper = sum(cot_flags_full) / len(cot_flags_full)
    nocot_abs_rate = sum(nocot_flags) / len(nocot_flags)
    cot_abs_rate_valid = sum(cot_flags) / len(cot_flags)
    mc = mcnemar_exact(cot_flags, nocot_flags)
    n_unparse = sum(1 for k in valid if nocot_pred[k] == "UNPARSEABLE")
    print(f"    Abs Rate CoT(paper)={cot_abs_rate_paper:.1%} ({sum(cot_flags_full)}/{len(cot_flags_full)})  "
          f"no-CoT={nocot_abs_rate:.1%} ({sum(nocot_flags)}/{len(nocot_flags)})  "
          f"Δ={nocot_abs_rate-cot_abs_rate_valid:+.1%}  McNemar b={mc['b_cot_only']} c={mc['c_nocot_only']} "
          f"p={mc['p_value']:.4f}  dropped(API-filter)={n_dropped} unparse={n_unparse}")
    return {
        "model": model, "dataset": ds, "n": len(ids), "n_valid": len(valid),
        "n_dropped_api_filter": n_dropped,
        "abs_rate_cot_paper": cot_abs_rate_paper, "abs_rate_cot_valid": cot_abs_rate_valid,
        "abs_rate_nocot": nocot_abs_rate,
        "delta": nocot_abs_rate - cot_abs_rate_valid, "mcnemar": mc, "n_unparseable_nocot": n_unparse,
        "cot_source": COT_SOURCES[(model, ds)],
        "per_sample": [
            {"id": ids[k], "cot_abstain": cot_flags_full[k],
             "nocot_abstain": nocot_pred[k] == "UNKNOWN", "nocot_pred": nocot_pred[k],
             "api_error": raw[k] == "__API_ERROR__"}
            for k in range(len(ids))
        ],
    }


async def main_async(args):
    base_cfg = load_config("configs/experiment.yaml")
    base_cfg["max_workers"] = args.max_workers
    base_cfg["override_model"] = True
    if not base_cfg.get("api_key"):
        raise SystemExit("No api_key resolved — need secrets.yaml or repo `config`.")
    print(f"Endpoint: {base_cfg.get('base_url')!r}  max_workers={base_cfg['max_workers']}")
    evaluator = Evaluator()

    pairs = [k for k in COT_SOURCES if args.only is None or k[0] == args.only]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "paper_baseline_nocot.json"
    rows = []
    for model, ds in pairs:
        rows.append(await run_one(model, ds, base_cfg, evaluator))
        # Incremental save after each (model, dataset) so a timeout can't lose results.
        out.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False))
        print(f"    [saved {len(rows)}/{len(pairs)} rows -> {out}]")
    print(f"\nWrote {out}")
    print("\n" + "=" * 104)
    print(f"{'Model':<30} {'DS':<6} {'n':>4}  {'Abs CoT(paper)':>14} {'Abs no-CoT':>11} "
          f"{'Δ':>7}  {'McNemar p':>10}")
    print("-" * 104)
    for r in rows:
        print(f"{r['model']:<30} {r['dataset']:<6} {r['n']:>4}  "
              f"{r['abs_rate_cot_paper']:>14.1%} {r['abs_rate_nocot']:>11.1%} "
              f"{r['delta']:>+7.1%}  {r['mcnemar']['p_value']:>10.4f}")
    print("=" * 104)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="restrict to one model name")
    ap.add_argument("--max-workers", type=int, default=100)
    return ap.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))

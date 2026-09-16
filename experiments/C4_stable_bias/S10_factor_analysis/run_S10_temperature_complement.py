"""S10(a) temperature sweep — the 300 items per dataset the first sweep missed.

The original sweep drew the 200 pooled batch-1 + batch-2 items; the paper
reports n=500 per dataset. This runs the complement so the two can be merged,
rather than re-running the 200 that are already done.

Everything that could change a number is held identical to the first sweep:
same S2 prompt builder, same `Evaluator.parse_judge_tiered`, same per-cell
output schema, same `max_tokens` (whatever `LLMHandler` reads from config), and
the temperature override applied the same way. Only the item set differs.

Note the first sweep's cells are also affected by a truncation problem found
later: on FLD roughly half of Gemini's outputs run to the token cap without ever
emitting a `Final answer:` line, and the whole-text fallback then reads a stray
"cannot determine" as an abstention. Raising the cap here would produce cleaner
items that are *not comparable* to the existing 200, so the cap is deliberately
left alone. Treat the truncation as a separate finding, not something to fix
halfway through a sample-size extension.

    python run_S10_temperature_complement.py --limit 20     # smoke
    python run_S10_temperature_complement.py                # full
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from core.config_loader import load_config          # noqa: E402
from core.dataset_loader import load_judge          # noqa: E402
from core.evaluator import Evaluator                # noqa: E402
from core.label_scheme import get_scheme            # noqa: E402
from core.llm_handler import LLMHandler             # noqa: E402
from core.prompts import build_judge_s2_prompt      # noqa: E402

MODELS = ["deepseek-v4-flash", "gpt-5.4-nano", "gemini-3.1-flash-lite"]
TEMPERATURES = [0.3, 0.7, 1.0, 1.5, 2.0]
DATASETS = ["FLD", "FOLIO"]
SWEEP_DIR = ROOT / "results" / "temperature_sweep"
OUT_DIR = ROOT / "results" / "temperature_sweep_p2"


def already_run_ids(dataset: str) -> set:
    """Item ids the first sweep covered, read from its own outputs."""
    ids = set()
    for f in SWEEP_DIR.glob(f"summary_T0p3_{dataset}_*.json"):
        ids |= {r["id"] for r in json.loads(f.read_text())["per_sample"]}
    return ids


def complement_samples(dataset: str, limit: int | None = None) -> List:
    """Answerable items the first sweep did not cover, in dataset order."""
    done = already_run_ids(dataset)
    if not done:
        raise SystemExit(f"no first-sweep outputs for {dataset}; refusing to guess "
                         f"which items are missing")
    out = [s for s in load_judge(dataset) if s.answer_idx >= 0 and s.id not in done]
    return out[:limit] if limit else out


async def query_at_temp(handler: LLMHandler, messages, temperature: float,
                        retries: int = 4):
    """`LLMHandler.batch_query` with the temperature overridden for this call
    only -- the handler hardcodes T=0 and is shared, so it is not mutated.

    Retries with backoff, which `LLMHandler` does not do. The gateway answers
    overload with `400 "The model request failed, please try again later"`, and
    without a retry those land in the data as `__API_ERROR__`, parse as
    UNPARSEABLE, and depress the cell's Abs Rate. That is not hypothetical: in
    the first sweep GPT-5.4-nano lost 39 of 200 calls at T=1.5, which alone
    produced a 13-point dip and made an otherwise flat temperature curve look
    like it moved.
    """
    async def one(msg):
        delay = 2.0
        last = ""
        for attempt in range(retries + 1):
            async with handler.semaphore:
                try:
                    resp = await handler.client.chat.completions.create(
                        model=handler.model_name, messages=msg,
                        temperature=temperature, max_tokens=handler.max_tokens)
                    return resp.choices[0].message.content or ""
                except Exception as exc:                            # noqa: BLE001
                    last = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                await asyncio.sleep(delay)
                delay *= 2
        return f"__API_ERROR__: {last}"

    from tqdm.asyncio import tqdm_asyncio
    return await tqdm_asyncio.gather(*[one(m) for m in messages],
                                     desc=f"    T={temperature}")


async def run_cell(handler, ev, scheme, samples, temperature, dataset, model):
    raw = await query_at_temp(
        handler, [build_judge_s2_prompt(scheme, s.question, s.context) for s in samples],
        temperature)
    parsed = [ev.parse_judge_tiered(r, scheme, with_unknown=True) for r in raw]
    preds, tiers = [p for p, _ in parsed], [t for _, t in parsed]
    n = len(samples)
    n_unk = sum(1 for p in preds if p == "UNKNOWN")
    n_err = sum(1 for r in raw if r.startswith("__API_ERROR__"))
    correct = sum(1 for p, s in zip(preds, samples)
                  if (s.answer_idx == 0 and p == "A") or (s.answer_idx == 1 and p == "B"))
    return {
        "temperature": temperature, "dataset": dataset, "model": model, "n": n,
        "abs_rate": n_unk / n if n else 0.0,
        "Acc": correct / n if n else 0.0,
        "n_api_errors": n_err,
        "counts": {k: sum(1 for p in preds if p == k)
                   for k in ("A", "B", "UNKNOWN", "UNPARSEABLE")},
        "tier_counts": {t: sum(1 for x in tiers if x == t)
                        for t in ("strict_em", "lenient_em", "judge", "unparseable")},
        "run_config": {"class_offset": "complement-of-first-sweep",
                       "max_tokens": handler.max_tokens,
                       "max_workers": handler.max_workers
                       if hasattr(handler, "max_workers") else None},
        "per_sample": [{"id": samples[i].id, "answer_idx": samples[i].answer_idx,
                        "pred": preds[i], "tier": tiers[i], "raw": raw[i]}
                       for i in range(n)],
    }


def tag(t: float) -> str:
    return "T" + str(t).replace(".", "p")


async def main_async(args):
    cfg = load_config(args.config)
    ev = Evaluator()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for model in (args.models or MODELS):
        sub = dict(cfg); sub["model_name"] = model; sub["max_workers"] = args.max_workers
        handler = LLMHandler(sub)
        for ds in (args.datasets or DATASETS):
            samples = complement_samples(ds, args.limit)
            scheme = get_scheme(ds)
            print(f"\n=== {model} / {ds}: {len(samples)} complement items ===", flush=True)
            for t in (args.temperatures or TEMPERATURES):
                out = OUT_DIR / f"summary_{tag(t)}_{ds}_{model}.json"
                if out.exists() and not args.overwrite:
                    print(f"    T={t} already done, skipping")
                    continue
                s = await run_cell(handler, ev, scheme, samples, t, ds, model)
                err_rate = s["n_api_errors"] / max(s["n"], 1)
                if err_rate > args.max_error_rate:
                    # Writing this would put failures into the Abs Rate as
                    # UNPARSEABLE and quietly bias the cell downward.
                    raise SystemExit(
                        f"\n{model} / {ds} / T={t}: {s['n_api_errors']}/{s['n']} "
                        f"calls failed ({err_rate:.0%} > {args.max_error_rate:.0%}). "
                        f"Not writing this cell. Lower --max_workers and rerun.")
                out.write_text(json.dumps(s, indent=2))
                err = f"  api_errors={s['n_api_errors']}" if s["n_api_errors"] else ""
                print(f"    T={t}  AbsRate={s['abs_rate']:.1%}  Acc={s['Acc']:.1%}"
                      f"  {s['counts']}{err}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/C4_stable_bias/S10_temperature_GPT_5_4_nano.yaml"))
    ap.add_argument("--limit", type=int, default=None, help="cap items (smoke test)")
    ap.add_argument("--max_workers", type=int, default=20,
                    help="The gateway starts returning 400s well below 100; "
                         "20 measured clean.")
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--datasets", nargs="+",
                    help="Restrict to these datasets (default: both).")
    ap.add_argument("--temperatures", nargs="+", type=float)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max_error_rate", type=float, default=0.02,
                    help="Abort rather than write a cell with more "
                         "failures than this.")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()

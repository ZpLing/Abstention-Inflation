"""S10 modulator (a) — temperature sweep, gateway models.

T in {0, 0.3, 0.7, 1.0, 1.5, 2.0} at 250 items per class. Only the models
whose temperature the endpoint actually applies are swept; for the rest the
setting is ignored, which is itself an S10 finding.

Model is selected with --model; the one reported is `gemini-3.1-flash-lite`.
It was screened first, and the screen is not optional: of the gateway models tested, gpt-5.4-nano,
o4-mini, gpt-5.5, gpt-5.6-sol, gemini-3.7-flash, kimi-k3, glm-5.3 and
glm-5.3-flash all accept a `temperature` argument and return the same answer
distribution at T=0 and T=2, while qwen3-max, qwen3.7-plus, qwen3.8-max,
ernie-5.1, llama3-70b and glm-4.7/5.0/5.1/5.2 reject T>1 outright. A sweep on
any of those measures nothing.

Why a fresh sweep rather than a complement
------------------------------------------
`results/temperature_sweep{,_p2}` were generated with **gemini-3.1-flash-lite**,
although every config file, figure label and paper mention calls the model
Gemini-3.1-Flash-Lite: the rename reached the file names
(`S10_temperature_Gemini_3_1_Flash_Lite.yaml`) and the display map in
the figure data layer, but never the `model_name` field. So the old cells cannot
be extended, only replaced.

Three things are aligned with the local sweeps (`run_temp_olmo_topk.sbatch`,
`run_temp_qwen_long.sbatch`) so all three models' curves are comparable:

* **The same 500 items** -- `select_samples(ds, 250, 0)`, i.e. 250 per gold
  class in dataset order. The 2.5 sweep drew a pooled 200 plus a 300-item
  complement, which is a different set.
* **The same parser** -- `_classify` from the local runner, so every cell
  reports `trusted_share` and a provenance breakdown. The old sweep stored only
  a bare `pred`, which is why its truncation problem could not be quantified
  from the summaries alone.
* **The same summary schema** -- `summarize()`, so the analysis and figure code
  reads all three models identically.

`max_tokens` is raised to 8192. At the default 4096 roughly a third of 2.5's
FLD generations ran to the cap without ever emitting a `Final answer:` line,
and the whole-text fallback then read a stray "cannot determine" as an
abstention. There is no comparability reason to keep that cap here: this run
replaces the 2.5 cells rather than extending them.

    python experiments/C4_stable_bias/S10_factor_analysis/run_S10_temperature.py \
        --model gemini-3.1-flash-lite --limit 8     # smoke
    python run_S10_temperature.py --model gemini-3.1-flash-lite   # full, 12 cells x 500"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from infra.label_scheme import get_scheme  # noqa: E402
from infra.llm_handler import LLMHandler  # noqa: E402
from infra.prompts import build_judge_s2_prompt  # noqa: E402
from infra.result_schema import results_dir, stamp
from loader.config_loader import load_config  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_s10_runner", Path(__file__).with_name("run_S10_local_sweep.py")
)
_runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runner)

TEMPERATURES = [0.0, 0.3, 0.7, 1.0, 1.5, 2.0]
DATASETS = ["FLD", "FOLIO"]
N_PER_CLASS = 250
MAX_TOKENS = 8192
SLUG_OF = {
    "gemini-3.1-flash-lite": "gemini_3.1_flash_lite",
}
CFG = ROOT / "configs" / "C4_stable_bias" / "S10_temperature_Gemini_3_1_Flash_Lite.yaml"


async def query_at_temp(
    handler: LLMHandler, messages, temperature: float, retries: int = 4
):
    """`batch_query` with the temperature overridden for this call only.

    The handler hardcodes T=0 and is shared, so it is not mutated. Retries with
    backoff, which `LLMHandler` does not do: the gateway answers overload with
    `400 "The model request failed, please try again later"`, and without a
    retry those land in the data as `__API_ERROR__`, parse as UNPARSEABLE, and
    depress the cell's Abs Rate. In the 2.5 sweep GPT-5.4-nano lost 39 of 200
    calls at T=1.5, which alone produced a 13-point dip and made an otherwise
    flat curve look like it moved.
    """

    async def one(msg):
        delay, last = 2.0, ""
        for attempt in range(retries + 1):
            async with handler.semaphore:
                try:
                    resp = await handler.client.chat.completions.create(
                        model=MODEL,
                        messages=msg,
                        temperature=temperature,
                        max_tokens=MAX_TOKENS,
                    )
                    return resp.choices[0].message.content or ""
                except Exception as exc:  # noqa: BLE001
                    last = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                await asyncio.sleep(delay)
                delay *= 2
        return f"__API_ERROR__: {last}"

    from tqdm.asyncio import tqdm_asyncio

    return await tqdm_asyncio.gather(
        *[one(m) for m in messages], desc=f"    T={temperature}"
    )


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3.1-flash-lite", choices=sorted(SLUG_OF))
    ap.add_argument(
        "--limit", type=int, default=None, help="Items per cell, for a smoke run."
    )
    ap.add_argument("--temperatures", type=float, nargs="+", default=TEMPERATURES)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument(
        "--max_workers",
        type=int,
        default=100,
        help="In-flight requests. The shared config sets 20, which "
        "is sized for the local reasoning endpoints; this "
        "sweep is 6000 short calls against a hosted model.",
    )
    args = ap.parse_args()

    global MODEL, OUT_DIR
    MODEL = args.model
    OUT_DIR = ROOT / results_dir("S10/temperature") / SLUG_OF[MODEL]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(CFG))
    cfg["model_name"] = MODEL
    cfg["max_workers"] = args.max_workers
    cfg["max_tokens"] = MAX_TOKENS
    handler = LLMHandler(cfg)
    print(f"{MODEL}: {args.max_workers} concurrent requests, max_tokens={MAX_TOKENS}")

    for ds in args.datasets:
        scheme = get_scheme(ds)
        samples = _runner.select_samples(ds, N_PER_CLASS, 0)
        if args.limit:
            samples = samples[: args.limit]
        prompts = [
            build_judge_s2_prompt(scheme, s.question, s.context) for s in samples
        ]
        print(f"\n===== {MODEL} :: {ds} =====")
        print(
            f"  {len(samples)} samples "
            f"({sum(1 for s in samples if s.answer_idx == 0)} true / "
            f"{sum(1 for s in samples if s.answer_idx == 1)} false)"
        )
        for temp in args.temperatures:
            raw_s2 = await query_at_temp(handler, prompts, temp)
            summary = _runner.summarize(
                ds,
                samples,
                None,
                raw_s2,
                MODEL,
                ran_s1=False,
                run_config={
                    "max_new_tokens": MAX_TOKENS,
                    "n_per_class": N_PER_CLASS,
                    "temperature": temp,
                    "settings": ["S2"],
                    "class_offset": 0,
                    "endpoint": "api",
                    "use_chat_template": True,
                },
            )
            tag = f"_T{temp}".replace(".", "p")
            path = OUT_DIR / f"{ds}_{MODEL}{tag}.json"
            summary = {**stamp("S10/temperature"), **summary}
            path.write_text(json.dumps(summary, indent=2))
            m = summary["metrics"]["S2"]
            err = sum(1 for r in raw_s2 if r.startswith("__API_ERROR__"))
            print(
                f"  T={temp}: Acc={m['label_acc']:.1%} "
                f"AbsRate={m['abs_rate']:.1%} "
                f"(strict {m['abs_rate_strict']:.1%}, "
                f"trusted {m['trusted_share']:.1%})"
                f"{f'  API errors={err}' if err else ''}  -> {path.name}"
            )


if __name__ == "__main__":
    asyncio.run(main())

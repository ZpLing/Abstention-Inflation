"""S10 modulators (a) and (c) — local checkpoints.

One script because both sweeps run a local HF checkpoint over the same S1/S2
prompts; what differs is which knob moves.

  (a) temperature  one --temperature per call, the Olmo-3-7B-Instruct half
                   of the sweep the API script covers for gateway models
  (c) size x alignment  four Gemma-4 scales in base and IT form at T=0;
                   --use_chat_template is what separates the two arms

The reported cells used --n_per_class 250 (500 items); the default of 100
would give 200.

A re-run of the sweep behind the size-by-alignment figure. The original numbers went through
`infra/paired_pass.py`, whose parser has no guard against a base model echoing the
prompt template: for S2 that template literally contains the word "Unknown", so
a non-answer scores as an abstention, and with the option absent (S1) the same
text scores UNPARSEABLE instead. The bias is therefore one-directional, into
the S2 Abs Rate, and it hits base models hardest — exactly the cells the paper
marked UNRELIABLE.

This runner records, for every item, *where* the label came from, so the Abs
Rate can be reported as a bracket instead of a single unaudited number:

    strict_em   whole response is the verb                  trustworthy
    final_line  the model's `Final answer:` line            trustworthy
    whole_text  fallback scan over the entire generation    not trustworthy
    echo        an unfilled `<...>` slot from the template  not an answer
    empty       nothing generated

`abs_rate_reported` counts every UNKNOWN (upper bound, comparable to the old
pipeline); `abs_rate_strict` counts only explicit commitments (lower bound).

Sampling reproduces `ABRunner._apply_sample_limit` exactly: the first N items
of each gold class in dataset order, no RNG.

    python run_S10_local_sweep.py --model_path .../gemma-4-E2B --model_tag gemma-4-E2B
    python run_S10_local_sweep.py --model_path .../gemma-4-E2B-it \
        --model_tag gemma-4-E2B-it --use_chat_template"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from infra.evaluator import Evaluator  # noqa: E402
from infra.label_scheme import get_scheme  # noqa: E402
from infra.prompts import (  # noqa: E402
    build_judge_s1_prompt,
    build_judge_s2_prompt,
)
from infra.result_schema import (  # noqa: E402
    TRUSTED_TIERS,
    model_slug,
    results_dir,
    stamp,
)
from loader.dataset_loader import load_judge  # noqa: E402

EV = Evaluator()
DATASETS = ["FLD", "FOLIO"]

# Unfilled `<...>` slots from the prompt's own format hint. The model was meant
# to REPLACE these; reproducing them verbatim means it echoed the template.
_ECHO_MARKERS = ("<your step-by-step", "<one of", "<letter>")
_PLACEHOLDER_RE = re.compile(r"<[^>]*>")


#: Longest tail still readable as a single token welded to the answer verb.
_RUNON_MAX_TAIL = 20

#: ...and the tail has to be plain letters. The artifact appends an ordinary
#: word fragment -- `string`, `out`, `thought`, `api` account for 324 of the
#: 366 occurrences. Tails carrying digits, punctuation or a script switch are a
#: different animal: `TrueAPI_CALL:`, `Unknownsize=100%`, `Falseغا`, `Unknown追求`
#: come from cells where the generation has degenerated into tag and config
#: text, and there the leading verb is as likely to be part of the degeneration
#: as a commitment the model meant. 38 rows, so excluding them costs almost
#: nothing and removes the one class of repair that cannot be defended.
_RUNON_TAIL_RE = re.compile(r"[A-Za-z]+")


def _final_answer_line(raw: str) -> str:
    """The model's FIRST `Final answer:` line that is not a template placeholder.

    ``Evaluator.extract_final_answer_line`` returns the *first* match, which is
    wrong for a model that thinks out loud. Qwen3.5 restates the required format
    inside its reasoning --

        Final answer: <one of True | False | Uncertain>
        ... </think>
        Final answer: True

    -- so the first match is the unfilled template and the real commitment
    comes after it. Scoring the first cost 139 of 500 FOLIO items at T=0, every
    one of which ends in a genuine answer.

    Only placeholders are skipped, not every earlier line. Taking the *last*
    match instead would also flip 113 gemma items that legitimately state more
    than one `Final answer:`, where the first is the model's actual commitment
    and the old parser was right to take it.

    Falls back to the first match when every line is a placeholder, so a reply
    that only ever echoes the template still reaches the echo guard.
    """
    lines = [
        EV._FINAL_ANSWER_PREFIX_STRIP_RE.sub("", m.group(0)).strip()
        for m in EV._FINAL_ANSWER_RE.finditer(raw or "")
    ]
    if not lines:
        return ""
    for line in lines:
        if not _PLACEHOLDER_RE.search(line):
            return line
    return lines[0]


def _runon_verb(final_line: str, scheme):
    """POS/NEG/ABSTAIN when the answer verb ran straight into stray text.

    The 26B-A4B-it FOLIO run emitted lines like ``Final answer: Trueout`` --
    an explicit commitment whose verb lost its trailing word boundary, so the
    scheme's ``\bTRUE\b`` no longer matched and the item scored UNPARSEABLE
    even though the model had answered. 60 of that cell's 62 unparseable items
    are this, and dropping them cost it 11.8 points of accuracy.

    Three conditions keep this from inventing labels. The verb must sit at the
    very start of the final-answer line, so prose that merely mentions "true"
    cannot match; it must run directly into another word character, so a verb
    that is already well-formed never reaches here (``scheme.parse`` has had
    its chance by then and this only ever fires on its failures); and what
    follows must look like *one glued word* -- plain letters only
    (:data:`_RUNON_TAIL_RE`), at most :data:`_RUNON_MAX_TAIL` of them.

    That last condition is the one that matters. Without it the repair also
    fired on lines where the newline before the model's *next section* was the
    thing that went missing -- ``FalseWait, I must re-read the hypothesis
    carefully.``, ``UnknownReasoning: The hypothesis ... depends on sent11``,
    ``Truestep-by-step reasoning:``. Those are not commitments with a stray
    token appended; the model kept going, and in the "Wait" case went on to
    reconsider. Reading the leading verb as the final answer there is exactly
    the mislabelling this guard exists to prevent.
    """
    for verb, canon in (
        (scheme.pos_verb, "POS"),
        (scheme.neg_verb, "NEG"),
        (scheme.abstain_verb, "ABSTAIN"),
    ):
        m = re.match(
            rf"\s*{re.escape(verb)}(?=\w)(.*)$", final_line, re.IGNORECASE | re.DOTALL
        )
        if not m:
            continue
        tail = m.group(1)
        if len(tail) <= _RUNON_MAX_TAIL and _RUNON_TAIL_RE.fullmatch(tail):
            return canon
        return None  # verb matched but the tail is not one glued word
    return None


def _classify(raw: str, scheme, with_unknown: bool):
    """(pred, provenance) — the label plus which tier actually produced it."""
    if not isinstance(raw, str) or not raw.strip():
        return "UNPARSEABLE", "empty"

    norm = EV._strict_normalize(raw)
    if norm == scheme.pos_verb.upper():
        return "A", "strict_em"
    if norm == scheme.neg_verb.upper():
        return "B", "strict_em"
    if norm == scheme.abstain_verb.upper():
        return ("UNKNOWN" if with_unknown else "UNPARSEABLE"), "strict_em"

    # An explicit `Final answer:` commitment outranks the echo guard: a reply
    # that quotes the format hint and then answers is a real answer.
    final_line = _final_answer_line(raw)
    if final_line and not _PLACEHOLDER_RE.search(final_line):
        canonical = scheme.parse(final_line)
        if canonical == "POS":
            return "A", "final_line"
        if canonical == "NEG":
            return "B", "final_line"
        if canonical == "ABSTAIN":
            return ("UNKNOWN" if with_unknown else "UNPARSEABLE"), "final_line"
        # Same explicit commitment, verb welded to trailing junk. Kept on its
        # own tier so the repair stays visible in `provenance`, but it is a
        # `Final answer:` line either way and counts as trustworthy.
        runon = _runon_verb(final_line, scheme)
        if runon == "POS":
            return "A", "final_line_runon"
        if runon == "NEG":
            return "B", "final_line_runon"
        if runon == "ABSTAIN":
            return ("UNKNOWN" if with_unknown else "UNPARSEABLE"), "final_line_runon"
    elif final_line:
        return "UNPARSEABLE", "echo"

    if any(m in raw for m in _ECHO_MARKERS):
        return "UNPARSEABLE", "echo"

    # No explicit commitment. Report what the old pipeline would have said, but
    # label it so it can be excluded from the strict estimate.
    canonical = scheme.parse(raw)
    if canonical == "POS":
        return "A", "whole_text"
    if canonical == "NEG":
        return "B", "whole_text"
    if canonical == "ABSTAIN":
        return ("UNKNOWN" if with_unknown else "UNPARSEABLE"), "whole_text"
    return "UNPARSEABLE", "none"


def select_samples(dataset: str, n_per_class: int, class_offset: int = 0):
    """`n_per_class` items of each gold class in dataset order, starting at
    `class_offset` within the class.

    With offset 0 this is exactly the selection `ABRunner._apply_sample_limit`
    makes for {true: N, false: N}, which is what the original gemma configs
    used ({true: 100, false: 100} = 200 items). The offset exists so the rest
    of the dataset can be run as a second pass and merged, rather than
    re-running the 200 already done: the paper reports n=500 per dataset, but
    those configs only ever covered 200 of them.
    """
    samples = load_judge(dataset)
    out = []
    for target in (0, 1):
        cls = [s for s in samples if s.answer_idx == target]
        out.extend(cls[class_offset : class_offset + n_per_class])
    return out


def messages_to_text(messages, tokenizer, use_chat_template: bool):
    if use_chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    # Base model: no chat template. Feed the user content and nudge with the
    # "Reasoning:" prefix the format hint already asks for.
    return messages[0]["content"] + "\n\nReasoning:"


@torch.no_grad()
def batch_generate(
    model,
    tokenizer,
    prompts,
    batch_size,
    max_new_tokens,
    temperature=None,
    top_p=None,
    top_k=None,
):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    out, t0, n = [], time.time(), len(prompts)
    for i in range(0, n, batch_size):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=4096
        ).to(model.device)
        kw = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        if temperature is None or temperature == 0.0:
            # Greedy. T=0 and "no sampling" are the same decode, and asking for
            # do_sample=True at T=0 is a division by zero in most kernels.
            gen = model.generate(
                **enc, do_sample=False, temperature=None, top_p=None, top_k=None, **kw
            )
        else:
            # Only pass top_p when asked for; otherwise leave the checkpoint's
            # own generation defaults untouched.
            if top_p is not None:
                kw["top_p"] = top_p
            # top_k=0 disables head truncation entirely, which is what the first
            # OLMo sweep ran at: nucleus alone did not hold the model together
            # above T=1, where only 13-20% of its words still came from the
            # vocabulary it uses at T=0. Passing a real k restores the kind of
            # truncation a served endpoint applies on the caller's behalf.
            gen = model.generate(
                **enc,
                do_sample=True,
                temperature=temperature,
                top_k=(0 if top_k is None else top_k),
                **kw,
            )
        out.extend(
            tokenizer.batch_decode(
                gen[:, enc.input_ids.shape[1] :], skip_special_tokens=True
            )
        )
        done = min(i + batch_size, n)
        el = time.time() - t0
        print(
            f"    {done}/{n} in {el:.0f}s  ETA {(n - done) / (done / el):.0f}s",
            flush=True,
        )
    return out


def run_setting(model, tokenizer, scheme, samples, setting, args):
    builder = build_judge_s1_prompt if setting == "S1" else build_judge_s2_prompt
    prompts = [
        messages_to_text(
            builder(scheme, s.question, s.context), tokenizer, args.use_chat_template
        )
        for s in samples
    ]
    print(f"  [{setting}] generating on {len(prompts)} prompts ...", flush=True)
    gens = batch_generate(
        model,
        tokenizer,
        prompts,
        args.batch_size,
        args.max_new_tokens,
        getattr(args, "temperature", None),
        getattr(args, "top_p", None),
        getattr(args, "top_k", None),
    )
    # The base path appends "Reasoning:" to the prompt, so the continuation
    # starts mid-reasoning; restore the prefix the parser expects.
    return [g if args.use_chat_template else "Reasoning:" + g for g in gens]


def summarize(
    ds, samples, raw_s1, raw_s2, model_tag, run_config=None, ran_s1: bool = True
):
    scheme = get_scheme(ds)
    # A setting that was not run has no accuracy. Scoring empty strings would
    # report label_acc=0.0, which reads downstream as "the model got everything
    # wrong" rather than "this was never measured".
    p1 = [_classify(r, scheme, False) for r in raw_s1] if ran_s1 else None
    p2 = [_classify(r, scheme, True) for r in raw_s2]
    n = len(samples)

    def acc(pairs):
        return (
            sum(
                1
                for (p, _), s in zip(pairs, samples)
                if (s.answer_idx == 0 and p == "A") or (s.answer_idx == 1 and p == "B")
            )
            / n
            if n
            else 0.0
        )

    n_unknown = sum(1 for p, _ in p2 if p == "UNKNOWN")
    n_strict = sum(1 for p, pv in p2 if p == "UNKNOWN" and pv in TRUSTED_TIERS)
    trusted = sum(1 for _, pv in p2 if pv in TRUSTED_TIERS)
    return {
        "schema": "paper-s1-s10/v1",
        "dataset": ds,
        "task_type": "tf",
        "model": model_tag,
        "trace_family": "hard" if ds == "FLD" else "soft",
        "n_total": n,
        "n_abstention_inflation": n_unknown,
        # Recorded so a later merge can refuse to splice passes that were not
        # generated the same way. Generation budget in particular changes Abs
        # Rate a lot: at 1024 the IT models hit the cap on most FLD items and
        # never emitted a `Final answer:` line at all.
        "run_config": run_config or {},
        "metrics": {
            "S1": (
                {"label_acc": acc(p1)}
                if p1 is not None
                else {"label_acc": None, "not_run": True}
            ),
            "S2": {
                "label_acc": acc(p2),
                "abs_rate": n_unknown / n if n else 0.0,
                "abs_rate_strict": n_strict / n if n else 0.0,
                "trusted_share": trusted / n if n else 0.0,
            },
        },
        "provenance": {
            "s1": (
                dict(Counter(pv for _, pv in p1)) if p1 is not None else {"not_run": n}
            ),
            "s2": dict(Counter(pv for _, pv in p2)),
        },
        "per_sample": [
            {
                "id": samples[i].id,
                "source": ds,
                "answer_idx": samples[i].answer_idx,
                **(
                    {"pred_s1": p1[i][0], "prov_s1": p1[i][1], "raw_s1": raw_s1[i]}
                    if p1 is not None
                    else {"pred_s1": None, "prov_s1": "not_run"}
                ),
                "pred_s2": p2[i][0],
                "prov_s2": p2[i][1],
                "raw_s2": raw_s2[i],
            }
            for i in range(n)
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_tag", required=True)
    ap.add_argument(
        "--use_chat_template",
        action="store_true",
        help="Set for IT models; base models have no chat template.",
    )
    ap.add_argument("--n_per_class", type=int, default=100)
    ap.add_argument(
        "--class_offset",
        type=int,
        default=0,
        help="Skip this many items of each gold class first; lets a "
        "second pass cover the rest of the dataset.",
    )
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Nucleus cutoff for sampled decoding. Unset means the "
        "parameter is not passed at all, leaving whatever the "
        "checkpoint's generation_config specifies (HuggingFace's "
        "own default is 1.0 when it specifies nothing). Note "
        "that pure temperature degenerates at the top of the "
        "range: a Qwen3.5-9B sweep at top_p=1.0 returned 72%% "
        "unparseable multilingual noise at T=2, which measures "
        "decoding collapse rather than abstention.",
    )
    ap.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Keep only the k highest-probability tokens before "
        "sampling. Unset leaves the generate() default of 0, "
        "i.e. no head truncation at all -- which is how the "
        "first OLMo sweep ran, and why its T>=1.5 cells "
        "decoded into token soup rather than answers. A "
        "served endpoint normally applies some k on the "
        "caller's behalf, so setting it makes a local sweep "
        "comparable to an API one; it also means the sweep "
        "measures temperature under fixed truncation rather "
        "than temperature alone.",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sample at this temperature instead of decoding "
        "greedily. S10(c) leaves it unset; S10(a) sweeps it.",
    )
    ap.add_argument(
        "--settings",
        nargs="+",
        default=["S1", "S2"],
        help="S10(a) only needs S2 -- S1 has no abstain option to "
        "measure and would double the generation cost.",
    )
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--out_dir", default=str(ROOT / results_dir("S10/size_alignment")))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {args.model_path} ...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    # `hf_device_map` only exists when accelerate actually sharded the model;
    # a model that fits on one GPU has no such attribute.
    placement = getattr(model, "hf_device_map", None)
    placement = set(placement.values()) if placement else {str(model.device)}
    print(
        f"  loaded in {time.time() - t0:.0f}s  type={type(model).__name__}  "
        f"devices={placement}",
        flush=True,
    )

    for ds in args.datasets:
        print(f"\n===== {args.model_tag} :: {ds} =====", flush=True)
        scheme = get_scheme(ds)
        samples = select_samples(ds, args.n_per_class, args.class_offset)
        print(
            f"  {len(samples)} samples (class offset {args.class_offset}) "
            f"({sum(1 for s in samples if s.answer_idx == 0)} true / "
            f"{sum(1 for s in samples if s.answer_idx == 1)} false)",
            flush=True,
        )
        raw_s1 = (
            run_setting(model, tokenizer, scheme, samples, "S1", args)
            if "S1" in args.settings
            else [""] * len(samples)
        )
        raw_s2 = run_setting(model, tokenizer, scheme, samples, "S2", args)
        summary = summarize(
            ds,
            samples,
            raw_s1,
            raw_s2,
            args.model_tag,
            ran_s1="S1" in args.settings,
            run_config={
                "max_new_tokens": args.max_new_tokens,
                "n_per_class": args.n_per_class,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "settings": list(args.settings),
                "class_offset": args.class_offset,
                "batch_size": args.batch_size,
                "use_chat_template": bool(args.use_chat_template),
            },
        )
        suffix = (
            ""
            if args.temperature is None
            else f"_T{args.temperature}".replace(".", "p")
        )
        path = (
            out_dir
            if args.temperature is not None
            else out_dir / model_slug(args.model_tag)
        ) / f"{ds}_{args.model_tag}{suffix}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            **stamp(
                "S10/size_alignment" if args.temperature is None else "S10/temperature"
            ),
            **summary,
        }
        path.write_text(json.dumps(summary, indent=2))
        m = summary["metrics"]
        a1 = m["S1"]["label_acc"]
        print(
            f"  -> Acc_S1={'(not run)' if a1 is None else format(a1, '.1%')}  "
            f"Acc_S2={m['S2']['label_acc']:.1%}  "
            f"AbsRate={m['S2']['abs_rate']:.1%} (strict {m['S2']['abs_rate_strict']:.1%}, "
            f"trusted {m['S2']['trusted_share']:.1%})  -> {path.name}",
            flush=True,
        )
        print(f"     provenance S2: {summary['provenance']['s2']}", flush=True)


if __name__ == "__main__":
    main()

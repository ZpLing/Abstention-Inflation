"""S10 audit — is the base-model Abs Rate real, or a parser artifact?

Post-hoc, zero API calls. Reads any `ab_summary_<ds>_<model>.json` that
carries `per_sample[].raw_s1 / raw_s2` and re-scores the abstentions by
*provenance*, i.e. which tier of `Evaluator.parse_judge_tiered` actually
produced the UNKNOWN.

Why this matters
----------------
`parse_judge_tiered` falls back to `LabelScheme.parse(whole_text)` when the
model does not emit a `Final answer:` line. That fallback checks
`abstain_patterns` FIRST and over the *entire* generation. For an
instruction-tuned model that is fine. For a **base** model it is not:

  * base models routinely echo the prompt template, which for S2 ends with
    `Final answer: <one of True | False | Unknown>` — the literal token
    `Unknown` — so a pure non-response scores as an abstention;
  * off-task rambling that happens to contain `insufficient`, `cannot be
    determined`, or a negated `not true` also lands in ABSTAIN via the
    negation guard.

The bias is *asymmetric*: with `with_unknown=False` (S1) the same garbage
returns UNPARSEABLE, with `with_unknown=True` (S2) it returns UNKNOWN. So
noise flows into the S2 Abs Rate only, and the reported base-vs-IT gap is
inflated by exactly that flow.

`run_S10_local_hf_sweep.py` guards against the echo case
(`_TEMPLATE_ECHO_MARKERS`); the YAML/vLLM path (`experiments/C1_structural_trigger/ab_runner.py`), which
produced the Fig. 9 base cells, does not.

Policy compared here
--------------------
  reported : whatever the run wrote out (`pred_s1` / `pred_s2`).
  strict   : an abstention counts ONLY if the model explicitly committed to
             it — strict EM on the abstain verb, or the `Final answer:` line
             resolves to ABSTAIN. Whole-text-fallback abstentions and prompt
             echoes become UNPARSEABLE, and the same rule is applied to S1 so
             the two conditions stay symmetric.

The two columns BRACKET the truth rather than replacing one number with
another: `abs_rate_reported` is an upper bound (every incidental "Unknown" counts),
`abs_rate_strict` is a lower bound (an abstention the model meant but phrased
without a `Final answer:` line is dropped). A small gap means the reported Abs
Rate is trustworthy; a gap the size of the metric itself means it is not.

Usage
-----
    python experiments/C4_stable_bias/S10_factor_analysis/audit_S10_base_abstention.py \
        results/gemma_it_vs_base results/gemma_it_s52
    # or a single file; `--json out.json` to dump the machine-readable report
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from infra.evaluator import Evaluator          # noqa: E402
from infra.label_scheme import get_scheme      # noqa: E402
from infra.result_schema import load_summary   # noqa: E402

EV = Evaluator()

# Unfilled placeholders from the prompt's own format hint. These are angle-
# bracket slots the model was supposed to REPLACE; reproducing them verbatim
# means it echoed the template instead of answering it. Only placeholders are
# listed — plain prompt sentences ("Format your response exactly as") are NOT,
# because a model may legitimately quote them on its way to a real answer.
_ECHO_MARKERS = (
    "<your step-by-step",
    "<one of",
    "<letter>",
)

_PLACEHOLDER_RE = re.compile(r"<[^>]*>")


def _is_echo(text: str) -> bool:
    return isinstance(text, str) and any(m in text for m in _ECHO_MARKERS)


def _is_placeholder(line: str) -> bool:
    """True if a `Final answer:` line is still an unfilled `<...>` slot."""
    return bool(_PLACEHOLDER_RE.search(line))


def classify(raw: str, scheme, with_unknown: bool):
    """Return (strict_pred, provenance).

    provenance ∈ {strict_em, final_line, whole_text, echo, empty, none}
    and names *where* the label came from, so whole-text-only abstentions can
    be separated from explicit ones.
    """
    if not isinstance(raw, str) or not raw.strip():
        return "UNPARSEABLE", "empty"

    norm = EV._strict_normalize(raw)
    if norm == scheme.pos_verb.upper():
        return "A", "strict_em"
    if norm == scheme.neg_verb.upper():
        return "B", "strict_em"
    if norm == scheme.abstain_verb.upper():
        return ("UNKNOWN" if with_unknown else "UNPARSEABLE"), "strict_em"

    # An explicit `Final answer:` commitment is authoritative and is checked
    # BEFORE the echo guard: a response that quotes the format hint and then
    # actually answers is a real answer, not an echo. Only an unfilled `<...>`
    # slot on the final-answer line counts as an echo here.
    final_line = EV.extract_final_answer_line(raw)
    if final_line and not _is_placeholder(final_line):
        canonical = scheme.parse(final_line)
        if canonical == "POS":
            return "A", "final_line"
        if canonical == "NEG":
            return "B", "final_line"
        if canonical == "ABSTAIN":
            return ("UNKNOWN" if with_unknown else "UNPARSEABLE"), "final_line"
    elif final_line:
        return "UNPARSEABLE", "echo"

    # No explicit commitment left. Template echo now outranks the whole-text
    # scan, since neither is an answer and `echo` is the more informative label.
    if _is_echo(raw):
        return "UNPARSEABLE", "echo"

    # The reported run would scan the whole text here; under the strict policy
    # a non-committal generation is a non-response, not an abstention.
    canonical = scheme.parse(raw)
    if canonical == "POS":
        return "A", "whole_text"
    if canonical == "NEG":
        return "B", "whole_text"
    if canonical == "ABSTAIN":
        return "UNPARSEABLE", "whole_text"
    return "UNPARSEABLE", "none"


def audit_file(path: Path):
    s = load_summary(path)
    rows = s.get("per_sample") or []
    if not rows or "raw_s2" not in rows[0]:
        return None
    scheme = get_scheme(s["dataset"])
    n = len(rows)

    rep = {"s1": Counter(), "s2": Counter()}
    strict = {"s1": Counter(), "s2": Counter()}
    prov = {"s1": Counter(), "s2": Counter()}
    # Provenance of the abstentions the run actually reported.
    abs_prov = Counter()

    for r in rows:
        for key, with_unknown in (("s1", False), ("s2", True)):
            raw = r.get(f"raw_{key}")
            reported = r.get(f"pred_{key}")
            sp, pv = classify(raw, scheme, with_unknown)
            rep[key][reported] += 1
            strict[key][sp] += 1
            prov[key][pv] += 1
            if key == "s2" and reported == "UNKNOWN":
                abs_prov[pv] += 1

    abs_rate_rep = rep["s2"]["UNKNOWN"] / n if n else 0.0
    abs_rate_strict = strict["s2"]["UNKNOWN"] / n if n else 0.0
    return {
        "file": str(path),
        "dataset": s["dataset"],
        "model": s.get("model"),
        "n_total": n,
        "abs_rate_reported": abs_rate_rep,
        "abs_rate_strict": abs_rate_strict,
        "abs_rate_delta": abs_rate_rep - abs_rate_strict,
        "reported_abstention_provenance": dict(abs_prov),
        "reported": {k: dict(v) for k, v in rep.items()},
        "strict": {k: dict(v) for k, v in strict.items()},
        "provenance": {k: dict(v) for k, v in prov.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+",
                    help="ab_summary_*.json files, or directories holding them")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    files = []
    for p in map(Path, args.paths):
        if p.is_dir():
            files.extend(sorted(p.glob("ab_summary_*.json")))
        elif p.is_file():
            files.append(p)
    if not files:
        print("no ab_summary_*.json found under the given paths")
        return

    reports = []
    for f in files:
        try:
            rep = audit_file(f)
        except Exception as exc:                      # noqa: BLE001
            print(f"  [skip] {f.name}: {exc}")
            continue
        if rep is None:
            print(f"  [skip] {f.name}: no per-sample raw outputs")
            continue
        reports.append(rep)

    hdr = (f"{'model':<26} {'ds':<7} {'n':>5} {'Abs Rate rep':>9} {'Abs Rate strict':>11} "
           f"{'delta':>8}   abstentions came from")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(reports, key=lambda x: (x["model"] or "", x["dataset"])):
        src = ", ".join(f"{k}={v}" for k, v in
                        sorted(r["reported_abstention_provenance"].items(),
                               key=lambda kv: -kv[1])) or "-"
        print(f"{(r['model'] or '?'):<26} {r['dataset']:<7} {r['n_total']:>5} "
              f"{r['abs_rate_reported']:>8.1%} {r['abs_rate_strict']:>10.1%} "
              f"{r['abs_rate_delta']:>+8.1%}   {src}")

    print("\nabstentions traced to `whole_text` or `echo` are parser artifacts, "
          "not model abstentions.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(reports, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()

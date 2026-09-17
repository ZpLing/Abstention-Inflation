"""Parser-provenance audit — how much of the reported Abs Rate is real?

Post-hoc, zero API calls. Re-scores every `ab_summary_*.json` that carries
per-sample raw outputs and reports *where* each label came from, because the
tiers in `core/evaluator.py` are not equally trustworthy:

  strict_em   whole response is the label                       — trustworthy
  final_line  the model's `Final answer:` line                  — trustworthy
  whole_text  fallback scan over the ENTIRE generation          — NOT trustworthy
  echo        the prompt's own format hint, reproduced verbatim — not an answer
  empty       nothing generated

`Evaluator` collapses the last three into `lenient_em` / `unparseable`, so the
`tier_counts` already stored in a summary cannot distinguish them. This script
can, and the split matters because the whole-text fallback is biased in one
direction per task format:

  TFQ  `LabelScheme.parse` checks abstain patterns FIRST over the whole text,
       so a stray "Unknown" / "insufficient" / "cannot be determined" — including
       the literal `Final answer: <one of True | False | Unknown>` hint echoed
       back — scores as an abstention. With the option absent (S1) the same text
       is UNPARSEABLE instead, so noise flows into the S2 Abs Rate only.

  MCQ  `_parse_letter` ends with `\\b([ABCDE])\\b` over the upper-cased text, so
       the English article "a" reads as option A, and any mention of "E" reads
       as the abstain option.

Reported vs strict
------------------
  reported : the run's own `pred_s1` / `pred_s2`.
  strict   : only strict_em and final_line count; whole_text / echo / empty
             become UNPARSEABLE, symmetrically in S1 and S2.

The two BRACKET the truth: reported Abs Rate is an upper bound, strict is a
lower bound. A small gap means the number is safe to report; a gap the size of
the metric itself means it is not.

    python experiments/appendix/audit_parser_provenance.py results/
    python experiments/appendix/audit_parser_provenance.py results/ --json out.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.evaluator import Evaluator          # noqa: E402
from core.label_scheme import get_scheme      # noqa: E402
from core.result_schema import load_summary   # noqa: E402

EV = Evaluator()

# Unfilled `<...>` slots from the prompt's own format hint: the model was meant
# to REPLACE these, so reproducing them verbatim means it echoed the template.
_ECHO_MARKERS = ("<your step-by-step", "<one of", "<letter>")
_PLACEHOLDER_RE = re.compile(r"<[^>]*>")


def _is_echo(text: str) -> bool:
    return any(m in text for m in _ECHO_MARKERS)


def _is_placeholder(line: str) -> bool:
    return bool(_PLACEHOLDER_RE.search(line))


def classify_tfq(raw, scheme, with_unknown):
    norm = EV._strict_normalize(raw)
    if norm == scheme.pos_verb.upper():
        return "A", "strict_em"
    if norm == scheme.neg_verb.upper():
        return "B", "strict_em"
    if norm == scheme.abstain_verb.upper():
        return ("UNKNOWN" if with_unknown else "UNPARSEABLE"), "strict_em"

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

    if _is_echo(raw):
        return "UNPARSEABLE", "echo"
    return "UNPARSEABLE", ("whole_text" if scheme.parse(raw) != "UNPARSEABLE" else "none")


def classify_mcq(raw, with_unknown):
    valid = "ABCDE" if with_unknown else "ABCD"
    norm = EV._strict_normalize(raw)
    if norm in {"A", "B", "C", "D"}:
        return norm, "strict_em"
    if with_unknown and norm in {"E", "UNKNOWN"}:
        return "UNKNOWN", "strict_em"

    final_line = EV.extract_final_answer_line(raw)
    if final_line and not _is_placeholder(final_line):
        letter = EV._parse_letter(final_line, valid)
        if letter is not None:
            if with_unknown and letter == "E":
                return "UNKNOWN", "final_line"
            return letter, "final_line"
        if with_unknown and re.search(r"\bunknown\b", final_line, re.IGNORECASE):
            return "UNKNOWN", "final_line"
    elif final_line:
        return "UNPARSEABLE", "echo"

    if _is_echo(raw):
        return "UNPARSEABLE", "echo"
    return "UNPARSEABLE", ("whole_text"
                           if EV._parse_letter(raw, valid) is not None else "none")


def classify(raw, scheme, with_unknown):
    if not isinstance(raw, str) or not raw.strip():
        return "UNPARSEABLE", "empty"
    return (classify_tfq(raw, scheme, with_unknown) if scheme is not None
            else classify_mcq(raw, with_unknown))


def audit_file(path: Path):
    s = load_summary(path)
    rows = s.get("per_sample") or []
    if not rows or "raw_s2" not in rows[0]:
        return None
    task = s.get("task_type") or ("tf" if "raw_s1" in rows[0] else "mcq")
    try:
        scheme = get_scheme(s["dataset"]) if task == "tf" else None
    except KeyError:
        scheme = None
        task = "mcq"

    # answer_idx == -1 marks a truly-Unknown item; every other setting filters
    # to answer_idx >= 0, so exclude them here too.
    rows = [r for r in rows if r.get("answer_idx", 0) >= 0]
    n = len(rows)
    if not n:
        return None

    rep, strict, prov = Counter(), Counter(), Counter()
    abs_prov = Counter()
    for r in rows:
        raw = r.get("raw_s2")
        reported = r.get("pred_s2")
        sp, pv = classify(raw, scheme, True)
        rep[reported] += 1
        strict[sp] += 1
        prov[pv] += 1
        if reported == "UNKNOWN":
            abs_prov[pv] += 1

    abs_rep = rep["UNKNOWN"] / n
    abs_strict = strict["UNKNOWN"] / n
    trusted = (prov["strict_em"] + prov["final_line"]) / n
    return {
        "file": str(path), "dataset": s["dataset"], "model": s.get("model"),
        "task_type": task, "n_total": n,
        "abs_rate_reported": abs_rep,
        "abs_rate_strict": abs_strict,
        "abs_rate_delta": abs_rep - abs_strict,
        "trusted_share": trusted,
        "abstention_provenance": dict(abs_prov),
        "provenance": dict(prov),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    files = []
    for p in map(Path, args.paths):
        files.extend(sorted(p.rglob("ab_summary_*.json")) if p.is_dir() else [p])

    reports = []
    for f in files:
        try:
            r = audit_file(f)
        except Exception as exc:                                  # noqa: BLE001
            print(f"  [skip] {f.name}: {exc}")
            continue
        if r:
            reports.append(r)

    hdr = (f"{'model':<30} {'dataset':<18} {'fmt':<4} {'n':>5} "
           f"{'AbsR rep':>9} {'AbsR strict':>12} {'delta':>8} {'trusted':>8}")
    print(hdr); print("-" * len(hdr))
    for r in sorted(reports, key=lambda x: -abs(x["abs_rate_delta"])):
        print(f"{str(r['model'])[:30]:<30} {r['dataset'][:18]:<18} {r['task_type']:<4} "
              f"{r['n_total']:>5} {r['abs_rate_reported']:>8.1%} "
              f"{r['abs_rate_strict']:>11.1%} {r['abs_rate_delta']:>+8.1%} "
              f"{r['trusted_share']:>7.1%}")

    if reports:
        worst = max(reports, key=lambda x: x["abs_rate_delta"])
        print(f"\ncells audited: {len(reports)}   "
              f"max Abs Rate inflation: {worst['abs_rate_delta']:+.1%} "
              f"({worst['model']} / {worst['dataset']})")
        clean = sum(1 for r in reports if abs(r["abs_rate_delta"]) < 0.01)
        print(f"cells where reported and strict agree within 1pp: {clean}/{len(reports)}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(reports, indent=2))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()

"""Build Table 1's MCQ columns (ARC / MedQA / MMLU / LogiQA) at n=500.

Each cell pools every ``ab_summary`` that covers items of ``dataset/<X>.json``
for that model, keyed by sample id, and reports the paired S1/S2 contrast:

    Acc (S1)      fraction answered correctly without an "Unknown" option
    Acc (S2)      same items with "E. Unknown" appended
    Abs Rate (S2) fraction of S2 replies that selected "Unknown"

Scoring follows the paper: an abstention counts as incorrect. A turn that
carries no answer at all is a different thing and is not scored — a request
that never came back, a gateway refusal, decoding collapse, or reasoning the
output cap cut off before it committed. ``Evaluator.classify_unanswered``
names which of those it was; an item is dropped from the cell unless BOTH
conditions produced an answer, and the drops are reported per cell by reason.

Ids are the join key, so a benchmark whose run used a retired subset file with
a different id->item mapping contributes nothing and is reported as missing
rather than silently pooled.

Rows are labelled with the model id that was actually queried; ``PAPER_ROW``
maps each one to the row name Table 1 gives it. Prints markdown; writes
nothing.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from core.evaluator import Evaluator

#: model id actually queried -> the row name Table 1 gives it.
PAPER_ROW = {
    "deepseek-v4-flash": "DeepSeek-V4-Flash",
    "gpt-5.4-nano": "GPT-5.4-nano",
    "gemini-3.1-flash-lite": "Gemini-3.1-Flash-Lite",
}
MODELS = list(PAPER_ROW)
BENCHMARKS = ["ARC", "MedQA", "MMLU", "LogiQA"]

#: Result dirs that hold runs on the *current* dataset files. Dirs built on the
#: retired ``*_200`` subsets are deliberately absent: their ids address other
#: items, so pooling them would mix two different samples of the benchmark.
SEARCH_DIRS = [
    "results/ab",
    "results/ab_e_option_baseline",
    "results/ab_mcq_full250",
    "results/ab_gpt5_nano",
    "results/ab_nano_batch2",
    "results/ab_nano_medqa250",
    "results/mcq_n500",
]

_ABSTAIN = "UNKNOWN"


def dataset_ids(name: str) -> set:
    items = json.loads((ROOT / "dataset" / f"{name}.json").read_text())
    return {i["id"] for i in items}


def unanswered_reason(rec: dict, setting: str):
    """Why this turn carries no answer, or None when it has one.

    Prefers the reason the runner recorded; recomputes it for the older
    summaries that predate that field.
    """
    recorded = (rec.get("unanswered") or {}).get(setting.upper())
    if recorded:
        return recorded
    if rec.get(f"pred_{setting}") != "UNPARSEABLE":
        return None
    return Evaluator.classify_unanswered(rec.get(f"raw_{setting}"))


def is_correct(pred, answer_idx) -> bool:
    if not pred or len(pred) != 1 or not pred.isalpha():
        return False
    return ord(pred) - ord("A") == answer_idx


def collect(model: str, ids: set) -> dict:
    """Newest-wins pooling of per-sample records whose id is in ``ids``."""
    by_id = {}
    for d in SEARCH_DIRS:
        if not (ROOT / d).is_dir():
            continue
        for f in sorted((ROOT / d).rglob("*summary*.json"),
                        key=lambda p: p.stat().st_mtime):
            try:
                s = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(s, dict) or s.get("model") != model:
                continue
            for r in s.get("per_sample", []):
                if r.get("id") in ids and "pred_s1" in r and "pred_s2" in r:
                    by_id[r["id"]] = r
    return by_id


def main():
    rows = []
    for mid in MODELS:
        for bench in BENCHMARKS:
            ids = dataset_ids(bench)
            recs = collect(mid, ids)
            pooled = len(recs)
            dropped = Counter()
            scored = []
            for r in recs.values():
                reasons = [unanswered_reason(r, s) for s in ("s1", "s2")]
                if any(reasons):
                    for reason in reasons:
                        if reason:
                            dropped[reason] += 1
                    continue
                scored.append(r)
            n = len(scored)
            if n == 0:
                rows.append((mid, bench, None, None, None, 0, pooled, len(ids), dropped))
                continue
            acc1 = 100 * sum(is_correct(r["pred_s1"], r["answer_idx"]) for r in scored) / n
            acc2 = 100 * sum(is_correct(r["pred_s2"], r["answer_idx"]) for r in scored) / n
            abs2 = 100 * sum(r["pred_s2"] == _ABSTAIN for r in scored) / n
            rows.append((mid, bench, acc1, acc2, abs2, n, pooled, len(ids), dropped))

    print("| Model (queried) | Benchmark | Acc (S1) | Acc (S2) | ΔAcc "
          "| Abs Rate (S2) | n scored | pooled/dataset |")
    print("|---|---|--:|--:|--:|--:|--:|--:|")
    for mid, bench, a1, a2, ab, n, pooled, total, _ in rows:
        if a1 is None:
            print(f"| {mid} | {bench} | — | — | — | — | 0 | {pooled}/{total} |")
            continue
        print(f"| {mid} | {bench} | {a1:.1f}% | {a2:.1f}% | {a2 - a1:+.1f} "
              f"| {ab:.1f}% | {n} | {pooled}/{total} |")

    drops = [(m, b, pooled - n, d) for m, b, a1, _, _, n, pooled, _, d in rows
             if a1 is not None and pooled - n]
    if drops:
        print("\nItems dropped (no answer in S1 and/or S2 — not scored as wrong):")
        for m, b, k, d in drops:
            why = ", ".join(f"{r}={c}" for r, c in sorted(d.items()))
            print(f"  {m} x {b}: {k}  ({why})")

    short = [(m, b, pooled, t) for m, b, _, _, _, _, pooled, t, _ in rows if pooled < t]
    if short:
        print("\nCells not yet at the dataset's full size:")
        for m, b, pooled, t in short:
            print(f"  {m} x {b}: {pooled}/{t}")

    print("\nRow names in Table 1: " + ", ".join(
        f"{mid} -> {row}" for mid, row in PAPER_ROW.items()))


if __name__ == "__main__":
    main()

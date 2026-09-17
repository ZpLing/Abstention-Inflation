"""Score the clean-prompt S3 rerun and contrast it with the superseded run.

Reads results/s3_clean_n500/summary_s3_{DS}_{MODEL}.json (written by
run_S3.py) and, when present, the old confounded condition in
results/positional_bias_n500/summary_unknown_C_{DS}_{MODEL}.json, which used the
same letter rendering *plus* the calibration note. The paired contrast on the
shared item ids isolates what the note was doing.

Prints a markdown table with Acc, Abs Rate and the paired McNemar test on the
abstain/not-abstain outcome.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CLEAN = ROOT / "results/s3_clean_n500"
OLD = ROOT / "results/positional_bias_n500"

MODELS = [
    ("DeepSeek-V4-Flash", "deepseek-v4-flash"),
    ("GPT-5.4-nano", "gpt-5.4-nano"),
    ("Gemini-3.1-Flash-Lite", "gemini-3.1-flash-lite"),
]
DATASETS = ["FLD", "FOLIO"]


def rows(path: Path):
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return {r["id"]: r for r in d["per_sample"] if not r.get("excluded")}, d


def correct(r) -> bool:
    return (r["pred"] == "A" and r["answer_idx"] == 0) or \
           (r["pred"] == "B" and r["answer_idx"] == 1)


def mcnemar(b: int, c: int):
    """Two-sided exact-ish McNemar on discordant pairs (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    # exact binomial two-sided
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n) * 2
    return min(1.0, p)


def main() -> None:
    print("| Model | Dataset | n | Acc (clean S3) | Abs Rate (clean S3) | "
          "Abs Rate (old S3 w/ note) | Δ | McNemar p |")
    print("|---|---|--:|--:|--:|--:|--:|--:|")
    macro = {"acc": [], "abs": []}
    for disp, mid in MODELS:
        for ds in DATASETS:
            new = rows(CLEAN / f"summary_s3_{ds}_{mid}.json")
            if new is None:
                print(f"| {disp} | {ds} | — | — | — | — | — | (missing) |")
                continue
            nrows, ndoc = new
            ids = sorted(nrows)
            n = len(ids)
            acc = 100 * sum(correct(nrows[i]) for i in ids) / n
            absr = 100 * sum(nrows[i]["pred"] == "UNKNOWN" for i in ids) / n
            macro["acc"].append(acc)
            macro["abs"].append(absr)

            old = rows(OLD / f"summary_unknown_C_{ds}_{mid}.json")
            if old is None:
                print(f"| {disp} | {ds} | {n} | {acc:.1f} | {absr:.1f} | — | — | — |")
                continue
            orows, _ = old
            shared = [i for i in ids if i in orows]
            oabs = 100 * sum(orows[i]["pred"] == "UNKNOWN" for i in shared) / len(shared)
            nabs = 100 * sum(nrows[i]["pred"] == "UNKNOWN" for i in shared) / len(shared)
            b = sum(1 for i in shared
                    if nrows[i]["pred"] == "UNKNOWN" and orows[i]["pred"] != "UNKNOWN")
            c = sum(1 for i in shared
                    if nrows[i]["pred"] != "UNKNOWN" and orows[i]["pred"] == "UNKNOWN")
            p = mcnemar(b, c)
            print(f"| {disp} | {ds} | {n} | {acc:.1f} | {absr:.1f} | {oabs:.1f} | "
                  f"{nabs - oabs:+.1f} | {p:.3g} |")
    if macro["acc"]:
        print(f"\nMacro over {len(macro['acc'])} cells: "
              f"Acc {sum(macro['acc']) / len(macro['acc']):.2f}, "
              f"Abs Rate {sum(macro['abs']) / len(macro['abs']):.2f}")


if __name__ == "__main__":
    main()

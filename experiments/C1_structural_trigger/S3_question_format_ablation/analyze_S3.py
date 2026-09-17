"""S3 Question Format Ablation — does the letter rendering move Abs Rate?

S3 re-renders the same ternary as A / B / C. If abstention followed the letters
rather than the label set, Abs Rate would move; the paper's claim is that it
barely does, which is what makes the trigger structural rather than a rendering
artifact.

Read from the paired summaries the main table is built from -- S1, S2 and S3
come out of one pass, so the contrast below is per item -- and scored on the
same keep-set, so these numbers and the main table's are the same numbers.

    python experiments/C1_structural_trigger/S3_question_format_ablation/analyze_S3.py
"""
import json
import sys
from pathlib import Path

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from infra.evaluator import Evaluator             # noqa: E402
from infra.result_schema import paired_keep_ids   # noqa: E402

_EV = Evaluator()

MODELS = [("dsv4flash", "deepseek-v4-flash", "DeepSeek-V4-Flash"),
          ("nano", "gpt-5.4-nano", "GPT-5.4-nano"),
          ("gemini31", "gemini-3.1-flash-lite", "Gemini-3.1-Flash-Lite")]
DATASETS = ("FLD", "FOLIO")


def cell(slug: str, model: str, dataset: str) -> dict:
    summary = json.loads(
        (ROOT / f"results/tfq/{slug}/ab_summary_{dataset}_{model}.json").read_text())
    # The S1/S2 keep-set, minus whatever S3 itself failed to answer -- the same
    # rule the runner applies when it writes metrics.S3.n_scored. Counting an
    # exhausted retry as "did not abstain" would understate S3's Abs Rate.
    keep = paired_keep_ids(summary)
    rows = []
    for r in summary["per_sample"]:
        if r["id"] not in keep:
            continue
        pred = r.get("pred_s3_format")
        if pred == "UNPARSEABLE" and _EV.classify_unanswered(
                r.get("raw_s3_format") or "") != "no_commitment":
            continue
        if pred:
            rows.append(r)
    n = len(rows)
    s2 = sum(r["pred_s2"] == "UNKNOWN" for r in rows)
    s3 = sum(r["pred_s3_format"] == "UNKNOWN" for r in rows)
    # discordant pairs: abstained under one rendering but not the other
    b = sum(r["pred_s2"] == "UNKNOWN" and r["pred_s3_format"] != "UNKNOWN" for r in rows)
    c = sum(r["pred_s2"] != "UNKNOWN" and r["pred_s3_format"] == "UNKNOWN" for r in rows)
    return {"n": n, "abs_s2": 100 * s2 / n, "abs_s3": 100 * s3 / n,
            "delta": 100 * (s3 - s2) / n, "b": b, "c": c,
            "p": binomtest(b, b + c, 0.5).pvalue if b + c else 1.0}


def main() -> None:
    print(f"{'Model':<24} {'Dataset':<7} {'n':>4} {'S2 Abs':>7} {'S3 Abs':>7} "
          f"{'Delta':>7} {'McNemar p':>10}")
    print("-" * 72)
    deltas = []
    for slug, model, label in MODELS:
        for ds in DATASETS:
            r = cell(slug, model, ds)
            deltas.append(abs(r["delta"]))
            flag = " *" if r["p"] < 0.05 else ""
            print(f"{label:<24} {ds:<7} {r['n']:>4} {r['abs_s2']:>6.1f}% "
                  f"{r['abs_s3']:>6.1f}% {r['delta']:>+6.1f} {r['p']:>10.3g}{flag}")
    print("-" * 72)
    print(f"Largest |Delta Abs Rate| in any cell : {max(deltas):.1f} points")
    print(f"Mean    |Delta Abs Rate|             : {sum(deltas)/len(deltas):.1f} points")
    print("\nThe letter rendering alone does not account for the abstention:\n"
          "compare these against the S1->S2 jump the same table reports.")


if __name__ == "__main__":
    main()

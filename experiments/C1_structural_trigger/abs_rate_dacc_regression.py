"""Abs Rate vs. Delta-Acc regression across every cell of the main table.

One point per (model, dataset): the S2 Abs Rate against the S1->S2 accuracy
change. The question is whether abstention alone accounts for the accuracy a
model loses when the option appears, which it does on TFQs and does not on
4-option MCQs.

Output: results/analysis/abs_rate_dacc_regression.json
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from infra.result_schema import paired_keep_ids   # noqa: E402
OUT = ROOT / "results/analysis/abs_rate_dacc_regression.json"

MODELS = [("dsv4flash", "deepseek-v4-flash", "DeepSeek-V4-Flash"),
          ("nano", "gpt-5.4-nano", "GPT-5.4-nano"),
          ("gemini31", "gemini-3.1-flash-lite", "Gemini-3.1-Flash-Lite")]
TFQ = ["FLD", "FOLIO"]
MCQ = ["ARC", "MedQA", "MMLU", "LogiQA"]


def load_points():
    pts = []
    for slug, model, label in MODELS:
        for ds in TFQ + MCQ:
            if ds in TFQ:
                path = ROOT / f"results/tfq/{slug}/ab_summary_{ds}_{model}.json"
            else:
                path = Path(glob.glob(str(ROOT / f"results/mcq/*/ab_summary_{ds}_{model}.json"))[0])
            # Scored on the paired keep-set, the same denominator Table 1
            # uses; the MCQ summaries' own metrics block is not on it.
            summary = json.loads(path.read_text())
            keep = paired_keep_ids(summary)
            rows = [r for r in summary["per_sample"] if r["id"] in keep]
            n = len(rows)
            gold = lambda r: chr(ord("A") + r["answer_idx"])
            a1 = sum(r["pred_s1"] == gold(r) for r in rows) / n
            a2 = sum(r["pred_s2"] == gold(r) for r in rows) / n
            pts.append({"model": label, "dataset": ds,
                        "type": "tf" if ds in TFQ else "mcq",
                        "abs_rate": sum(r["pred_s2"] == "UNKNOWN" for r in rows) / n,
                        "d_acc": a2 - a1})
    return pts


def regress(x, y, label):
    slope, intercept, r, p, _se = stats.linregress(x, y)
    rho, p_rho = stats.spearmanr(x, y)
    return {"label": label, "n": len(x), "slope": round(slope, 4),
            "intercept": round(intercept, 4), "r": round(r, 4),
            "r2": round(r ** 2, 4), "p_pearson": float(f"{p:.4e}"),
            "spearman_rho": round(rho, 4), "p_spearman": float(f"{p_rho:.4e}")}


def main():
    pts = load_points()
    x = np.array([p["abs_rate"] for p in pts])
    y = np.array([p["d_acc"] for p in pts])
    tf = np.array([p["type"] == "tf" for p in pts])

    groups = [regress(x, y, "All (TFQ+MCQ)"),
              regress(x[tf], y[tf], "TFQ only"),
              regress(x[~tf], y[~tf], "MCQ only")]

    print("=" * 65)
    print("Abs Rate - Delta Acc Regression")
    print("=" * 65)
    for g in groups:
        print(f"\n[{g['label']}]  n={g['n']}")
        print(f"  OLS slope={g['slope']:+.4f}  intercept={g['intercept']:+.4f}")
        print(f"  Pearson r={g['r']:+.4f}  R2={g['r2']:.4f}  p={g['p_pearson']:.3g}")
        print(f"  Spearman rho={g['spearman_rho']:+.4f}  p={g['p_spearman']:.3g}")

    print("\n-- Per-point detail " + "-" * 44)
    print(f"{'Model':<24} {'Dataset':<8} {'Type':<5} {'Abs Rate':>9} {'dAcc':>8}")
    for p in pts:
        print(f"{p['model']:<24} {p['dataset']:<8} {p['type']:<5} "
              f"{p['abs_rate']:>8.1%} {p['d_acc']:>+8.1%}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"groups": groups, "points": pts}, indent=2))
    print(f"\nSaved -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

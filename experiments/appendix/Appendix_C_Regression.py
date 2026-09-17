"""Appendix C — Abs Rate against the accuracy the option costs.

One point per (model, dataset) cell of the main table: the S2 Abs Rate on the
x-axis, the S1 -> S2 accuracy change on the y-axis. If abstention alone
accounted for the loss, the slope would be -1 and the fit tight. On TFQs it is
nearly that; on 4-option MCQs neither axis has variance to speak of and the
correlation vanishes, which is the asymmetry the paper is about.

Scored on the same paired keep-set the main table uses -- the items both
settings answered -- because a regression over one denominator against a table
built on another is not describing the same experiment.

    python experiments/Appendix_C_Regression.py
    python experiments/Appendix_C_Regression.py --out results/analysis/appendix_c.json
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from infra.result_schema import paired_keep_ids   # noqa: E402

MODELS = [("dsv4flash", "deepseek-v4-flash", "DeepSeek-V4-Flash"),
          ("nano", "gpt-5.4-nano", "GPT-5.4-nano"),
          ("gemini31", "gemini-3.1-flash-lite", "Gemini-3.1-Flash-Lite")]
TFQ = ["FLD", "FOLIO"]
MCQ = ["ARC", "MedQA", "MMLU", "LogiQA"]


def _cell(slug: str, model: str, dataset: str) -> dict:
    """Abs Rate and Delta Acc for one cell, on the paired keep-set."""
    if dataset in TFQ:
        path = ROOT / f"results/tfq/{slug}/ab_summary_{dataset}_{model}.json"
    else:
        hits = glob.glob(str(ROOT / f"results/mcq/*/ab_summary_{dataset}_{model}.json"))
        if not hits:
            raise FileNotFoundError(f"no summary for {dataset} / {model}")
        path = Path(hits[0])

    summary = json.loads(path.read_text())
    keep = paired_keep_ids(summary)
    rows = [r for r in summary["per_sample"] if r["id"] in keep]
    n = len(rows)
    gold = lambda r: chr(ord("A") + r["answer_idx"])          # noqa: E731
    acc1 = sum(r["pred_s1"] == gold(r) for r in rows) / n
    acc2 = sum(r["pred_s2"] == gold(r) for r in rows) / n
    return {"model": [m[2] for m in MODELS if m[1] == model][0],
            "dataset": dataset,
            "type": "tf" if dataset in TFQ else "mcq",
            "n": n,
            "abs_rate": sum(r["pred_s2"] == "UNKNOWN" for r in rows) / n,
            "d_acc": acc2 - acc1}


def regress(x, y, label: str) -> dict:
    slope, intercept, r, p, _se = stats.linregress(x, y)
    rho, p_rho = stats.spearmanr(x, y)
    return {"label": label, "n_cells": len(x),
            "slope": round(slope, 4), "intercept": round(intercept, 4),
            "pearson_r": round(r, 4), "r2": round(r ** 2, 4),
            "p_pearson": float(f"{p:.4e}"),
            "spearman_rho": round(rho, 4), "p_spearman": float(f"{p_rho:.4e}")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "results/analysis/appendix_c_regression.json"),
                    help="Where to write the fits as JSON.")
    args = ap.parse_args()

    points = [_cell(slug, model, ds)
              for slug, model, _label in MODELS
              for ds in TFQ + MCQ]
    x = np.array([p["abs_rate"] for p in points])
    y = np.array([p["d_acc"] for p in points])
    tf = np.array([p["type"] == "tf" for p in points])

    fits = [regress(x, y, "All cells"),
            regress(x[tf], y[tf], "TFQ only"),
            regress(x[~tf], y[~tf], "MCQ only")]

    print("=" * 66)
    print("Appendix C — Abs Rate vs. Delta Acc")
    print("=" * 66)
    for f in fits:
        print(f"\n[{f['label']}]  {f['n_cells']} cells")
        print(f"  OLS slope   {f['slope']:+.4f}   intercept {f['intercept']:+.4f}")
        print(f"  Pearson  r  {f['pearson_r']:+.4f}   R2 {f['r2']:.4f}   p {f['p_pearson']:.3g}")
        print(f"  Spearman rho {f['spearman_rho']:+.4f}   p {f['p_spearman']:.3g}")

    print("\n" + "-" * 66)
    print(f"{'Model':<24} {'Dataset':<8} {'Type':<5} {'n':>5} {'Abs Rate':>9} {'dAcc':>8}")
    for p in points:
        print(f"{p['model']:<24} {p['dataset']:<8} {p['type']:<5} {p['n']:>5} "
              f"{p['abs_rate']:>8.1%} {p['d_acc']:>+8.1%}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"fits": fits, "points": points}, indent=2))
    print(f"\nSaved -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

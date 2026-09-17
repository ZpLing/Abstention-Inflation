"""
T-4A: Abs Rate–ΔAcc scatter regression
使用 results_summary.txt 中 n=200 池化版干净数据，跑 OLS + Spearman 相关。
输出：results/analysis/abs_rate_dacc_regression.json
"""

import json, os
import numpy as np
from scipy import stats

# ── 干净数据点（来自 results_summary.txt PART 0，n=200 池化版）────────────────
# 格式: (model_short, dataset, task_type, Abs Rate, ΔAcc)
DATA = [
    # TF — FLD (n=200)
    ("deepseek", "FLD",   "tf",  0.380, -0.185),
    ("nano",     "FLD",   "tf",  0.660, -0.280),
    ("gemini",   "FLD",   "tf",  0.410, -0.175),
    # TF — FOLIO (n=200)
    ("deepseek", "FOLIO", "tf",  0.130, -0.080),
    ("nano",     "FOLIO", "tf",  0.245, -0.145),
    ("gemini",   "FOLIO", "tf",  0.150, -0.080),
    # TF — FEVER (deepseek n=250, nano/gemini n=100)
    ("deepseek", "FEVER", "tf",  0.096, -0.024),
    ("nano",     "FEVER", "tf",  0.100, -0.060),
    ("gemini",   "FEVER", "tf",  0.060, -0.050),
    # MCQ — ARC-Challenge (n=200)
    ("deepseek", "ARC-C", "mcq", 0.004, +0.032),
    ("nano",     "ARC-C", "mcq", 0.005, -0.015),
    ("gemini",   "ARC-C", "mcq", 0.000, -0.005),
    # MCQ — MedQA (n=200 generic)
    ("deepseek", "MedQA", "mcq", 0.020, +0.045),
    ("nano",     "MedQA", "mcq", 0.035, +0.045),
    ("gemini",   "MedQA", "mcq", 0.015, +0.025),
]

models   = [d[0] for d in DATA]
datasets = [d[1] for d in DATA]
types    = [d[2] for d in DATA]
abs_rate      = np.array([d[3] for d in DATA])
dAcc     = np.array([d[4] for d in DATA])

tf_mask  = np.array([t == "tf"  for t in types])
mcq_mask = np.array([t == "mcq" for t in types])


def regress(x, y, label):
    slope, intercept, r, p, se = stats.linregress(x, y)
    rho, psp = stats.spearmanr(x, y)
    return {
        "label":     label,
        "n":         len(x),
        "slope":     round(slope, 4),
        "intercept": round(intercept, 4),
        "r":         round(r, 4),
        "r2":        round(r**2, 4),
        "p_pearson": float(f"{p:.4e}"),
        "spearman_rho": round(rho, 4),
        "p_spearman":   float(f"{psp:.4e}"),
    }


results = {
    "all":  regress(abs_rate,          dAcc,          "All (TF+MCQ)"),
    "tf":   regress(abs_rate[tf_mask], dAcc[tf_mask], "TF only"),
    "mcq":  regress(abs_rate[mcq_mask],dAcc[mcq_mask],"MCQ only"),
    "data_points": [
        {"model": m, "dataset": ds, "type": t, "abs_rate": a, "dAcc": d}
        for m, ds, t, a, d in zip(models, datasets, types, abs_rate.tolist(), dAcc.tolist())
    ]
}

# ── 打印摘要 ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("Abs Rate – ΔAcc Regression Analysis")
print("=" * 65)
for key in ("all", "tf", "mcq"):
    r = results[key]
    print(f"\n[{r['label']}]  n={r['n']}")
    print(f"  OLS slope={r['slope']:+.4f}  intercept={r['intercept']:+.4f}")
    print(f"  Pearson r={r['r']:+.4f}  R²={r['r2']:.4f}  p={r['p_pearson']:.2e}")
    print(f"  Spearman ρ={r['spearman_rho']:+.4f}  p={r['p_spearman']:.2e}")

print("\n── Per-point detail ──────────────────────────────────────────")
print(f"{'Model':<10} {'Dataset':<8} {'Type':<5} {'Abs Rate':>7} {'ΔAcc':>8}")
for pt in results["data_points"]:
    print(f"{pt['model']:<10} {pt['dataset']:<8} {pt['type']:<5} {pt['Abs Rate']:>7.1%} {pt['dAcc']:>+8.1%}")

# ── 保存 ─────────────────────────────────────────────────────────────────────
out = "/Users/timchef/WakenLLM-toolkit/results/analysis/abs_rate_dacc_regression.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved → {out}")

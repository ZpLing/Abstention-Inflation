"""(a) F1T vs γ-rate cross-cell correlation analysis

Across the 6 cells (3 models × {FLD, FOLIO}), test whether F1T (trace
structural quality) correlates with γ-rate (NLI decisive AND correct AND
pred=UNKNOWN, normalized by total Abstention Inflation samples).

Hypothesis: higher reasoning-trace quality → larger γ (a correct trace is a prerequisite for a flippable conclusion).
"""
import json
import sys
from pathlib import Path
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, ".")

ROOT = Path(".")

# (model, dataset, NLI summary path, F1T source summary paths)
CELLS = [
    ("nano", "FLD",
     "results/probe/cot_tasksource_FLD_nano_n200.json",
     ["results/ab_gpt5_nano/ab_summary_FLD_gpt-5.4-nano.json",
      "results/ab_nano_batch2/ab_summary_FLD_gpt-5.4-nano.json"]),
    ("gemini", "FLD",
     "results/probe/cot_tasksource_FLD_gemini_n200.json",
     ["results/ab_gemini_flash_lite/ab_summary_FLD_gemini-3.1-flash-lite.json",
      "results/ab_gemini_batch2/ab_summary_FLD_gemini-3.1-flash-lite.json"]),
    ("deepseek", "FLD",
     "results/probe/cot_tasksource_FLD_deepseek_n200.json",
     ["results/ab_e_option_baseline/ab_summary_FLD_deepseek-v4-flash.json",
      "results/ab_deepseek_batch2/ab_summary_FLD_deepseek-v4-flash.json"]),
    ("nano", "FOLIO",
     "results/probe/cot_tasksource_FOLIO_nano_n200.json",
     ["results/ab_gpt5_nano/ab_summary_FOLIO_gpt-5.4-nano.json",
      "results/ab_nano_batch2/ab_summary_FOLIO_gpt-5.4-nano.json"]),
    ("gemini", "FOLIO",
     "results/probe/cot_tasksource_FOLIO_gemini_n200.json",
     ["results/ab_gemini_flash_lite/ab_summary_FOLIO_gemini-3.1-flash-lite.json",
      "results/ab_gemini_batch2/ab_summary_FOLIO_gemini-3.1-flash-lite.json"]),
    ("deepseek", "FOLIO",
     "results/probe/cot_tasksource_FOLIO_deepseek_n200.json",
     ["results/ab_followup/ab_summary_FOLIO_deepseek-v4-flash.json",
      "results/ab_deepseek_batch2/ab_summary_FOLIO_deepseek-v4-flash.json"]),
]


def pool_f1t_s2(paths):
    """Sample-weighted F1T_S2 across batch1+batch2."""
    total_n = 0
    weighted = 0.0
    for p in paths:
        d = json.loads(Path(p).read_text())
        n = d.get("n_total", 0)
        f = d.get("metrics", {}).get("S2", {}).get("trace_f1", 0)
        weighted += (f or 0) * n
        total_n += n
    return weighted / max(total_n, 1)


def gamma_rate(probe_path):
    """γ rate from NLI probe = decisive AND correct, divided by total Abs Rate."""
    d = json.loads(Path(probe_path).read_text())
    s2 = d.get("cot_probe", {}).get("s2_stats", {})
    if not s2: return None, None
    n = s2["n"]
    gamma_correct = s2.get("gamma_correct", 0)
    return gamma_correct / max(n, 1), n


print(f"{'Model':<10} {'DS':<6} {'F1T_S2':>8} {'γ-rate':>8} {'n_ai':>6}")
print("-" * 50)

f1ts = []
gammas = []
labels = []
for model, ds, probe, f1t_paths in CELLS:
    g, n = gamma_rate(probe)
    f = pool_f1t_s2(f1t_paths)
    f1ts.append(f); gammas.append(g); labels.append(f"{model}-{ds[:1]}")
    print(f"{model:<10} {ds:<6} {f:>8.3f} {g:>8.1%} {n:>6d}")

print()
rho, p_sp = spearmanr(f1ts, gammas)
r,   p_pe = pearsonr(f1ts, gammas)
print(f"Across {len(f1ts)} cells:")
print(f"  Spearman ρ = {rho:+.3f}  p = {p_sp:.4g}")
print(f"  Pearson  r = {r:+.3f}  p = {p_pe:.4g}")
print()
print("Interpretation:")
print("  ρ > 0  → higher trace quality predicts larger γ share (supports the hypothesis)")
print("  ρ ≈ 0  → no relationship (γ does not depend on trace structure)")
print("  ρ < 0  → higher trace quality predicts smaller γ (reverse direction, uncommon)")

Path("results/analysis").mkdir(parents=True, exist_ok=True)
out = ROOT / "results/analysis/trace_quality_vs_gamma.json"
out.write_text(json.dumps({
    "cells": [{"label": l, "f1t_s2": f, "gamma_rate": g}
              for l, f, g in zip(labels, f1ts, gammas)],
    "spearman_rho": rho, "spearman_p": p_sp,
    "pearson_r": r, "pearson_p": p_pe,
}, indent=2))
print(f"\nSaved → {out}")

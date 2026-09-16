"""S2 vs S5 paired comparison on FLD/FOLIO across 3 models.

S2 = TF prompt with appended "Unknown" verb option (proved/disproved/unknown)
S5 = same task reformulated as MCQ (A=Proved, B=Disproved, C=Unknown)

Question: does abstention behavior change when the abstain option is rendered
as a letter (C) rather than a verb (Unknown)?
"""
import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, ".")
from scipy.stats import binomtest

ROOT = Path(".")
PAIRS = [
    ("nano",     "FLD",   ["results/ab_gpt5_nano/ab_summary_FLD_gpt-5.4-nano.json",
                            "results/ab_nano_batch2/ab_summary_FLD_gpt-5.4-nano.json"]),
    ("gemini",   "FLD",   ["results/ab_gemini_flash_lite/ab_summary_FLD_gemini-3.1-flash-lite.json",
                            "results/ab_gemini_batch2/ab_summary_FLD_gemini-3.1-flash-lite.json"]),
    ("deepseek", "FLD",   ["results/ab_e_option_baseline/ab_summary_FLD_deepseek-r1-distill-llama-8b.json",
                            "results/ab_deepseek_batch2/ab_summary_FLD_deepseek-r1-distill-llama-8b.json"]),
    ("nano",     "FOLIO", ["results/ab_gpt5_nano/ab_summary_FOLIO_gpt-5.4-nano.json",
                            "results/ab_nano_batch2/ab_summary_FOLIO_gpt-5.4-nano.json"]),
    ("gemini",   "FOLIO", ["results/ab_gemini_flash_lite/ab_summary_FOLIO_gemini-3.1-flash-lite.json",
                            "results/ab_gemini_batch2/ab_summary_FOLIO_gemini-3.1-flash-lite.json"]),
    ("deepseek", "FOLIO", ["results/ab_followup/ab_summary_FOLIO_deepseek-r1-distill-llama-8b.json",
                            "results/ab_deepseek_batch2/ab_summary_FOLIO_deepseek-r1-distill-llama-8b.json"]),
]

def is_correct(pred, ai):
    if pred in ("UNKNOWN", "UNPARSEABLE", None, ""):
        return False
    return ord(pred) - ord("A") == ai

def mcnemar_p(n10, n01):
    d = n10 + n01
    if d == 0: return 1.0
    return binomtest(n01, n=d, p=0.5, alternative="two-sided").pvalue

def sig(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."

print(f"{'Model':<10} {'DS':<6} {'n':>4}  "
      f"{'abs_rate_s2':>7} {'abs_rate_s5':>7} {'ΔAIR':>7}  "
      f"{'Acc_S2':>7} {'Acc_S5':>7} {'ΔAcc':>7}  "
      f"{'disc(+/-)':>10} {'p':>9}  sig")
print("-" * 110)

rows = []
for model, ds, paths in PAIRS:
    pooled = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        pooled.extend(d.get("per_sample", []))
    n = len(pooled)
    if n == 0:
        print(f"{model:<10} {ds:<6} skip"); continue

    abs_rate_s2 = sum(1 for s in pooled if s.get("pred_s2") == "UNKNOWN")
    ai_s5 = sum(1 for s in pooled if s.get("pred_s5") == "UNKNOWN")
    correct_s2 = sum(1 for s in pooled if is_correct(s.get("pred_s2",""), s.get("answer_idx",-1)))
    correct_s5 = sum(1 for s in pooled if is_correct(s.get("pred_s5",""), s.get("answer_idx",-1)))

    # paired McNemar on (S2 correct, S5 correct)
    pairs = [(is_correct(s.get("pred_s2",""), s.get("answer_idx",-1)),
              is_correct(s.get("pred_s5",""), s.get("answer_idx",-1))) for s in pooled]
    n10 = sum(1 for a,b in pairs if a and not b)   # S2 right, S5 wrong
    n01 = sum(1 for a,b in pairs if not a and b)   # S2 wrong, S5 right
    p = mcnemar_p(n10, n01)

    print(f"{model:<10} {ds:<6} {n:>4d}  "
          f"{abs_rate_s2/n:>6.1%} {ai_s5/n:>6.1%} {(ai_s5-abs_rate_s2)/n:>+6.1%}  "
          f"{correct_s2/n:>6.1%} {correct_s5/n:>6.1%} {(correct_s5-correct_s2)/n:>+6.1%}  "
          f"{n01:>4d}/{n10:<4d} {p:>9.4g}  {sig(p)}")
    rows.append({"model":model,"dataset":ds,"n":n,
                 "abs_rate_s2":abs_rate_s2/n,"ai_s5":ai_s5/n,
                 "acc_s2":correct_s2/n,"acc_s5":correct_s5/n,
                 "delta_acc":correct_s5/n-correct_s2/n,
                 "n10":n10,"n01":n01,"p":p})

# Pooled overall
pooled_pairs = []
for model, ds, paths in PAIRS:
    for p in paths:
        d = json.loads(Path(p).read_text())
        for s in d.get("per_sample", []):
            pooled_pairs.append((is_correct(s.get("pred_s2",""), s.get("answer_idx",-1)),
                                 is_correct(s.get("pred_s5",""), s.get("answer_idx",-1))))
n10 = sum(1 for a,b in pooled_pairs if a and not b)
n01 = sum(1 for a,b in pooled_pairs if not a and b)
n11 = sum(1 for a,b in pooled_pairs if a and b)
n00 = sum(1 for a,b in pooled_pairs if not a and not b)
total = n10+n01+n11+n00
p = mcnemar_p(n10, n01)
print()
print(f"Pooled all 6 cells (n={total}):")
print(f"  Acc_S2 = {(n11+n10)/total:.1%}   Acc_S5 = {(n11+n01)/total:.1%}   "
      f"ΔAcc = {((n11+n01)-(n11+n10))/total:+.1%}   p = {p:.4g}  {sig(p)}")

Path("results/analysis").mkdir(parents=True, exist_ok=True)
out = ROOT / "results/analysis/s2_vs_s5_comparison.json"
out.write_text(json.dumps({
    "rows": rows,
    "pooled": {"n": total, "n11":n11,"n10":n10,"n01":n01,"n00":n00,
               "acc_s2":(n11+n10)/total,"acc_s5":(n11+n01)/total,"p":p},
}, indent=2))
print(f"\nSaved → {out}")

"""S4 Word Content Ablation — does the abstain word matter, or just the slot?

Two halves, both reported in the paper:

  synonyms      W2-W5 replace "Unknown" with "I don't know", "Indeterminate",
                "Cannot be determined from the facts", "Insufficient
                information". If abstention tracked the word's meaning, a
                near-synonym would move it; it does not.
  random words  the third option becomes "Triangular" or "Cerulean", words with
                no bearing on the task. Models still select that slot at close
                to the Unknown rate, which is what makes the trigger structural.

Both are scored against the S2 cell of the paired summaries the main table is
built from, so these numbers and Table 1's are the same numbers.

    python experiments/C1_structural_trigger/S4_word_content_ablation/analyze_S4.py
"""
import json
from glob import glob
from math import sqrt
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parents[3]
MODEL = "deepseek-v4-flash"

WORDINGS = ["W1", "W2", "W3", "W4", "W5"]
WORDING_TEXTS = {
    "W1": "Unknown / Uncertain (baseline)",
    "W2": "I don't know",
    "W3": "Indeterminate",
    "W4": "Cannot be determined from the facts",
    "W5": "Insufficient information",
}
WORDING_CLASS = {"W1": "synonym", "W2": "synonym", "W3": "synonym",
                 "W4": "cause", "W5": "cause"}
DATASETS = ["FLD", "FOLIO"]


def load_w1_per_sample(ds):
    """W1 is the S2 cell of the main table -- id -> pred_s2.

    Read from the paired summary Table 1 is built from, so the wording sweep is
    compared against the same run the paper reports rather than an earlier one.
    """
    path = ROOT / f"results/tfq/dsv4flash/ab_summary_{ds}_{MODEL}.json"
    ab = json.loads(path.read_text())
    return {ps["id"]: ps["pred_s2"] for ps in ab.get("per_sample", [])}


def load_wording_per_sample(w, ds):
    p = ROOT / f"results/wording_sweep/summary_{w}_{ds}_{MODEL}.json"
    s = json.loads(p.read_text())
    return {ps["id"]: ps["pred"] for ps in s["per_sample"]}


def mcnemar_b_c(pred_a, pred_b, sample_ids, abstain_label="UNKNOWN"):
    """Paired binary outcomes: a_abstain XOR b_abstain. Returns (b, c)
    where b = (a abstain, b not), c = (b abstain, a not)."""
    b, c = 0, 0
    for sid in sample_ids:
        a_ab = pred_a.get(sid) == abstain_label
        b_ab = pred_b.get(sid) == abstain_label
        if a_ab and not b_ab:
            b += 1
        elif b_ab and not a_ab:
            c += 1
    return b, c


def mcnemar_exact_p(b, c):
    """Two-sided exact binomial McNemar p-value."""
    n = b + c
    if n == 0:
        return 1.0, 0.0
    # Use normal approximation for n large; exact for small
    k = min(b, c)
    if n <= 25:
        # Exact two-sided binomial(n, 0.5)
        from math import comb
        p_two_tail = 2 * sum(comb(n, i) * 0.5 ** n for i in range(k + 1))
        p_two_tail = min(p_two_tail, 1.0)
        z = (b - c) / sqrt(b + c) if (b + c) > 0 else 0
        return p_two_tail, z
    # Continuity-corrected chi-square approximation → z
    z = (abs(b - c) - 1) / sqrt(b + c)
    p = 2 * (1 - NormalDist().cdf(abs(z)))
    if (b - c) < 0:
        z = -z
    return p, z


RPC = ROOT / "results/random_perturbation_control"
RPC_MODELS = [("deepseek-v4-flash", "DeepSeek-V4-Flash"),
              ("gpt-5.4-nano", "GPT-5.4-nano"),
              ("gemini-3.1-flash-lite", "Gemini-3.1-Flash-Lite")]


def random_word_half():
    """The Triangular / Cerulean control, against the same S2 baseline."""
    print("\n" + "=" * 78)
    print("Random words — rate at which the third slot is selected")
    print("=" * 78)
    print(f"{'Model':<24} {'Dataset':<7} {'Unknown':>8} {'Rand1':>8} {'Rand2':>8} "
          f"{'max |d|':>8}")
    worst = []
    for model, label in RPC_MODELS:
        for ds in DATASETS:
            path = RPC / f"rpc_{ds}_{model}.json"
            if not path.exists():
                print(f"{label:<24} {ds:<7} (missing)")
                continue
            d = json.loads(path.read_text())
            base = d["C1_Unknown"]["opt_x_rate"]
            r1 = d["C2_Rand1"]["opt_x_rate"]
            r2 = d["C3_Rand2"]["opt_x_rate"]
            delta = max(abs(r1 - base), abs(r2 - base)) * 100
            worst.append(delta)
            print(f"{label:<24} {ds:<7} {base:>7.1%} {r1:>7.1%} {r2:>7.1%} "
                  f"{delta:>7.1f}pp")
    if worst:
        print("-" * 78)
        print(f"Largest shift from the Unknown baseline : {max(worst):.1f} points")
        print(f"Mean shift                              : {sum(worst)/len(worst):.1f} points")


def main():
    print(f"=== §6.1 Wording sweep aggregation ({MODEL}) ===\n")

    cells = {}  # (ds, w) -> {n, n_ai, abs_rate, pred_by_id}

    for ds in DATASETS:
        # W1 from main exp
        w1_pred = load_w1_per_sample(ds)
        # Use the same ordered ID list as the wording sweep
        w2_path = ROOT / f"results/wording_sweep/summary_W2_{ds}_{MODEL}.json"
        w2_summary = json.loads(w2_path.read_text())
        sample_ids = [ps["id"] for ps in w2_summary["per_sample"]]
        # W1 restricted to these IDs
        w1_pred_aligned = {sid: w1_pred[sid] for sid in sample_ids if sid in w1_pred}
        n_w1 = len(w1_pred_aligned)
        n_ai_w1 = sum(1 for v in w1_pred_aligned.values() if v == "UNKNOWN")
        cells[(ds, "W1")] = {
            "n": n_w1,
            "n_ai": n_ai_w1,
            "abs_rate": n_ai_w1 / n_w1 if n_w1 else 0,
            "pred_by_id": w1_pred_aligned,
        }
        # W2-W5
        for w in ["W2", "W3", "W4", "W5"]:
            pred = load_wording_per_sample(w, ds)
            pred_aligned = {sid: pred[sid] for sid in sample_ids if sid in pred}
            n = len(pred_aligned)
            n_ai = sum(1 for v in pred_aligned.values() if v == "UNKNOWN")
            cells[(ds, w)] = {
                "n": n,
                "n_ai": n_ai,
                "abs_rate": n_ai / n if n else 0,
                "pred_by_id": pred_aligned,
            }

    # Print Abs Rate table
    print(f"{'Dataset':8s} {'Wording':5s} {'n':>4s} {'n_ai':>6s} {'Abs Rate':>7s}  {'Wording text'}")
    print("-" * 90)
    for ds in DATASETS:
        for w in WORDINGS:
            c = cells[(ds, w)]
            print(f"{ds:8s} {w:5s} {c['n']:>4} {c['n_ai']:>6} {c['abs_rate']:>7.1%}  {WORDING_TEXTS[w]}")
        print()

    # Paired McNemar W1 vs each Wi
    print("=== Paired McNemar (W1 vs Wi, same items) ===\n")
    print(f"{'Dataset':8s} {'W1':>4s} {'vs':>3s} {'Wi':>4s} {'b':>4s} {'c':>4s} {'Δ_AIR':>7s} {'p':>10s}")
    print("-" * 60)
    mcnemar_table = []
    for ds in DATASETS:
        w1 = cells[(ds, "W1")]
        sample_ids = list(w1["pred_by_id"].keys())
        for w in ["W2", "W3", "W4", "W5"]:
            wi = cells[(ds, w)]
            b, c = mcnemar_b_c(w1["pred_by_id"], wi["pred_by_id"], sample_ids)
            p, z = mcnemar_exact_p(b, c)
            delta = wi["abs_rate"] - w1["abs_rate"]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"{ds:8s} {'W1':>4s} {'vs':>3s} {w:>4s} {b:>4} {c:>4} {delta:>+6.1%} {p:>10.4f} {sig}")
            mcnemar_table.append({
                "ds": ds, "wi": w, "b": b, "c": c,
                "delta_abs_rate": delta, "p": p, "z": z, "sig": sig,
            })
        print()

    # Variance partition: σ²(synonym) vs σ²(cause)
    print("=== Variance partition (per dataset) ===\n")
    var_partition = {}
    for ds in DATASETS:
        abs_rates = {w: cells[(ds, w)]["abs_rate"] for w in WORDINGS}
        synonyms = [abs_rates["W1"], abs_rates["W2"], abs_rates["W3"]]
        causes = [abs_rates["W4"], abs_rates["W5"]]
        all_w = list(abs_rates.values())

        def var(xs):
            if len(xs) < 2:
                return 0.0
            m = sum(xs) / len(xs)
            return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)

        v_all = var(all_w)
        v_syn = var(synonyms)
        v_cau = var(causes)
        # σ²(group) = mean of group variances
        # σ²(between groups) = variance of group means × n_per_group
        v_within = ((len(synonyms) - 1) * v_syn + (len(causes) - 1) * v_cau) / (len(synonyms) + len(causes) - 2)
        m_syn = sum(synonyms) / len(synonyms)
        m_cau = sum(causes) / len(causes)
        # cluster shift in pp
        cluster_shift = m_cau - m_syn

        print(f"{ds}:")
        print(f"  Abs Rate all 5 wordings:   {[round(a*100, 1) for a in all_w]}")
        print(f"  σ(synonyms W1/W2/W3): {sqrt(v_syn)*100:.2f} pp  (range {min(synonyms)*100:.1f}-{max(synonyms)*100:.1f})")
        print(f"  σ(cause W4/W5):       {sqrt(v_cau)*100:.2f} pp  (range {min(causes)*100:.1f}-{max(causes)*100:.1f})")
        print(f"  σ(all 5):             {sqrt(v_all)*100:.2f} pp")
        print(f"  Δ(cause - synonym):   {cluster_shift*100:+.1f} pp  ← cause-attribution effect")
        print()

        var_partition[ds] = {
            "abs_rates": abs_rates,
            "sigma_synonyms": sqrt(v_syn),
            "sigma_cause": sqrt(v_cau),
            "sigma_all5": sqrt(v_all),
            "cluster_shift": cluster_shift,
            "synonym_mean": m_syn,
            "cause_mean": m_cau,
        }

    # Save
    out = {
        "model": MODEL,
        "wordings": WORDING_TEXTS,
        "wording_class": WORDING_CLASS,
        "cells": {f"{ds}_{w}": {k: v for k, v in cells[(ds, w)].items() if k != "pred_by_id"}
                   for ds in DATASETS for w in WORDINGS},
        "mcnemar_W1_vs_Wi": mcnemar_table,
        "variance_partition": var_partition,
    }
    out_path = ROOT / "results/analysis/wording_sweep_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")



if __name__ == "__main__":
    main()
    random_word_half()

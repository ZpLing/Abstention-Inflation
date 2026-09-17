"""Effect of adding the Unknown option (S1 → S2) on accuracy, split by task type.

Loads all ab_summary_*.json files under results/ab_*/ directories.
For each (model, dataset), computes per-sample (correct_s1, correct_s2) pairs.
Groups by MCQ vs TF, then runs:
  - McNemar's test (within-group pooled): p-value for accuracy change
  - Per-dataset summary table: ΔAcc = Acc_S2 − Acc_S1, Abs Rate
  - Spearman correlation: Abs Rate vs ΔAcc across all datasets

Usage:
    python -m scripts.analyze_acc_effect
    python -m scripts.analyze_acc_effect --results_dirs results/ab_gpt5_nano results/ab_gemini_flash_lite
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


_BAD = {"UNKNOWN", "UNPARSEABLE", None, ""}

MCQ_DATASETS = {"ARC-Easy_250", "ARC-Challenge_250", "ARC-Easy", "ARC-Challenge", "MedQA",
                "MedQA-Step1", "MedQA-Step2_3"}
TF_DATASETS  = {"FLD", "FOLIO", "FEVER"}


def is_correct(pred: str, answer_idx: int) -> bool:
    if pred in _BAD:
        return False
    return ord(pred) - ord("A") == answer_idx


def load_summaries(results_dirs: List[Path]) -> List[dict]:
    summaries = []
    for d in results_dirs:
        for path in sorted(d.glob("ab_summary_*.json")):
            try:
                with open(path) as f:
                    data = json.load(f)
                data["_source_file"] = str(path)
                summaries.append(data)
            except Exception as e:
                print(f"[warn] Could not load {path}: {e}")
    return summaries


def mcnemar_table(pairs: List[Tuple[bool, bool]]) -> Tuple[int, int, int, int]:
    """Build 2x2 McNemar contingency table.
    Returns (n11, n10, n01, n00):
        n11 = both correct
        n10 = s1 correct, s2 wrong
        n01 = s1 wrong, s2 correct
        n00 = both wrong
    """
    n11 = sum(1 for a, b in pairs if a and b)
    n10 = sum(1 for a, b in pairs if a and not b)
    n01 = sum(1 for a, b in pairs if not a and b)
    n00 = sum(1 for a, b in pairs if not a and not b)
    return n11, n10, n01, n00


def mcnemar_p(n10: int, n01: int) -> float:
    """McNemar's test p-value (exact binomial on discordant pairs).
    H0: no difference in marginal probabilities (P(S2 correct) == P(S1 correct)).
    """
    discordant = n10 + n01
    if discordant == 0:
        return 1.0
    try:
        from scipy.stats import binomtest  # type: ignore  # scipy >= 1.7
        return binomtest(n01, n=discordant, p=0.5, alternative="two-sided").pvalue
    except ImportError:
        from scipy.stats import binom_test  # type: ignore  # scipy < 1.7
        return binom_test(n01, n=discordant, p=0.5, alternative="two-sided")


def spearman(x: List[float], y: List[float]) -> Tuple[float, float]:
    from scipy.stats import spearmanr  # type: ignore
    if len(x) < 3:
        return float("nan"), float("nan")
    rho, p = spearmanr(x, y)
    return float(rho), float(p)


def pearson(x: List[float], y: List[float]) -> Tuple[float, float]:
    from scipy.stats import pearsonr  # type: ignore
    if len(x) < 3:
        return float("nan"), float("nan")
    r, p = pearsonr(x, y)
    return float(r), float(p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dirs", nargs="*",
        help="Directories to scan (default: all results/ab_*/ under project root)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    if args.results_dirs:
        results_dirs = [Path(d) for d in args.results_dirs]
    else:
        results_dirs = sorted(root.glob("results/ab_*/"))

    print(f"Scanning {len(results_dirs)} result directory/ies:")
    for d in results_dirs:
        print(f"  {d}")

    summaries = load_summaries(results_dirs)
    print(f"\nLoaded {len(summaries)} summary files.\n")

    # ----------------------------------------------------------------
    # 1. Per-dataset row: ΔAcc, Abs Rate, (model, dataset, task_type)
    # ----------------------------------------------------------------
    rows = []
    mcq_pairs:    List[Tuple[bool, bool]] = []  # S1 vs S2
    mcq_pairs_13: List[Tuple[bool, bool]] = []  # S1 vs S3
    mcq_pairs_23: List[Tuple[bool, bool]] = []  # S2 vs S3
    tf_pairs:     List[Tuple[bool, bool]] = []
    tf_pairs_13:  List[Tuple[bool, bool]] = []
    tf_pairs_23:  List[Tuple[bool, bool]] = []

    for s in summaries:
        ds      = s.get("dataset", "?")
        model   = s.get("model", "?")
        tt      = s.get("task_type", "?")
        metrics = s.get("metrics", {})
        s1m     = metrics.get("S1", {})
        s2m     = metrics.get("S2", {})
        acc_s1  = s1m.get("label_acc", None)
        acc_s2  = s2m.get("label_acc", None)
        n_total = s.get("n_total", 0)
        n_ai   = s.get("n_abstention_inflation", 0)

        if acc_s1 is None or acc_s2 is None or n_total == 0:
            print(f"  [skip] {model}/{ds}: missing metrics.")
            continue

        delta_acc = acc_s2 - acc_s1
        abs_rate  = n_ai / n_total

        # Per-sample pairs for all setting comparisons
        per_sample = s.get("per_sample", [])
        p12_here: List[Tuple[bool, bool]] = []  # S1 vs S2
        p13_here: List[Tuple[bool, bool]] = []  # S1 vs S3
        p23_here: List[Tuple[bool, bool]] = []  # S2 vs S3
        for ps in per_sample:
            ai = ps.get("answer_idx", -1)
            if ai < 0:
                continue
            c1 = is_correct(ps.get("pred_s1", ""), ai)
            c2 = is_correct(ps.get("pred_s2", ""), ai)
            c3 = is_correct(ps.get("pred_s3", ""), ai)
            p12_here.append((c1, c2))
            p13_here.append((c1, c3))
            p23_here.append((c2, c3))

        if tt == "mcq":
            mcq_pairs.extend(p12_here)
            mcq_pairs_13.extend(p13_here)
            mcq_pairs_23.extend(p23_here)
        elif tt == "tf":
            tf_pairs.extend(p12_here)
            tf_pairs_13.extend(p13_here)
            tf_pairs_23.extend(p23_here)

        acc_s3 = metrics.get("S3", {}).get("label_acc", None)
        rows.append({
            "model":     model,
            "dataset":   ds,
            "task_type": tt,
            "acc_s1":    acc_s1,
            "acc_s2":    acc_s2,
            "acc_s3":    acc_s3,
            "delta_s1s2": acc_s2 - acc_s1,
            "delta_s1s3": (acc_s3 - acc_s1) if acc_s3 is not None else None,
            "delta_s2s3": (acc_s3 - acc_s2) if acc_s3 is not None else None,
            "delta_acc":  delta_acc,
            "abs_rate":        abs_rate,
            "n":          n_total,
            "n_pairs":    len(p12_here),
        })

    if not rows:
        print("No data found. Check results directories.")
        sys.exit(1)

    # ----------------------------------------------------------------
    # 2. Per-dataset table
    # ----------------------------------------------------------------
    print("=" * 100)
    print(f"{'Model':<35} {'Dataset':<22} {'Type':<4} {'Acc_S1':>7} {'Acc_S2':>7} {'Acc_S3':>7} "
          f"{'ΔS1→S2':>8} {'ΔS1→S3':>8} {'ΔS2→S3':>8} {'Abs Rate':>7} {'n':>5}")
    print("-" * 100)
    for r in sorted(rows, key=lambda x: (x["task_type"], x["model"], x["dataset"])):
        s3 = f"{r['acc_s3']:>7.1%}" if r['acc_s3'] is not None else "      —"
        d13 = f"{r['delta_s1s3']:>+8.1%}" if r['delta_s1s3'] is not None else "       —"
        d23 = f"{r['delta_s2s3']:>+8.1%}" if r['delta_s2s3'] is not None else "       —"
        print(f"{r['model']:<35} {r['dataset']:<22} {r['task_type']:<4} "
              f"{r['acc_s1']:>7.1%} {r['acc_s2']:>7.1%} {s3} "
              f"{r['delta_s1s2']:>+8.1%} {d13} {d23} "
              f"{r['abs_rate']:>7.1%} {r['n']:>5d}")

    # ----------------------------------------------------------------
    # 3. Group-level McNemar test: MCQ vs TF, per setting comparison
    # ----------------------------------------------------------------
    def _mcnemar_block(label_type, setting_label, pairs):
        if not pairs:
            return
        n11, n10, n01, n00 = mcnemar_table(pairs)
        total = n11 + n10 + n01 + n00
        a1 = (n11 + n10) / total
        a2 = (n11 + n01) / total
        delta = a2 - a1
        p = mcnemar_p(n10, n01)
        sig = ('*** (p<0.001)' if p < 0.001 else '** (p<0.01)' if p < 0.01
               else '* (p<0.05)' if p < 0.05 else '(n.s.)')
        print(f"\n  {label_type} — {setting_label}  (n={total})")
        print(f"    Acc_A={a1:.3f}  Acc_B={a2:.3f}  ΔAcc={delta:+.3f}  "
              f"discordant: +{n01} / -{n10}  p={p:.4g}  {sig}")

    print("\n" + "=" * 70)
    print("McNemar's test per setting comparison")
    print("(pooled across all model × dataset combinations within each type)")
    print("A→B means: does accuracy change from setting A to setting B?")
    print("-" * 70)
    for type_label, p12, p13, p23 in [
        ("MCQ",        mcq_pairs, mcq_pairs_13, mcq_pairs_23),
        ("TF (Judge)", tf_pairs,  tf_pairs_13,  tf_pairs_23),
    ]:
        _mcnemar_block(type_label, "S1 → S2  (+Unknown option)", p12)
        _mcnemar_block(type_label, "S1 → S3  (+Unknown +mitigation)", p13)
        _mcnemar_block(type_label, "S2 → S3  (mitigation effect only)", p23)

    # ----------------------------------------------------------------
    # 4. Spearman & Pearson correlation: Abs Rate vs ΔAcc
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Correlation: Abs Rate vs ΔAcc (across dataset × model cells)")
    print("-" * 70)

    for label, tt_filter in [("All", None), ("MCQ only", "mcq"), ("TF only", "tf")]:
        subset = rows if tt_filter is None else [r for r in rows if r["task_type"] == tt_filter]
        if len(subset) < 3:
            print(f"{label}: too few data points ({len(subset)}).")
            continue
        abs_rates   = [r["abs_rate"] for r in subset]
        deltas = [r["delta_acc"] for r in subset]
        rho, p_sp = spearman(abs_rates, deltas)
        r,   p_pe = pearson(abs_rates, deltas)
        print(f"\n{label}  (n_cells={len(subset)})")
        print(f"  Spearman ρ = {rho:+.3f}  p = {p_sp:.4g}")
        print(f"  Pearson  r = {r:+.3f}  p = {p_pe:.4g}")

    # ----------------------------------------------------------------
    # 5. Save results
    # ----------------------------------------------------------------
    out_dir = root / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "rows": rows,
        "mcq_mcnemar": None,
        "tf_mcnemar":  None,
    }
    for key, pairs in [("mcq_mcnemar", mcq_pairs), ("tf_mcnemar", tf_pairs)]:
        if pairs:
            n11, n10, n01, n00 = mcnemar_table(pairs)
            total = n11 + n10 + n01 + n00
            output[key] = {
                "n_samples": total,
                "acc_s1": (n11 + n10) / total,
                "acc_s2": (n11 + n01) / total,
                "delta_acc": ((n11 + n01) - (n11 + n10)) / total,
                "contingency": {"n11": n11, "n10": n10, "n01": n01, "n00": n00},
                "mcnemar_p": mcnemar_p(n10, n01),
            }

    out_path = out_dir / "acc_effect_by_type.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()

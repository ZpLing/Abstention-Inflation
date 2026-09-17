"""§5.3 — Difficulty stratification on FLD via proof-tree `steps`.

Hypothesis: harder problems (more reasoning steps) trigger stronger γ
(abstention inflation). MCQ difficulty stratification (ARC-Easy vs Challenge,
MedQA-Step1 vs Step2_3) showed weak effects — FLD samples come with the
`steps` field already annotated, so we use that as a finer-grained difficulty
proxy.

Inputs (no new API calls):
  - data/Judge/FLD.json — source FLD with `original_data.steps`
  - results/ab_*/ab_summary_FLD_*.json — existing S1/S2 runs

Per sample:
  - id "FLD_NNNN"  → index NNNN into source array → look up steps
  - abstained_s2 := (pred_s2 == "UNKNOWN")
  - gold_unknown := (answer_idx == 2)   # A=Proved, B=Disproved, C=Unknown
  - answerable    := not gold_unknown   # Abs Rate universe

Reports:
  1. Per-(model, step-bin) abstention rate (Abs Rate on answerable; raw rate on all).
  2. Pooled (across models) bin table.
  3. Spearman ρ between steps (continuous) and binary abstention indicator.
  4. Cochran-Armitage trend test on abstention vs step bin.

Output:
  - stdout table
  - results/analysis/fld_steps_abstention.json

Usage:
    python -m scripts.analyze_fld_steps_abstention
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FLD_SOURCE = ROOT / "data" / "Judge" / "FLD.json"
ID_RE = re.compile(r"FLD_(\d+)")

# Step bins chosen from observed FLD distribution (1..19, peaks at 7–11):
#   tiny / short / medium / long / very-long
BINS: List[Tuple[str, int, int]] = [
    ("1-3",   1,  3),
    ("4-6",   4,  6),
    ("7-9",   7,  9),
    ("10-12", 10, 12),
    ("13-15", 13, 15),
    ("16+",   16, 99),
]


def steps_to_bin(steps: int) -> str:
    for label, lo, hi in BINS:
        if lo <= steps <= hi:
            return label
    return "?"


def load_source_steps() -> Dict[int, int]:
    rows = json.loads(FLD_SOURCE.read_text())
    out: Dict[int, int] = {}
    for i, r in enumerate(rows):
        od = r.get("original_data", {})
        s = od.get("steps") or od.get("original_tree_steps")
        if s is not None:
            out[i] = int(s)
    return out


def collect_per_sample(steps_map: Dict[int, int]) -> List[dict]:
    """One row per (model, sample) — deduplicated across batches by id."""
    seen: set = set()  # (model, source_idx)
    rows: List[dict] = []
    for path in sorted(ROOT.glob("results/ab_*/ab_summary_FLD_*.json")):
        d = json.loads(path.read_text())
        model = d.get("model", "?")
        for ps in d.get("per_sample", []):
            m = ID_RE.match(ps.get("id", ""))
            if not m:
                continue
            idx = int(m.group(1))
            if idx not in steps_map:
                continue
            key = (model, idx)
            if key in seen:
                continue
            seen.add(key)
            ai = ps.get("answer_idx", -1)
            pred_s1 = ps.get("pred_s1", "")
            pred_s2 = ps.get("pred_s2", "")
            rows.append({
                "model":          model,
                "source_idx":     idx,
                "steps":          steps_map[idx],
                "answer_idx":     ai,
                "gold_unknown":   ai == 2,
                "abstained_s1":   pred_s1 == "UNKNOWN",
                "abstained_s2":   pred_s2 == "UNKNOWN",
                "source_file":    path.name,
            })
    return rows


def spearman(x: List[float], y: List[float]) -> Tuple[float, float]:
    if len(x) < 3:
        return float("nan"), float("nan")
    from scipy.stats import spearmanr
    rho, p = spearmanr(x, y)
    return float(rho), float(p)


def cochran_armitage(bins_n: List[int], bins_pos: List[int]) -> Tuple[float, float]:
    """Cochran-Armitage trend test. Scores = bin index (0..K-1).
    Returns (z, two-sided p)."""
    import math
    from scipy.stats import norm
    K = len(bins_n)
    if K < 2 or sum(bins_n) == 0:
        return float("nan"), float("nan")
    scores = list(range(K))
    N = sum(bins_n)
    R = sum(bins_pos)
    if R == 0 or R == N:
        return float("nan"), float("nan")
    p_bar = R / N
    T = sum(scores[i] * (bins_pos[i] - bins_n[i] * p_bar) for i in range(K))
    s_bar = sum(scores[i] * bins_n[i] for i in range(K)) / N
    var = p_bar * (1 - p_bar) * sum(bins_n[i] * (scores[i] - s_bar) ** 2 for i in range(K))
    if var <= 0:
        return float("nan"), float("nan")
    z = T / math.sqrt(var)
    p = 2 * (1 - norm.cdf(abs(z)))
    return float(z), float(p)


def summarise_group(rows: List[dict], universe: str) -> dict:
    """universe: 'all' (every sample) or 'answerable' (gold != UNKNOWN, Abs Rate)."""
    if universe == "answerable":
        rows = [r for r in rows if not r["gold_unknown"]]

    by_bin: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_bin[steps_to_bin(r["steps"])].append(r)

    bin_rows = []
    bins_n: List[int] = []
    bins_pos: List[int] = []
    for label, _, _ in BINS:
        rs = by_bin.get(label, [])
        n = len(rs)
        n_abs = sum(1 for r in rs if r["abstained_s2"])
        rate = (n_abs / n) if n else float("nan")
        mean_steps = (sum(r["steps"] for r in rs) / n) if n else float("nan")
        bin_rows.append({
            "bin":            label,
            "n":              n,
            "n_abstain_s2":   n_abs,
            "abstain_rate":   rate,
            "mean_steps":     mean_steps,
        })
        bins_n.append(n)
        bins_pos.append(n_abs)

    rho, p_sp = spearman([r["steps"] for r in rows], [int(r["abstained_s2"]) for r in rows])
    z, p_ca = cochran_armitage(bins_n, bins_pos)

    return {
        "universe":    universe,
        "n_total":     sum(bins_n),
        "n_abstain":   sum(bins_pos),
        "rate_total":  (sum(bins_pos) / sum(bins_n)) if sum(bins_n) else float("nan"),
        "bins":        bin_rows,
        "spearman":    {"rho": rho, "p": p_sp},
        "cochran_armitage": {"z": z, "p": p_ca},
    }


def print_table(title: str, summary: dict) -> None:
    print(f"\n{title}")
    print("-" * 78)
    print(f"  {'bin':<8} {'n':>5} {'n_absS2':>8} {'rate':>8} {'mean_steps':>11}")
    for b in summary["bins"]:
        rate = f"{b['abstain_rate']:.1%}" if b["n"] else "    —"
        ms   = f"{b['mean_steps']:.2f}" if b["n"] else "    —"
        print(f"  {b['bin']:<8} {b['n']:>5d} {b['n_abstain_s2']:>8d} {rate:>8} {ms:>11}")
    print(f"  {'TOTAL':<8} {summary['n_total']:>5d} {summary['n_abstain']:>8d} "
          f"{summary['rate_total']:>7.1%}")
    sp = summary["spearman"]
    ca = summary["cochran_armitage"]
    print(f"  Spearman ρ(steps, abstain) = {sp['rho']:+.3f}  p = {sp['p']:.4g}")
    print(f"  Cochran-Armitage trend: z = {ca['z']:+.3f}  p = {ca['p']:.4g}")


def main() -> None:
    if not FLD_SOURCE.exists():
        sys.exit(f"FLD source not found at {FLD_SOURCE}")
    steps_map = load_source_steps()
    print(f"Loaded {len(steps_map)} FLD source items with `steps` annotation.")

    rows = collect_per_sample(steps_map)
    print(f"Collected {len(rows)} (model, sample) cells from results/ab_*/ab_summary_FLD_*.json.")
    if not rows:
        sys.exit("No FLD ab_summary data found.")

    by_model: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    output: dict = {
        "bins":         [{"label": b[0], "lo": b[1], "hi": b[2]} for b in BINS],
        "n_models":     len(by_model),
        "n_cells":      len(rows),
        "per_model":    {},
        "pooled":       {},
    }

    print("\n" + "=" * 78)
    print("PER-MODEL — universe: answerable (gold ≠ UNKNOWN)  → Abs Rate by step-bin")
    print("=" * 78)
    for model in sorted(by_model):
        s_ans = summarise_group(by_model[model], universe="answerable")
        s_all = summarise_group(by_model[model], universe="all")
        output["per_model"][model] = {"answerable": s_ans, "all": s_all}
        print_table(f"[{model}]  answerable", s_ans)

    print("\n" + "=" * 78)
    print("POOLED ACROSS MODELS — universe: answerable (Abs Rate)")
    print("=" * 78)
    pooled_ans = summarise_group(rows, universe="answerable")
    pooled_all = summarise_group(rows, universe="all")
    output["pooled"] = {"answerable": pooled_ans, "all": pooled_all}
    print_table("[POOLED] answerable (Abs Rate)", pooled_ans)
    print_table("[POOLED] all samples (raw abstain rate)", pooled_all)

    out_path = ROOT / "results" / "analysis" / "fld_steps_abstention.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()

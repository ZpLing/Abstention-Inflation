"""S10 modulator (a) — temperature sweep aggregation and paired tests.

Inputs:
  T=0.0 baseline: main experiment pooled per_sample (FLD/FOLIO, deepseek)
                  — pred_s2 == "UNKNOWN" indicates abstain
  T={0.3, 0.7, 1.0, 1.5, 2.0}: results/S10_factor_analysis/temperature/<model-slug>/<DS>_<model>_T<t>.json

Outputs:
  Console (and, with --out PATH, the same summary as JSON):
  Abs Rate table, paired McNemar T=0 vs each T, Spearman ρ(T, Abs Rate)"""

import json
from math import comb, sqrt
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parents[3]
import sys as _sys

_sys.path.insert(0, str(ROOT))  # noqa: E402
from infra.result_schema import model_slug, results_dir  # noqa: E402

#: Set from --model. The sweep is reported on the two checkpoints whose
#: sampling temperature the endpoint actually applies (gemini-3.1-flash-lite
#: and Olmo-3-7B-Instruct); the gateway ignored it for the other models,
#: which is itself an S10 finding. Any swept model's cells are read the same way.
MODEL = "gemini-3.1-flash-lite"
#: Where the sweep's cells are read from; --results-root sets it.
RESULTS_ROOT = Path("results")


def _cell_path(ds, t_tag):
    return (
        ROOT
        / results_dir("S10/temperature", RESULTS_ROOT)
        / model_slug(MODEL)
        / f"{ds}_{MODEL}_{t_tag}.json"
    )


TEMPS = [0.0, 0.3, 0.7, 1.0, 1.5, 2.0]
DATASETS = ["FLD", "FOLIO"]


def load_w0_per_sample(ds):
    """T=0 baseline, taken from the sweep's own T=0.0 cell.

    Same model, same run, same items as every other temperature, so the paired
    contrast below is per item. An earlier version pooled two deepseek batches
    from a different collection round, which is a different sample.
    """
    rows = json.loads(_cell_path(ds, "T0p0").read_text()).get("per_sample", [])
    return {r["id"]: r["pred_s2"] for r in rows}


def load_temp_per_sample(t, ds):
    t_tag = f"T{t:.1f}".replace(".", "p")
    s = json.loads(_cell_path(ds, t_tag).read_text())
    return {ps["id"]: ps.get("pred_s2", ps.get("pred")) for ps in s["per_sample"]}


def mcnemar_b_c(pa, pb, ids):
    b = c = 0
    for sid in ids:
        a_unk = pa.get(sid) == "UNKNOWN"
        b_unk = pb.get(sid) == "UNKNOWN"
        if a_unk and not b_unk:
            b += 1
        elif b_unk and not a_unk:
            c += 1
    return b, c


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0, 0.0
    if n <= 25:
        k = min(b, c)
        p = 2 * sum(comb(n, i) * 0.5**n for i in range(k + 1))
        return min(p, 1.0), (b - c) / sqrt(b + c) if (b + c) > 0 else 0
    z = (abs(b - c) - 1) / sqrt(b + c)
    p = 2 * (1 - NormalDist().cdf(abs(z)))
    return p, (c - b) / sqrt(b + c)


def spearman_rho(xs, ys):
    """Hand-rolled Spearman ρ for 4-point series."""

    def rank(values):
        s = sorted(range(len(values)), key=lambda i: values[i])
        r = [0] * len(values)
        for ri, i in enumerate(s):
            r[i] = ri + 1
        return r

    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    rho = 1 - 6 * d2 / (n * (n * n - 1))
    return rho


def main(out_path=None):
    print(f"=== S10 temperature sweep analysis ({MODEL}) ===\n")

    cells = {}  # (ds, t) -> {n, n_ai, abs_rate, acc, pred_by_id}

    for ds in DATASETS:
        # Get sample IDs from any T>0 cell (they share IDs)
        t_tag = f"T{0.3:.1f}".replace(".", "p")
        ref = json.loads(_cell_path(ds, t_tag).read_text())
        sample_ids = [ps["id"] for ps in ref["per_sample"]]
        # answer_idx map
        ans_by_id = {ps["id"]: ps["answer_idx"] for ps in ref["per_sample"]}

        # T=0
        w0 = load_w0_per_sample(ds)
        w0_aligned = {sid: w0[sid] for sid in sample_ids if sid in w0}
        n0 = len(w0_aligned)
        n_ai_0 = sum(1 for v in w0_aligned.values() if v == "UNKNOWN")
        # Acc on T=0: count pred==A if answer_idx==0, pred==B if answer_idx==1
        acc_correct_0 = sum(
            1
            for sid in sample_ids
            if (ans_by_id[sid] == 0 and w0.get(sid) == "A")
            or (ans_by_id[sid] == 1 and w0.get(sid) == "B")
        )
        cells[(ds, 0.0)] = {
            "n": n0,
            "n_abstention_inflation": n_ai_0,
            "abs_rate": n_ai_0 / n0 if n0 else 0,
            "acc": acc_correct_0 / n0 if n0 else 0,
            "pred_by_id": w0_aligned,
        }

        for t in [0.3, 0.7, 1.0, 1.5, 2.0]:
            pred = load_temp_per_sample(t, ds)
            pred_aligned = {sid: pred[sid] for sid in sample_ids if sid in pred}
            n = len(pred_aligned)
            n_ai = sum(1 for v in pred_aligned.values() if v == "UNKNOWN")
            acc_correct = sum(
                1
                for sid in sample_ids
                if (ans_by_id[sid] == 0 and pred_aligned.get(sid) == "A")
                or (ans_by_id[sid] == 1 and pred_aligned.get(sid) == "B")
            )
            cells[(ds, t)] = {
                "n": n,
                "n_abstention_inflation": n_ai,
                "abs_rate": n_ai / n if n else 0,
                "acc": acc_correct / n if n else 0,
                "pred_by_id": pred_aligned,
            }

    # Abs Rate table
    print(
        f"{'Dataset':8s} {'T':>4s} {'n':>4s} {'n_ai':>6s} {'Abs Rate':>7s} {'Acc':>7s}"
    )
    print("-" * 45)
    for ds in DATASETS:
        for t in TEMPS:
            c = cells[(ds, t)]
            print(
                f"{ds:8s} {t:>4.1f} {c['n']:>4} {c['n_abstention_inflation']:>6} {c['abs_rate']:>7.1%} {c['acc']:>7.1%}"
            )
        print()

    # Paired McNemar T=0 vs each T
    print("=== Paired McNemar (T=0 vs each T, same items) ===\n")
    print(
        f"{'Dataset':8s} {'T_a':>4s} {'vs':>3s} {'T_b':>4s} {'b':>4s} {'c':>4s} {'Δ_AIR':>7s} {'p':>10s}"
    )
    print("-" * 60)
    mcn_records = []
    for ds in DATASETS:
        c0 = cells[(ds, 0.0)]
        ids = list(c0["pred_by_id"].keys())
        for t in [0.3, 0.7, 1.0, 1.5, 2.0]:
            ct = cells[(ds, t)]
            b, c = mcnemar_b_c(c0["pred_by_id"], ct["pred_by_id"], ids)
            p, z = mcnemar_p(b, c)
            delta = ct["abs_rate"] - c0["abs_rate"]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(
                f"{ds:8s} {0.0:>4.1f} {'vs':>3s} {t:>4.1f} {b:>4} {c:>4} {delta:>+6.1%} {p:>10.4f} {sig}"
            )
            mcn_records.append(
                {"ds": ds, "T": t, "b": b, "c": c, "delta": delta, "p": p, "z": z}
            )
        print()

    # Spearman ρ(T, Abs Rate) per dataset
    print("=== Spearman ρ(T, Abs Rate) per dataset ===\n")
    sp_records = {}
    for ds in DATASETS:
        abs_rates = [cells[(ds, t)]["abs_rate"] for t in TEMPS]
        rho = spearman_rho(TEMPS, abs_rates)
        rng = max(abs_rates) - min(abs_rates)
        print(
            f"{ds}: AbsRates={[round(a * 100, 1) for a in abs_rates]}  range={rng * 100:.1f}pp  ρ={rho:+.3f}"
        )
        sp_records[ds] = {"abs_rates": abs_rates, "rho": rho, "range_pp": rng * 100}

    # Save
    out = {
        "model": MODEL,
        "cells": {
            f"{ds}_T{t}": {k: v for k, v in cells[(ds, t)].items() if k != "pred_by_id"}
            for ds in DATASETS
            for t in TEMPS
        },
        "mcnemar_T0_vs_Tt": mcn_records,
        "spearman_per_ds": sp_records,
    }
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        print(f"\nWrote {out_path}")


def _cli():
    import argparse

    global MODEL, RESULTS_ROOT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        default=MODEL,
        help="Which swept model to analyse (gemini-3.1-flash-lite or "
        "Olmo-3-7B-Instruct are the reported ones).",
    )
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--out", default=None, help="Also write the summary as JSON here.")
    args = ap.parse_args()
    MODEL = args.model
    RESULTS_ROOT = Path(args.results_root)
    return args


if __name__ == "__main__":
    main(_cli().out)

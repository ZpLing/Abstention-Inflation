"""Acc-F1T-NLI triangle dissociation (sample-level joint analysis).

For each Abstention Inflation sample (S2 == UNKNOWN on answerable subset), check whether all
three γ-signals co-occur within the same sample:

    Acc_S2 = 0          (label flipped to UNKNOWN — Abstention Inflation sample by definition)
    F1T_S2 ≥ F1T_S1     (trace structure not degraded; paired within-subject)
    NLI_S2 decisive     (CoT content still PROVED/DISPROVED, not NEUTRAL)

Triangle-decoupling rate = fraction of Abstention Inflation samples satisfying all three.
A stricter "gold-aligned" variant requires NLI_S2.derived == gold_proof_label.

Reuses:
    - results/ab*/ab_summary_<DS>_<model>.json    (per_sample raw_s1/s2)
    - results/probe/cot_tasksource_<DS>_<tag>_n200.json  (per-sample NLI)
    - core.trace_extractors / metrics  (F1T per-sample)
"""
import json
import sys
from glob import glob
from pathlib import Path
from typing import Dict, List

ROOT = Path(".")
sys.path.insert(0, str(ROOT))

from core import metrics, trace_extractors           # noqa: E402
from core.dataset_loader import load_judge           # noqa: E402
from core.evaluator import Evaluator                               # noqa: E402

MODELS = [
    ("deepseek", "deepseek-r1-distill-llama-8b",
        {"FLD":   ["ab_e_option_baseline", "ab_deepseek_batch2"],
         "FOLIO": ["ab_followup",         "ab_deepseek_batch2"]}),
    ("nano",     "gpt-5.4-nano",
        {"FLD":   ["ab_gpt5_nano",         "ab_nano_batch2"],
         "FOLIO": ["ab_gpt5_nano",         "ab_nano_batch2"]}),
    ("gemini",   "gemini-3.1-flash-lite",
        {"FLD":   ["ab_gemini_flash_lite", "ab_gemini_batch2"],
         "FOLIO": ["ab_gemini_flash_lite", "ab_gemini_batch2"]}),
]
DATASETS = ["FLD", "FOLIO"]


def per_sample_f1t_fld(samples_by_id, ev, ai_records):
    """FLD set-F1 per sample (paired S1, S2)."""
    out = {}
    for ps in ai_records:
        s = samples_by_id.get(ps["id"])
        if s is None:
            continue
        gold = trace_extractors.fld_gold_atoms(s)
        r1 = ev.extract_reasoning(ps["raw_s1"])
        r2 = ev.extract_reasoning(ps["raw_s2"])
        p1 = trace_extractors.fld_pred_atoms(r1, s)
        p2 = trace_extractors.fld_pred_atoms(r2, s)
        _, _, f1_s1 = metrics.trace_set_f1(p1, gold)
        _, _, f1_s2 = metrics.trace_set_f1(p2, gold)
        out[ps["id"]] = (f1_s1, f1_s2)
    return out


def per_sample_f1t_folio(samples_by_id, ev, ai_records):
    """FOLIO BERTScore per sample, batched once across all pairs."""
    cands, refs, ids, which = [], [], [], []
    for ps in ai_records:
        s = samples_by_id.get(ps["id"])
        if s is None:
            continue
        gold_text = trace_extractors.folio_gold_text(s)
        for setting_key in ("raw_s1", "raw_s2"):
            r = ev.extract_reasoning(ps[setting_key])
            pred_text = trace_extractors.folio_pred_text(r, s)
            cands.append(pred_text or "")
            refs.append(gold_text or "")
            ids.append(ps["id"])
            which.append(setting_key)
    # Drop empty pairs (BERTScore would error or return junk)
    keep = [i for i, (c, g) in enumerate(zip(cands, refs)) if c and g]
    cands_k = [cands[i] for i in keep]
    refs_k  = [refs[i]  for i in keep]
    print(f"    [FOLIO] BERTScore on {len(cands_k)} (cand, ref) pairs ...")
    scores = metrics._bertscore_pair(cands_k, refs_k) if cands_k else []
    score_map = {}
    for j, idx in enumerate(keep):
        score_map[(ids[idx], which[idx])] = scores[j]

    out = {}
    seen = set()
    for ps in ai_records:
        sid = ps["id"]
        if sid in seen:
            continue
        seen.add(sid)
        f1_s1 = score_map.get((sid, "raw_s1"), 0.0)
        f1_s2 = score_map.get((sid, "raw_s2"), 0.0)
        out[sid] = (f1_s1, f1_s2)
    return out


def main():
    ev = Evaluator()

    # Pre-load source samples for gold-trace lookup.
    samples_by_id = {}
    gold_label_by_id = {}
    for ds in DATASETS:
        for s in load_judge(ds):
            samples_by_id[s.id] = s
            gold_label_by_id[s.id] = (s.extra or {}).get("native_label")

    summary = {"cells": {}, "pooled": {}}

    pooled_total = 0
    pooled_intact = 0
    pooled_decisive = 0
    pooled_triangle = 0
    pooled_triangle_gold = 0

    for tag, mdl, ab_dirs_per_ds in MODELS:
        for ds in DATASETS:
            # Pool per_sample across batch1 + batch2 so we match the NLI probe
            # input (which used both batches concatenated).
            pooled_per_sample = []
            seen_ids = set()
            for ab_dir in ab_dirs_per_ds.get(ds, []):
                ab_paths = glob(str(ROOT / f"results/{ab_dir}/ab_summary_{ds}_*.json"))
                if not ab_paths:
                    continue
                ab_one = json.load(open(ab_paths[0]))
                for ps in ab_one.get("per_sample", []):
                    if ps["id"] in seen_ids:
                        continue
                    seen_ids.add(ps["id"])
                    pooled_per_sample.append(ps)
            if not pooled_per_sample:
                print(f"[skip] no per_sample for {tag} {ds}")
                continue
            ab = {"per_sample": pooled_per_sample}

            nli_path = ROOT / f"results/probe/cot_tasksource_{ds}_{tag}_n200.json"
            if not nli_path.exists():
                print(f"[skip] no NLI probe for {tag} {ds}")
                continue
            nli = json.load(open(nli_path))
            nli_s1 = {r["id"]: r for r in nli["rows"] if r["setting"] == "S1"}
            nli_s2 = {r["id"]: r for r in nli["rows"] if r["setting"] == "S2"}

            # Abstention Inflation samples in this cell
            ai = [ps for ps in ab["per_sample"] if ps["pred_s2"] == "UNKNOWN"]
            print(f"[cell] {tag} {ds}  n_ai={len(ai)}")

            if ds == "FLD":
                f1t_map = per_sample_f1t_fld(samples_by_id, ev, abs_rate)
            else:
                f1t_map = per_sample_f1t_folio(samples_by_id, ev, abs_rate)

            rows = []
            for ps in abs_rate:
                sid = ps["id"]
                f1_s1, f1_s2 = f1t_map.get(sid, (0.0, 0.0))
                n2 = nli_s2.get(sid, {}).get("derived", "UNK")
                n1 = nli_s1.get(sid, {}).get("derived", "UNK")
                gold_label = gold_label_by_id.get(sid, "")
                # Strip __PROVED__ / __DISPROVED__ to PROVED / DISPROVED
                gold_short = gold_label.strip("_")

                f1_intact = f1_s2 >= f1_s1
                decisive_s2 = n2 in ("PROVED", "DISPROVED")
                decisive_s1 = n1 in ("PROVED", "DISPROVED")
                triangle = f1_intact and decisive_s2
                triangle_gold = triangle and (n2 == gold_short)

                rows.append({
                    "id": sid,
                    "f1t_s1": round(f1_s1, 4),
                    "f1t_s2": round(f1_s2, 4),
                    "f1t_intact": f1_intact,
                    "nli_s1": n1,
                    "nli_s2": n2,
                    "decisive_s1": decisive_s1,
                    "decisive_s2": decisive_s2,
                    "gold": gold_short,
                    "triangle_decouple": triangle,
                    "triangle_gold_aligned": triangle_gold,
                })

            n = len(rows)
            n_intact = sum(1 for r in rows if r["f1t_intact"])
            n_dec = sum(1 for r in rows if r["decisive_s2"])
            n_tri = sum(1 for r in rows if r["triangle_decouple"])
            n_tri_gold = sum(1 for r in rows if r["triangle_gold_aligned"])

            summary["cells"][f"{tag}_{ds}"] = {
                "model": mdl, "dataset": ds, "tag": tag,
                "n_ai": n,
                "n_f1t_intact": n_intact,
                "n_decisive_s2": n_dec,
                "n_triangle": n_tri,
                "n_triangle_gold": n_tri_gold,
                "rate_f1t_intact": (n_intact / n) if n else 0.0,
                "rate_decisive": (n_dec / n) if n else 0.0,
                "rate_triangle": (n_tri / n) if n else 0.0,
                "rate_triangle_gold": (n_tri_gold / n) if n else 0.0,
                "rows": rows,
            }
            pooled_total += n
            pooled_intact += n_intact
            pooled_decisive += n_dec
            pooled_triangle += n_tri
            pooled_triangle_gold += n_tri_gold

    summary["pooled"] = {
        "n_ai": pooled_total,
        "rate_f1t_intact":     pooled_intact / pooled_total     if pooled_total else 0.0,
        "rate_decisive":       pooled_decisive / pooled_total   if pooled_total else 0.0,
        "rate_triangle":       pooled_triangle / pooled_total   if pooled_total else 0.0,
        "rate_triangle_gold":  pooled_triangle_gold / pooled_total if pooled_total else 0.0,
    }

    out_path = ROOT / "results/analysis/acc_f1t_nli_triangle.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")

    # Pretty table
    print()
    print(f"{'cell':18s} {'n_ai':>5} {'F1Tintact':>10} {'NLIdec':>8} {'TRI':>7} {'TRI+gold':>9}")
    print("-" * 64)
    for k, c in summary["cells"].items():
        print(f"{k:18s} {c['n_ai']:>5} "
              f"{c['rate_f1t_intact']:>9.1%} "
              f"{c['rate_decisive']:>7.1%} "
              f"{c['rate_triangle']:>6.1%} "
              f"{c['rate_triangle_gold']:>8.1%}")
    p = summary["pooled"]
    print("-" * 64)
    print(f"{'POOLED':18s} {p['n_ai']:>5} "
          f"{p['rate_f1t_intact']:>9.1%} "
          f"{p['rate_decisive']:>7.1%} "
          f"{p['rate_triangle']:>6.1%} "
          f"{p['rate_triangle_gold']:>8.1%}")


if __name__ == "__main__":
    main()

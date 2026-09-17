"""Exp 5 / R2 — Contrastive Self-Consistency Decoding (post-hoc, zero API calls).

R2 is derived from Theorem 2(1) (mid-layer fidelity): when the model abstains
under Y^+e = Y ∪ {Unknown}, query it again under Y (no escape option). If the
no-escape query returns a non-abstain answer, override the abstention.

This script consumes existing ab_summary_<ds>_<model>.json files (which already
contain pred_s1 and pred_s2 for every per-sample row) and applies the override
purely in post-processing. Zero new API calls.

Override rule (per sample):
    if pred_s2 == 'UNKNOWN' and pred_s1 in {A, B, C, D}:
        final = pred_s1            # R2 fires
    else:
        final = pred_s2            # keep S2

Outputs to results/remedy_r2/r2_summary_<ds>_<model>.json:
    {
      "model": ..., "dataset": ..., "n_total": ...,
      "n_r2_fired": <# overrides applied>,
      "metrics": {
        "S1": {label_acc},
        "S2": {label_acc, abs_rate},
        "R2": {label_acc, abs_rate, delta_acc_vs_s2, delta_abs_rate_vs_s2}
      },
      "per_sample": [{id, gold, s1, s2, r2, r2_fired}]
    }
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from core.result_schema import paired_keep_ids
from typing import Any, Dict, List, Tuple

# Gold answer_idx convention from ab_runner: 0->A, 1->B, 2->C, 3->D.
IDX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}
ABSTAIN = "UNKNOWN"
UNPARSE = "UNPARSEABLE"
VALID_LETTERS = {"A", "B", "C", "D"}


def apply_r2(per_sample: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for s in per_sample:
        s1, s2 = s["pred_s1"], s["pred_s2"]
        fired = s2 == ABSTAIN and s1 in VALID_LETTERS
        r2 = s1 if fired else s2
        out.append({
            "id": s["id"],
            "gold": IDX_TO_LETTER.get(s["answer_idx"]),
            "s1": s1,
            "s2": s2,
            "r2": r2,
            "r2_fired": fired,
        })
    return out


def metrics(per: List[Dict[str, Any]], key: str) -> Dict[str, float]:
    n = len(per)
    if n == 0:
        return {"label_acc": 0.0, "abs_rate": 0.0, "n": 0}
    n_acc = sum(1 for s in per if s[key] == s["gold"])
    n_abs = sum(1 for s in per if s[key] == ABSTAIN)
    return {"label_acc": n_acc / n, "abs_rate": n_abs / n, "n": n}


def process_one(summary_path: Path, out_dir: Path) -> Dict[str, Any]:
    raw = json.loads(summary_path.read_text())
    # The same paired keep-set the main table is scored on, so R2's S1 and S2
    # columns reproduce it rather than quoting a second denominator.
    keep = paired_keep_ids(raw)
    per_sample = [r for r in raw["per_sample"] if r["id"] in keep]
    r2_rows = apply_r2(per_sample)

    m_s1 = metrics(r2_rows, "s1")
    m_s2 = metrics(r2_rows, "s2")
    m_r2 = metrics(r2_rows, "r2")
    n_fired = sum(1 for r in r2_rows if r["r2_fired"])

    result = {
        "model": raw.get("model"),
        "dataset": raw.get("dataset"),
        "task_type": raw.get("task_type"),
        "n_total": raw.get("n_total"),
        "n_scored": len(r2_rows),
        "n_r2_fired": n_fired,
        "metrics": {
            "S1": {"label_acc": m_s1["label_acc"]},
            "S2": {"label_acc": m_s2["label_acc"], "abs_rate": m_s2["abs_rate"]},
            "R2": {
                "label_acc": m_r2["label_acc"],
                "abs_rate": m_r2["abs_rate"],
                "delta_acc_vs_s2": m_r2["label_acc"] - m_s2["label_acc"],
                "delta_abs_rate_vs_s2": m_r2["abs_rate"] - m_s2["abs_rate"],
                "delta_acc_vs_s1": m_r2["label_acc"] - m_s1["label_acc"],
            },
        },
        "per_sample": r2_rows,
        "source_summary": str(summary_path),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    ds = raw.get("dataset", summary_path.stem)
    model = raw.get("model", "unknown").replace("/", "_")
    out_path = out_dir / f"r2_summary_{ds}_{model}.json"
    out_path.write_text(json.dumps(result, indent=2))
    return result


#: Directories whose summaries carry a different schema or a superseded run.
_SKIP_DIRS = {"ab_followup", "ab_s5", "ab_e_option_baseline", "ab_mcq_extended"}

#: R2 needs a paired S1/S2 cell. The sweeps vary one knob on the S2 prompt and
#: never record an S1 side, so including them would report an S1 accuracy of
#: zero and an override that cannot fire. The TFQ cells are read from the same
#: paired summaries as the MCQ cells; the S11 slot-C file reproduces the S2
#: prompt but is a separate run, so pairing it with a separate S1 sweep would
#: report R2 on numbers that are not the ones the main table states.
_SKIP_PREFIXES = ("s10_", "temperature_sweep", "wording_sweep", "gemma_",
                  "qwen_", "_smoke", "probe", "positional_bias")


def find_summaries(results_root: Path) -> List[Path]:
    """Every paired ab_summary under results/, whatever it is nested in.

    The MCQ cells moved to `results/mcq/<dataset>_<model>/`, so matching
    on a top-level directory named `ab*` would silently skip all of them; a
    plain recursive glob goes too far the other way and sweeps in the
    single-condition runs, so a summary has to actually carry both sides.
    """
    out = []
    for p in sorted(results_root.rglob("ab_summary_*.json")):
        parts = set(p.parts)
        if _SKIP_DIRS & parts:
            continue
        if any(part.startswith(_SKIP_PREFIXES) for part in p.parts):
            continue
        try:
            rows = json.loads(p.read_text()).get("per_sample") or []
        except (ValueError, OSError):
            continue
        if any("pred_s1" in r and "pred_s2" in r for r in rows[:5]):
            out.append(p)
    return out




def pretty_table(rows: List[Dict[str, Any]]) -> str:
    header = f"{'Model':<32} {'Dataset':<24} {'Task':<5} {'S1 Acc':>8} {'S2 Acc':>8} {'S2 Abs':>8} {'R2 Acc':>8} {'R2 Abs':>8} {'ΔAcc':>8} {'ΔAbs':>8}"
    lines = [header, "-" * len(header)]
    for r in rows:
        m = r["metrics"]
        lines.append(
            f"{(r['model'] or '?')[:32]:<32} "
            f"{(r['dataset'] or '?')[:24]:<24} "
            f"{(r['task_type'] or '?')[:5]:<5} "
            f"{m['S1']['label_acc']*100:>7.1f}% "
            f"{m['S2']['label_acc']*100:>7.1f}% "
            f"{m['S2']['abs_rate']*100:>7.1f}% "
            f"{m['R2']['label_acc']*100:>7.1f}% "
            f"{m['R2']['abs_rate']*100:>7.1f}% "
            f"{m['R2']['delta_acc_vs_s2']*100:>+7.1f} "
            f"{m['R2']['delta_abs_rate_vs_s2']*100:>+7.1f}"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default="results")
    ap.add_argument("--out_dir", default="results/remedy_r2")
    args = ap.parse_args()

    root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    summaries = find_summaries(root)
    print(f"Found {len(summaries)} ab_summary files")

    rows = []
    for p in summaries:
        try:
            row = process_one(p, out_dir)
            rows.append(row)
        except Exception as e:
            print(f"  SKIP {p.name}: {e}")

    rows.sort(key=lambda r: ((r.get("model") or ""), (r.get("dataset") or "")))
    table = pretty_table(rows)
    print("\n" + table)

    (out_dir / "r2_overview.txt").write_text(table + "\n")
    (out_dir / "r2_overview.json").write_text(
        json.dumps([{k: v for k, v in r.items() if k != "per_sample"} for r in rows], indent=2)
    )
    print(f"\nWrote {len(rows)} summaries to {out_dir}/")


if __name__ == "__main__":
    main()

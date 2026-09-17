"""Exp 3 (paper §3 Exp 3) — Difficulty layered analysis on ARC.

Reads the per-dataset summaries that ABRunner already saved for ARC-Easy_250
and ARC-Challenge_250, and produces three result sets:

    easy       — label metrics on ARC-Easy alone.
    challenge  — same, on ARC-Challenge alone.
    total      — label metrics on the union (the "ARC" aggregate point).

Trace metrics are dataset-specific and require sample reload (because raw
WorldTree atoms aren't serialized into the summary). Exp 3's contribution is
the difficulty contrast on label-level abstention behavior, so it reports only
label_acc / label_f1 / abstention_rate per setting. Run scripts/exp3_trace.py
(future) for trace-level breakdown.

Run:
    python -m core.s10_difficulty_runner \\
        --easy      results/ab/ab_summary_ARC-Easy_250_<model>.json \\
        --challenge results/ab/ab_summary_ARC-Challenge_250_<model>.json \\
        [--out      results/ab/exp3_difficulty_<model>.json]
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

from . import metrics
from .result_schema import get_field


# ---------------------------------------------------------------------------
# Re-derive label metrics from a saved summary (or a synthetic combined one)
# ---------------------------------------------------------------------------

def _label_metrics_from_arrays(per_sample: List[dict], s5_rerun: List[dict],
                                 task_type: str = "mcq") -> Dict:
    preds_s1 = [r["pred_s1"] for r in per_sample]
    preds_s2 = [r["pred_s2"] for r in per_sample]
    preds_s3 = [r["pred_s3"] for r in per_sample]
    answer_idxs = [r["answer_idx"] for r in per_sample]

    preds_s4 = [r.get("pred_s4") for r in s5_rerun if r.get("pred_s4") is not None]
    ai_answer_idxs = [r["answer_idx"] for r in s5_rerun]

    if task_type == "mcq":
        classes_no_unk = metrics.mcq_classes(with_unknown=False)
        classes_with_unk = metrics.mcq_classes(with_unknown=True)
    else:
        classes_no_unk = metrics.judge_classes(with_unknown=False)
        classes_with_unk = metrics.judge_classes(with_unknown=True)

    return {
        "n_total":  len(per_sample),
        "n_abstention_inflation": len(s5_rerun),
        "metrics": {
            "S1": {
                "label_acc": metrics.label_acc(preds_s1, answer_idxs),
                "label_f1":  metrics.label_macro_f1(preds_s1, answer_idxs, classes_no_unk),
                "abst_rate": metrics.abs_rate(preds_s1),  # 0 by design
            },
            "S2": {
                "label_acc": metrics.label_acc(preds_s2, answer_idxs),
                "label_f1":  metrics.label_macro_f1(preds_s2, answer_idxs, classes_with_unk),
                "abst_rate": metrics.abs_rate(preds_s2),
            },
            "S3": {
                "label_acc": metrics.label_acc(preds_s3, answer_idxs),
                "label_f1":  metrics.label_macro_f1(preds_s3, answer_idxs, classes_with_unk),
                "abst_rate": metrics.abs_rate(preds_s3),
            },
            "S4": {
                "label_acc": metrics.label_acc(preds_s4, ai_answer_idxs) if preds_s4 else 0.0,
                "label_f1":  (metrics.label_macro_f1(preds_s4, ai_answer_idxs, classes_no_unk)
                              if preds_s4 else 0.0),
                "n_evaluated": len(s5_rerun),
            },
        },
    }


def _metrics_from_summary(summary: dict) -> Dict:
    return _label_metrics_from_arrays(summary["per_sample"],
                                        get_field(summary, "s5_rerun", []),
                                        task_type=summary.get("task_type", "mcq"))


def _combined_metrics(easy: dict, challenge: dict) -> Dict:
    return _label_metrics_from_arrays(
        easy["per_sample"] + challenge["per_sample"],
        get_field(easy, "s5_rerun", []) + get_field(challenge, "s5_rerun", []),
        task_type=easy.get("task_type", "mcq"),
    )


# ---------------------------------------------------------------------------
# Top-level report
# ---------------------------------------------------------------------------

def build_exp3_report(easy_path: Path, challenge_path: Path) -> dict:
    easy = json.loads(Path(easy_path).read_text())
    challenge = json.loads(Path(challenge_path).read_text())
    return {
        "model": easy.get("model"),
        "judge_model": easy.get("judge_model"),
        "datasets": {
            "easy":      easy["dataset"],
            "challenge": challenge["dataset"],
        },
        "easy":      _metrics_from_summary(easy),
        "challenge": _metrics_from_summary(challenge),
        "total":     _combined_metrics(easy, challenge),
    }


def _print_summary(report: dict) -> None:
    print("\n=== Exp 3 difficulty layered analysis ===")
    print(f"  model: {report['model']}")
    print(f"  Easy:      {report['datasets']['easy']}   ({report['easy']['n_total']} samples)")
    print(f"  Challenge: {report['datasets']['challenge']}   ({report['challenge']['n_total']} samples)")
    print(f"  Total:     ARC (combined)                   ({report['total']['n_total']} samples)")
    print()
    fmt = "  {:<22s}  {:>10s}  {:>10s}  {:>10s}"
    print(fmt.format("metric", "Easy", "Challenge", "Total"))
    print(fmt.format("-" * 22, "-" * 10, "-" * 10, "-" * 10))
    for setting in ["S1", "S2", "S3", "S4"]:
        for key in ["label_acc", "label_f1", "abst_rate"]:
            e = report["easy"]["metrics"][setting].get(key)
            c = report["challenge"]["metrics"][setting].get(key)
            t = report["total"]["metrics"][setting].get(key)
            if e is None:
                continue
            print(fmt.format(f"{setting}.{key}", f"{e:.2%}", f"{c:.2%}", f"{t:.2%}"))


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--easy",      required=True, help="Path to ab_summary_ARC-Easy_250_<model>.json")
    p.add_argument("--challenge", required=True, help="Path to ab_summary_ARC-Challenge_250_<model>.json")
    p.add_argument("--out",       default=None,  help="Optional output JSON path")
    args = p.parse_args()

    report = build_exp3_report(args.easy, args.challenge)

    out_path = Path(args.out) if args.out else (
        Path(args.easy).parent / f"exp3_difficulty_{report['model']}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote {out_path}")
    _print_summary(report)


if __name__ == "__main__":
    main()

"""
C3 Suppression Detection: classify UNKNOWN outputs as suppressed (Abs Rate) vs genuine (CAR).

Hypothesis: alignment-suppressed UNKNOWN has a "flip layer" — early layers prefer
the gold answer (logit_gap < 0), late-layer alignment pressure pushes UNKNOWN to top
(logit_gap > 0). Genuine UNKNOWN (CAR) lacks this pattern: UNKNOWN is preferred from
the start, or the gold-competing label is never strongly preferred.

Method:
  For Abstention Inflation samples: compute early-layer logit_gap (layers 1-8) and flip_layer.
  For CAR samples: compute early-layer rank_unknown (since gold_id == unknown_id,
    logit_gap is trivially 0; use rank_unknown as the signal instead).
  Then build a simple threshold classifier and report AUC + accuracy.

Input:  results/c3/logit_lens_instruct.json  (or any checkpoint)
Output: results/c3/suppression_detect_{ckpt}.json
        figures/c3_suppression_detect.pdf

Run:
    python scripts/c3_suppression_detect.py --ckpt instruct
    python scripts/c3_suppression_detect.py --ckpt rl_zero
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from core.result_schema import canonical_sample_type

def _normalize_sample_types(records):
    """Map pre-unification ``sample_type`` values ("AIR"/"non_AIR") onto the
    canonical ``"ai"``/``"non_ai"`` used since the paper-terminology rename, so
    result files written before it still load. Mutates and returns ``records``."""
    for r in records:
        if "sample_type" in r:
            r["sample_type"] = canonical_sample_type(r["sample_type"])
    return records


plt.rcParams.update({
    "font.size": 11,
    "font.weight": "bold",
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EARLY_LAYERS = slice(1, 9)    # layers 1-8 (skip embedding layer 0)
LATE_LAYERS  = slice(-4, None) # last 4 transformer layers

COLORS = {
    "abs_rate":     "#d62728",   # red
    "CAR":     "#1f77b4",   # blue
    "non_ai": "#ff7f0e",
    "non_CAR": "#aec7e8",
}


def load_data(ckpt: str) -> list[dict]:
    path = ROOT / "results" / "c3" / f"logit_lens_{ckpt}.json"
    raw = json.loads(path.read_text())
    return _normalize_sample_types(raw["per_sample"])


def early_logit_gap(sample: dict) -> float:
    """Mean logit_gap over early layers. <0 means gold preferred early."""
    gaps = [l["logit_gap"] for l in sample["layers"][EARLY_LAYERS]]
    return float(np.mean(gaps))


def flip_layer(sample: dict) -> int:
    """First layer where logit_gap >= 0 (UNKNOWN overtakes gold). Returns -1 if never."""
    for layer in sample["layers"]:
        if layer["logit_gap"] >= 0:
            return layer["layer"]
    return -1


def early_rank_unknown(sample: dict) -> float:
    """Mean rank(UNKNOWN) over early layers. Lower = UNKNOWN more preferred."""
    ranks = [l["rank_unknown"] for l in sample["layers"][EARLY_LAYERS]]
    return float(np.mean(ranks))


def final_rank_unknown(sample: dict) -> float:
    ranks = [l["rank_unknown"] for l in sample["layers"][LATE_LAYERS]]
    return float(np.mean(ranks))


def compute_auc_threshold(pos_scores, neg_scores):
    """Simple AUC via threshold sweep. pos=Abs Rate (we want low early_gap to detect them)."""
    labels = [1] * len(pos_scores) + [0] * len(neg_scores)
    scores = list(pos_scores) + list(neg_scores)
    thresholds = sorted(set(scores))
    best_acc = 0.0
    best_thr = 0.0
    tprs, fprs = [], []
    for thr in thresholds:
        preds = [1 if s <= thr else 0 for s in scores]
        tp = sum(p == 1 and l == 1 for p, l in zip(preds, labels))
        fp = sum(p == 1 and l == 0 for p, l in zip(preds, labels))
        fn = sum(p == 0 and l == 1 for p, l in zip(preds, labels))
        tn = sum(p == 0 and l == 0 for p, l in zip(preds, labels))
        tprs.append(tp / (tp + fn + 1e-9))
        fprs.append(fp / (fp + tn + 1e-9))
        acc = (tp + tn) / len(labels)
        if acc > best_acc:
            best_acc = acc
            best_thr = thr
    # AUC via trapezoidal rule
    sorted_pairs = sorted(zip(fprs, tprs))
    xs = [p[0] for p in sorted_pairs]
    ys = [p[1] for p in sorted_pairs]
    auc = float(np.trapz(ys, xs))
    return abs(auc), best_acc, best_thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="instruct",
                    choices=["base", "instruct", "rl_zero"])
    args = ap.parse_args()

    samples = load_data(args.ckpt)
    print(f"Loaded {len(samples)} samples from {args.ckpt}")

    ai     = [s for s in samples if s["sample_type"] == "ai"]
    car    = [s for s in samples if s["sample_type"] == "CAR"]
    non_ai = [s for s in samples if s["sample_type"] == "non_ai"]
    noncar = [s for s in samples if s["sample_type"] == "non_CAR"]
    print(f"  n_ai={len(ai)}  non_ai={len(non_ai)}  CAR={len(car)}  non_CAR={len(noncar)}")

    # ------------------------------------------------------------------
    # Feature 1: early-layer logit_gap (for Abs Rate vs non_ai, answerable only)
    # ------------------------------------------------------------------
    ai_early    = [early_logit_gap(s) for s in abs_rate]
    non_ai_early = [early_logit_gap(s) for s in non_ai]

    # Abs Rate should have more negative early_gap (gold preferred early),
    # but because both group models predict correctly in non-Abstention-Inflation (gold > UNKNOWN),
    # the separation comes from HOW MUCH early preference exists.
    # Flip-layer: Abs Rate has finite flip (UNKNOWN wins late), non-Abstention-Inflation has no flip.
    ai_flip    = [flip_layer(s) for s in abs_rate]
    non_ai_flip = [flip_layer(s) for s in non_ai]

    # Count flip rates
    ai_flip_rate    = sum(1 for f in ai_flip    if f >= 0) / max(len(ai_flip), 1)
    non_ai_flip_rate = sum(1 for f in non_ai_flip if f >= 0) / max(len(non_ai_flip), 1)
    print(f"\n[Abs Rate vs non-Abstention-Inflation] answerable samples:")
    print(f"  Abs Rate    early_gap  mean={np.mean(ai_early):.3f}  flip_rate={ai_flip_rate:.2%}")
    print(f"  non_ai early_gap mean={np.mean(non_ai_early):.3f}  flip_rate={non_ai_flip_rate:.2%}")
    print(f"  Abs Rate    flip_layer (median, among those that flip): "
          f"{np.median([f for f in ai_flip if f>=0]):.0f}" if any(f>=0 for f in ai_flip) else "  (none flipped)")

    # ------------------------------------------------------------------
    # Feature 2: early-layer rank_unknown (for CAR vs non_CAR, unknown-gold only)
    # ------------------------------------------------------------------
    car_early_rank    = [early_rank_unknown(s) for s in car]
    noncar_early_rank = [early_rank_unknown(s) for s in noncar]
    print(f"\n[CAR vs non-CAR] gold=UNKNOWN samples:")
    print(f"  CAR     early_rank(UNK) mean={np.mean(car_early_rank):.0f}  (lower=more preferred)")
    print(f"  non_CAR early_rank(UNK) mean={np.mean(noncar_early_rank):.0f}")

    # ------------------------------------------------------------------
    # Classifier: can early_logit_gap distinguish suppressed (Abs Rate) from non-Abstention-Inflation?
    # Suppressed: early_gap < threshold (gold clearly preferred early)
    # Non-suppressed: early_gap >= threshold (UNKNOWN already preferred early)
    # ------------------------------------------------------------------
    auc, acc, thr = compute_auc_threshold(ai_early, non_ai_early)
    print(f"\n[Classifier: Abs Rate vs non-Abstention-Inflation via early_logit_gap]")
    print(f"  AUC={auc:.3f}  best_acc={acc:.3f}  threshold={thr:.3f}")

    # ------------------------------------------------------------------
    # Flip-layer distribution: core suppression evidence
    # ------------------------------------------------------------------
    ai_flipped = [f for f in ai_flip if f >= 0]
    print(f"\n[Flip-layer stats for Abstention Inflation samples in {args.ckpt}]")
    if ai_flipped:
        print(f"  {len(ai_flipped)}/{len(ai_flip)} samples have a flip")
        print(f"  mean={np.mean(ai_flipped):.1f}  median={np.median(ai_flipped):.0f}  "
              f"std={np.std(ai_flipped):.1f}")
    else:
        print("  No flips detected (UNKNOWN never overtook gold)")

    # ------------------------------------------------------------------
    # Save JSON summary
    # ------------------------------------------------------------------
    out_data = {
        "checkpoint": args.ckpt,
        "abs_rate": {
            "n": len(abs_rate),
            "early_logit_gap_mean": float(np.mean(ai_early)),
            "flip_rate": ai_flip_rate,
            "flip_layer_median": float(np.median(ai_flipped)) if ai_flipped else None,
            "flip_layer_mean":   float(np.mean(ai_flipped))   if ai_flipped else None,
        },
        "non_ai": {
            "n": len(non_ai),
            "early_logit_gap_mean": float(np.mean(non_ai_early)),
            "flip_rate": non_ai_flip_rate,
        },
        "car": {
            "n": len(car),
            "early_rank_unknown_mean": float(np.mean(car_early_rank)),
        },
        "non_car": {
            "n": len(noncar),
            "early_rank_unknown_mean": float(np.mean(noncar_early_rank)),
        },
        "classifier_auc":      auc,
        "classifier_acc":      acc,
        "classifier_threshold": thr,
    }
    out_dir = ROOT / "results" / "c3"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"suppression_detect_{args.ckpt}.json").write_text(
        json.dumps(out_data, indent=2)
    )

    # ------------------------------------------------------------------
    # Figure: 3 panels
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(f"Suppression Detection via Logit-Lens — {args.ckpt}",
                 fontsize=13, fontweight="bold")

    # Panel 1: layer-by-layer mean logit_gap for Abs Rate vs non_ai
    ax = axes[0]
    n_layers = len(samples[0]["layers"])
    for group, label, color in [
        (abs_rate,    "Abs Rate (suppressed)",  COLORS["abs_rate"]),
        (non_ai, "non-Abstention-Inflation (correct)", COLORS["non_ai"]),
    ]:
        if not group:
            continue
        mat = np.array([[l["logit_gap"] for l in s["layers"]] for s in group])
        mean = mat.mean(axis=0)
        sem  = mat.std(axis=0) / np.sqrt(len(group))
        xs = list(range(n_layers))
        ax.plot(xs, mean, label=label, color=color, lw=2)
        ax.fill_between(xs, mean - sem, mean + sem, color=color, alpha=0.15)
    ax.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.5)
    ax.axvspan(EARLY_LAYERS.start, EARLY_LAYERS.stop, color="gray", alpha=0.08,
               label="Early layers (1-8)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("logit_gap = logit(UNK) − logit(gold)")
    ax.set_title("Abs Rate: late-layer flip signature")
    ax.legend(prop={"size": 11, "weight": "bold"})

    # Panel 2: flip-layer histogram
    ax = axes[1]
    if ai_flipped:
        ax.hist(ai_flipped, bins=range(0, n_layers + 1), color=COLORS["abs_rate"],
                alpha=0.7, label=f"Abs Rate flips (n={len(ai_flipped)})")
    if any(f >= 0 for f in non_ai_flip):
        nf = [f for f in non_ai_flip if f >= 0]
        ax.hist(nf, bins=range(0, n_layers + 1), color=COLORS["non_ai"],
                alpha=0.7, label=f"non-Abstention-Inflation flips (n={len(nf)})")
    ax.set_xlabel("Layer where UNKNOWN first overtakes gold")
    ax.set_ylabel("Count")
    ax.set_title(f"Flip-layer distribution\nAbstention Inflation flip rate={ai_flip_rate:.0%}, "
                 f"non-Abstention-Inflation={non_ai_flip_rate:.0%}")
    ax.legend(prop={"size": 11, "weight": "bold"})

    # Panel 3: rank_unknown early vs final for all 4 groups
    ax = axes[2]
    groups_rank = [
        (abs_rate,    "abs_rate",     COLORS["abs_rate"]),
        (non_ai, "non-Abstention-Inflation", COLORS["non_ai"]),
        (car,    "CAR",     COLORS["CAR"]),
        (noncar, "non-CAR", COLORS["non_CAR"]),
    ]
    xs_pos = [0, 1, 3, 4]
    for pos, (group, label, color) in zip(xs_pos, groups_rank):
        if not group:
            continue
        early_r = [early_rank_unknown(s) for s in group]
        final_r = [final_rank_unknown(s) for s in group]
        ax.bar(pos,     np.mean(early_r), 0.4, color=color, alpha=0.9, label=f"{label} early")
        ax.bar(pos+0.4, np.mean(final_r), 0.4, color=color, alpha=0.4)
    ax.set_xticks([0.2, 1.2, 3.2, 4.2])
    ax.set_xticklabels(["abs_rate", "non-Abstention-Inflation", "CAR", "non-CAR"],
                       fontsize=11, fontweight="bold")
    ax.set_ylabel("rank(UNKNOWN)  [lower=more preferred]")
    ax.set_title("rank(UNK): early layers (solid) vs final (faded)")
    solid = mpatches.Patch(color="gray", alpha=0.9, label="early layers 1-8")
    faded = mpatches.Patch(color="gray", alpha=0.4, label="final 4 layers")
    ax.legend(handles=[solid, faded],
              prop={"size": 11, "weight": "bold"})

    for panel_ax in axes:
        plt.setp(panel_ax.get_xticklabels(), fontsize=11, fontweight="bold")
        plt.setp(panel_ax.get_yticklabels(), fontsize=11, fontweight="bold")

    plt.tight_layout()
    out_pdf = (
        Path.home()
        / "Desktop"
        / "EMNLP Abstention Inflation"
        / "Abstention Chart"
        / f"c3_suppression_detect_{args.ckpt}.pdf"
    )
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_pdf), dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {out_pdf}")


if __name__ == "__main__":
    main()

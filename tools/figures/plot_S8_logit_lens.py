"""
C3 Final Figure — two panels, three checkpoints as line plots.

The instruction-tuned line is the **SFT-only** rung, not the released
Instruct. The claim this figure carries is that the override is installed by
instruction tuning, and only Instruct-SFT isolates that: the released Instruct
is Base -> SFT -> DPO -> RLVR, so pitting it against RL-Zero compares a model
that *contains* RLVR against an RL model and cannot separate the two. SFT alone
already reaches 87% of the released model's output-layer prior, so nothing is
given up by using the rung that matches the claim. The remaining stages are not
hidden: the full ladder is printed by this script's text summary, for the
caption to quote. It is not drawn -- +DPO and +RLVR sit within ~1.8 logits of
SFT and would overlap it into a single band, and Panel A's legend already fills
most of the panel, leaving no room for an inset.

Panel A (S1 vs S2 format trigger):
  Answerable samples (abstain + non-abstain).
  6 lines: 3 checkpoints × {S1 dashed, S2 solid}.
  Color = checkpoint, style = condition.

Panel B (Unfaithful Unknown vs Wrong S1 in S2):
  6 lines: 3 checkpoints × {Unfaithful solid, Wrong dashed}.
  Color = checkpoint, style = group.

Run:
    python scripts/c3_plot.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Repo root on the path before importing from `core` -- this script is run from
# its own directory, so the package is not otherwise importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

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

ROOT = Path(__file__).resolve().parents[2]   # repo root
sys.path.insert(0, str(ROOT))

from core.result_schema import canonical_sample_type   # noqa: E402
sys.path.insert(0, str(ROOT))

DATA_DIR     = ROOT / "results" / "c3"
OUT_DIR      = ROOT / "figures"
INFERENCE_PATH = ROOT / "results" / "c3" / "olmo_inference_FLD.json"

CKPTS = ["base", "sft", "rl_zero"]

#: The alignment ladder reported by the text summary, in recipe order. Nothing
#: plots it: it exists so the caption can quote where the SFT rung sits relative
#: to the released model. RL-Zero is appended separately -- it branches off Base
#: rather than continuing the staged pipeline.
LADDER = ["base", "sft", "dpo", "instruct"]
# Labels carry the full checkpoint name for two reasons. Ai2 changed the
# capitalisation at the third generation -- `Olmo-3`, where `OLMo-2` was the
# earlier style -- and Olmo-3 ships at 7B *and* 32B, so the size is not
# implied. RL-Zero additionally comes in five task variants (Math, Code, IF,
# General, Mix); the one probed here is General -- the file is
# Olmo-3-7B-RL-Zero-General, kept out of the label to hold the legend narrow.
CKPT_STYLE = {
    "base":     dict(color="#9e9e9e", lw=1.8, label="Olmo-3-7B (Base)"),
    # The SFT-only rung -- Base -> SFT, nothing after it. `logit_lens_instruct`
    # is NOT this model: it is the released Instruct (Base -> SFT -> DPO ->
    # RLVR), which is what these panels used to plot under an "SFT on Base"
    # label. Verified numerically rather than from the file name: mean |diff| in
    # per-layer lse_unk over the 600 shared samples puts `instruct` 0.017 from
    # logit_lens_instruct_final.json (the same model, run twice), 0.291 from
    # dpo, 0.437 from sft and 3.64 from base -- the ordering of a ladder
    # endpoint, whose nearest neighbour is the stage immediately before it.
    "sft":      dict(color="#e07b39", lw=2.2, label="Olmo-3-7B-Instruct-SFT"),
    "rl_zero":  dict(color="#c0392b", lw=2.0, label="Olmo-3-7B-RL-Zero"),
}


def load(ckpt: str) -> list[dict]:
    raw = json.loads((DATA_DIR / f"logit_lens_{ckpt}.json").read_text())
    return _normalize_sample_types(raw["per_sample"])


def metric_matrix(samples: list[dict], condition: str, metric: str = "lse_unk") -> np.ndarray:
    key = f"layers_{condition}"
    return np.array([[l[metric] for l in s[key]] for s in samples])


def sem(mat: np.ndarray) -> np.ndarray:
    return mat.std(axis=0) / np.sqrt(mat.shape[0])


GOLD_MAP = {
    "__PROVED__":    "PROVED",
    "__DISPROVED__": "DISPROVED",
    "__UNKNOWN__":   "UNKNOWN",
}


def load_confirmed_ids() -> set[str]:
    """S1-confirmed unfaithful: pred_s1 correct AND pred_s2 = UNKNOWN AND gold != UNKNOWN."""
    raw = json.loads(INFERENCE_PATH.read_text())
    return {s["id"] for s in raw
            if s.get("pred_s1") in ("PROVED", "DISPROVED")
            and s.get("pred_s2") == "UNKNOWN"
            and s["proof_label"] != "__UNKNOWN__"}


def load_wrong_s1_ids() -> set[str]:
    """Samples that answered wrong in S1: pred_s1 != gold AND gold != UNKNOWN."""
    raw = json.loads(INFERENCE_PATH.read_text())
    return {s["id"] for s in raw
            if GOLD_MAP.get(s["proof_label"]) != "UNKNOWN"
            and s.get("pred_s1") in ("PROVED", "DISPROVED")
            and s.get("pred_s1") != GOLD_MAP.get(s["proof_label"])}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    confirmed_ids = load_confirmed_ids()
    wrong_s1_ids  = load_wrong_s1_ids()
    print(f"S1-confirmed unfaithful IDs: {len(confirmed_ids)}")
    print(f"Wrong S1 answer IDs:         {len(wrong_s1_ids)}")

    data = {}
    for ckpt in CKPTS:
        path = DATA_DIR / f"logit_lens_{ckpt}.json"
        if not path.exists():
            print(f"[skip] {ckpt}: file not found")
            continue
        samples = load(ckpt)
        if "layers_s1" not in samples[0]:
            print(f"[skip] {ckpt}: old format, no layers_s1")
            continue
        data[ckpt] = samples
        counts = {t: sum(1 for s in samples if s["sample_type"] == t)
                  for t in ("abs_rate", "non_ai", "CAR", "non_CAR")}
        print(f"{ckpt}: {len(samples)} samples  {counts}")

    if not data:
        print("[error] no data found"); return

    n_layers = len(next(iter(data.values()))[0]["layers_s1"])
    xs = list(range(1, n_layers + 1))

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 4))

    # ── Panel A: S1 vs S2 format trigger ─────────────────────────────
    for ckpt, style in CKPT_STYLE.items():
        if ckpt not in data:
            continue
        answerable = [s for s in data[ckpt]
                      if s["sample_type"] in ("ai", "non_ai")]
        for cond, ls, alpha in [("s1", "--", 0.10), ("s2", "-", 0.14)]:
            mat  = metric_matrix(answerable, cond, metric="logit_gap")
            mean = mat.mean(axis=0)
            se   = sem(mat)
            ax_a.plot(xs, mean, color=style["color"], ls=ls, lw=style["lw"])
            ax_a.fill_between(xs, mean - se, mean + se,
                              color=style["color"], alpha=alpha)

    ax_a.axhline(0, color="black", lw=0.8, ls=":", alpha=0.4)
    ax_a.set_xlabel("Model Layer", fontsize=12, fontweight="bold")
    ax_a.set_ylabel("Δ log P(\"Unknown\")", fontsize=12, fontweight="bold")
    ax_a.set_title("Representation w/ and w/o \"Unknown\" Option",
                   fontsize=12, fontweight="bold")
    ax_a.set_xticks([1, 8, 16, 24, 33])
    ax_a.grid(True, alpha=0.15)
    handles_a = [
        Line2D([0], [0], color=CKPT_STYLE[c]["color"],
               lw=CKPT_STYLE[c]["lw"], label=CKPT_STYLE[c]["label"])
        for c in CKPTS if c in data
    ] + [
        Line2D([0], [0], color="black", lw=1.5, ls="-",  label="w/ \"Unknown\""),
        Line2D([0], [0], color="black", lw=1.5, ls="--", label="w/o \"Unknown\""),
    ]
    ax_a.legend(handles=handles_a, loc="upper left",
                prop={"size": 11, "weight": "bold"},
                ncol=1, framealpha=0.6, handlelength=1.4,
                handletextpad=0.5, labelspacing=0.3, borderpad=0.45)

    # ── Panel B: Unfaithful Unknown vs Wrong-S1 samples (S2) ─────────
    for ckpt, style in CKPT_STYLE.items():
        if ckpt not in data:
            continue
        unfaithful = [s for s in data[ckpt] if s["id"] in confirmed_ids]
        wrong_s1   = [s for s in data[ckpt] if s["id"] in wrong_s1_ids]

        for grp, cond, ls, alpha in [
            (unfaithful, "s2", "-",  0.14),
            (wrong_s1,   "s2", "--", 0.10),
        ]:
            if not grp:
                continue
            mat  = metric_matrix(grp, cond)
            mean = mat.mean(axis=0)
            se   = sem(mat)
            ax_b.plot(xs, mean, color=style["color"], ls=ls, lw=style["lw"])
            ax_b.fill_between(xs, mean - se, mean + se,
                              color=style["color"], alpha=alpha)

    ax_b.set_xlabel("Model Layer", fontsize=12, fontweight="bold")
    ax_b.set_ylabel("log P(\"Unknown\")", fontsize=12, fontweight="bold")
    ax_b.set_title("Abstention Inflation and Wrong Prediction",
                   fontsize=12, fontweight="bold")
    ax_b.set_xticks([1, 8, 16, 24, 33])
    ax_b.grid(True, alpha=0.15)
    handles_b = [
        Line2D([0], [0], color=CKPT_STYLE[c]["color"],
               lw=CKPT_STYLE[c]["lw"], label=CKPT_STYLE[c]["label"])
        for c in CKPTS if c in data
    ] + [
        Line2D([0], [0], color="black", lw=1.5, ls="-",  label="Abstention Inflation"),
        Line2D([0], [0], color="black", lw=1.5, ls="--", label="Wrong Prediction"),
    ]
    ax_b.legend(handles=handles_b, loc="upper left",
                prop={"size": 11, "weight": "bold"},
                ncol=1, framealpha=0.6, handlelength=1.4,
                handletextpad=0.5, labelspacing=0.3, borderpad=0.45)

    for panel_ax in (ax_a, ax_b):
        plt.setp(panel_ax.get_xticklabels(), fontsize=11, fontweight="bold")
        plt.setp(panel_ax.get_yticklabels(), fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig.savefig(str(OUT_DIR / "c3_main.pdf"), dpi=300, bbox_inches="tight")
    print(f"Saved → {OUT_DIR}/c3_main.pdf")
    plt.close()

    # ── Text summary ──────────────────────────────────────────────────
    print("\n" + "=" * 55)
    for ckpt in CKPTS:
        if ckpt not in data:
            continue
        answerable = [s for s in data[ckpt]
                      if s["sample_type"] in ("ai", "non_ai")]
        ai = [s for s in data[ckpt] if s["sample_type"] == "ai"]
        car = [s for s in data[ckpt] if s["sample_type"] == "CAR"]

        s1f = metric_matrix(answerable, "s1", "logit_gap")[:, -1].mean()
        s2f = metric_matrix(answerable, "s2", "logit_gap")[:, -1].mean()
        print(f"\n[{ckpt}]  S1 final={s1f:.3f}  S2 final={s2f:.3f}  "
              f"delta={s2f-s1f:.3f}")
        unfaithful = [s for s in data[ckpt] if s["id"] in confirmed_ids]
        wrong_s1   = [s for s in data[ckpt] if s["id"] in wrong_s1_ids]
        if unfaithful:
            print(f"  Unfaithful S2 lse_unk final={metric_matrix(unfaithful,'s2','lse_unk')[:,-1].mean():.3f}  n={len(unfaithful)}")
        if wrong_s1:
            print(f"  Wrong-S1   S2 lse_unk final={metric_matrix(wrong_s1,  's2','lse_unk')[:,-1].mean():.3f}  n={len(wrong_s1)}")

    # ── Alignment ladder (the inset), printed so the caption can quote it ──
    print("\n" + "=" * 55)
    print("alignment ladder — output-layer logit_gap, answerable samples")
    # Collect first: the "% of final" column needs the last rung's value while
    # printing the first, so this cannot be done in one pass.
    vals = {}
    for ckpt in LADDER + ["rl_zero"]:
        if not (DATA_DIR / f"logit_lens_{ckpt}.json").exists():
            continue
        answerable = [s for s in load(ckpt)
                      if s["sample_type"] in ("ai", "non_ai")]
        vals[ckpt] = (metric_matrix(answerable, "s1", "logit_gap")[:, -1].mean(),
                      metric_matrix(answerable, "s2", "logit_gap")[:, -1].mean())
    final = vals.get("instruct", (None, None))[1]
    for ckpt in LADDER + ["rl_zero"]:
        if ckpt not in vals:
            print(f"  {ckpt:<9} [missing]")
            continue
        s1, s2 = vals[ckpt]
        share = f"{100 * s2 / final:5.0f}% of final" if final else ""
        print(f"  {ckpt:<9} S1={s1:+6.2f}  S2={s2:+6.2f}  "
              f"delta={s2 - s1:+6.2f}  {share}")


if __name__ == "__main__":
    main()

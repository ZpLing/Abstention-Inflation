"""Structural probe supplementary experiment — PLAN_v2 §四 小实验 1D.

Four sub-experiments (P1–P4) operate on hidden states extracted from a *local*
HuggingFace causal-LM (closed-API models do not expose hidden states):

    P1  AID extraction       — diff-of-means(S2 − S1) at the answer token; pick
                                best layer ℓ* by per-layer logistic AUC.
    P2  "Knows but won't say" — per-layer 4-class linear probe (sklearn) trained
                                on S1-correct samples; inference on Abs Rate S2 hidden
                                states; report c_letter confidence.
    P3  3-way attribution    — softmax classifier at ℓ* trained on
                                γ (S1-correct), β (S1-incorrect), α (FLD genuine-
                                unknown S2=Unknown). Apply to Abs Rate; argmax + conf.
    P4  Directional ablation — at ℓ* hook the residual stream and subtract the
                                AID projection during generation; parse the new
                                final letter; report mech-recovery & correct-
                                recovery split by P3 3-way label.

Hardware: MacBook-class fp16 small models only. Defaults set up for the local
checkout `models/gemma-4-E2B-it`. Override via config.

Inputs (read from disk, not re-queried):
    - results/ab/ab_summary_<dataset>_<model>.json    (main Exp 1 product)
    - results/supplementary/supp_summary_FLD_<model>.json  (CAR α-pool source)

Outputs:
    - results/supplementary/probe_1d_<dataset>_<probe_model>.json
"""
from .runner import Probe1DRunner

__all__ = ["Probe1DRunner"]

# Experiments

One folder per claim, one subfolder per experimental setting. Every setting
folder has its own README with the exact command and output path.

```
C1_structural_trigger/     §4.1  the trigger is structural, not semantic
  S1_baseline/                   original label set, no extra option
  S2_unknown_option_added/       the manipulation under study
  S3_question_format_ablation/   TFQ re-rendered as A/B/C letters
  S4_word_content_ablation/      abstain word → synonym / random word

C2_introspective_gap/      §4.2  the model denies it can answer
  S5_without_unknown_rerun/      remove the option, force a commitment
  S6_self_diagnosis/             the model attributes its own abstention

C3_late_layer_override/    §4.3  where in the network it happens
  S7_reasoning_trace_evaluation/ DeBERTa NLI probe over whole traces
  S8_logit_lens_probe/           OLMo-3-7B, 33 layers, numbered 01_…07_

C4_stable_bias/            §4.4  a stable bias from instruction tuning
  S9_stability/                  3 re-draws at default T; truly-Unknown mirror
  S10_factor_analysis/           difficulty, temperature, size, alignment
  S11_option_position/           the abstain verb moves to slot 1 / 2 / 3

appendix_E_mitigation/     App. E  R1, the logit calibration remedy
```

Each setting has exactly one runner and one analysis. Where a setting is
collected by the shared runner rather than by a script of its own -- S1, S2, S3
and the S5 rerun all come out of one pass so their per-item outputs stay paired
-- the folder holds a thin `run.py` that points `main.py` at the right config.

The numbers the paper states are rebuilt by `reporting/`, not from here, and
always from the canonical summaries under `results/tfq_n500/`,
`results/mcq_n500/` and each setting's own output directory. An analysis that
reads a second run of the same prompt -- the S11 slot-C cell reproduces the S2
prompt byte for byte, but it is a different run -- will disagree with Table 1,
which is how several stale numbers survived in this repo for a while.

Shared code is in `core/`; nothing here defines its own prompts, parser or
metrics.

Run anything from the repo root — the scripts insert the root on `sys.path`
themselves, so the working directory has to be the repo root for the relative
`results/` and `dataset/` paths to resolve.

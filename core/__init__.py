"""Shared infrastructure for the Abstention Inflation experiments.

The paper-aligned per-setting entry points live under ``experiments/Cx_*/Sy_*/``.
This package is what they all import from.

Modules
-------
config_loader         YAML config + secrets merging
data_handler          Thin compatibility wrapper around :mod:`dataset_loader`
dataset_loader        Unified loader for ``software/dataset/<name>.json``
                      (FLD, FLD_unknown, FOLIO, FOLIO_unknown, ARC, MedQA,
                       MMLU, LogiQA)
label_scheme          ``LabelScheme`` — per-dataset verbs, framing labels,
                      task instructions, output parser. FLD and FOLIO both
                      use True / False; FLD's abstain verb is ``Unknown`` and
                      FOLIO's is ``Uncertain``.
prompts               S1..S6 prompt builders (MCQ + Judge / TFQ)
llm_handler           Async OpenAI-compatible client wrapper
evaluator             Standalone output parser used by legacy summaries
metrics               Acc, Abs Rate, Abs Rate, F1, recovery, etc.
trace_extractors      S7 reasoning-trace tokenisation & alignment

Runner classes (one entry point each, mapped to a paper setting)
----------------
ab_runner.ABRunner                          S1 / S2 / S3 + S4 forced-choice
s5_rerun_runner                              S5 (w/o "Unknown" Option Rerun)
s6_self_diagnosis_runner                     S6 (Self-Diagnosis)
truly_unknown_runner.TrulyUnknownRunner     S9 truly-Unknown perception
s10_model_sweep_runner.ModelSweepRunner  S10 Alignment & Size sweep
s10_difficulty_runner                        S10 FLD step-count difficulty
posthoc_mitigation_runner                    Post-hoc mitigation baseline
                                              (Discussion appendix)
"""

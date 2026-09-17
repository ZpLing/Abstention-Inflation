# S7 — Reasoning Traces Evaluation

S7 takes the **already-run** S1 and S2 outputs (no new API calls), extracts
the Reasoning block of every CoT, and answers two questions:

1. **Trace quality invariance.**
   Per-item paired F1 between S1 and S2 traces. The paper finds F1 is
   essentially unchanged — adding "Unknown" does not degrade CoT quality —
   even as accuracy crashes. See `compute_F1T.py`.

2. **What does the trace itself encode?**
   A DeBERTa-v3-large NLI probe classifies each trace (premise = Reasoning
   block, hypothesis = the dataset's Conclusion / Hypothesis statement) into
   Entailment / Contradiction / Neutral. The paper reports 42.3 % of
   Abstention Inflation traces are non-Neutral (gold-aligned). See
   `nli_probe_DeBERTa.py`; `nli_probe_tasksource.py` is an alternative NLI
   backbone used for the robustness check in the appendix.

## Scripts in this folder

| Script | Purpose |
| ------ | ------- |
| `compute_F1T.py` | F1T (per-item, paired S1 vs S2) on FLD / FOLIO traces. |
| `nli_probe_DeBERTa.py` | NLI probe with DeBERTa-v3-large MultiNLI. |
| `nli_probe_tasksource.py` | NLI probe with `tasksource/deberta-base-long-nli` (robustness check). |
| `analyze_acc_F1T_NLI_triangle.py` | Cross-cell triangulation of Acc × F1T × NLI from S2 transcripts. |
| `analyze_trace_quality_vs_gamma.py` | Trace quality regressed against γ = Abs Rate. |
| `analyze_trace_pairwise_baseline.py` | Pairwise S1↔S2 baseline diagnostics. |
| `analyze_trace_step_embedding.py` | Sentence-embedding cosine between traces (alternate metric). |

## Run

```bash
# F1T (trace quality invariance)
python experiments/C3_later_layer_override/S7_reasoning_traces_evaluation/compute_F1T.py \
    --runs results/<run-name>

# NLI probe (DeBERTa MultiNLI)
python experiments/C3_later_layer_override/S7_reasoning_traces_evaluation/nli_probe_DeBERTa.py \
    --transcripts results/<run-name>
```

## Inputs / outputs

Inputs: the `predictions_s1[]` and `predictions_s2[]` arrays inside
`ab_summary_<dataset>_<model>.json` produced by S2. No new model calls.

Outputs: per-item NLI labels and per-cell summary JSON under
`results/analysis/s7_*.json`. The plot script
`plots/fig_S7_trace_invariance.py` (= old `plot_c0_trace_invariance.py`)
produces Figure 4 in the paper.

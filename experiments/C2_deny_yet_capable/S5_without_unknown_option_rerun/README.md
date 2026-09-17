# S5 — w/o "Unknown" Option Rerun

Paper §4.2.1, Figure 4 (left).

On the Abstention Inflation samples produced by S2 (items whose gold label is
True/False but the model answered "Unknown"), a **multi-turn** follow-up is
issued:

1. the original S2 conversation is replayed verbatim (user turn + the model's
   own "Unknown" answer), then
2. a follow-up user turn removes the abstain option and forces a commitment to
   the two original labels.

The paper reports 52–75% per-cell accuracy — well above the 50% random
baseline — evidence that the abstention hid a recoverable answer rather than a
knowledge boundary.

## Where the code lives

S5 is not a separate pass: `core.ab_runner.ABRunner` already runs S2, so it
selects the abstaining samples in the same loop and issues the follow-up when
`run_s5_rerun: true`. The prompt builders are
`core.prompts.build_{judge,mcq}_s5_rerun_prompt`, and the results land in the
`s5_rerun` block of `ab_summary_<dataset>_<model>.json`.

## Run

```bash
# any C1 config works — S5 only needs S1 + S2 to have run
python experiments/C2_deny_yet_capable/S5_without_unknown_option_rerun/run.py \
    --config configs/C1_structural_trigger/DeepSeek_R1_FLD_FOLIO.yaml
```

## Aggregate

```bash
python experiments/C2_deny_yet_capable/S5_without_unknown_option_rerun/analyze_S5.py \
    --datasets FLD FOLIO
```

Prints per-(model, dataset) accuracy on the abstaining subset, the gap from the
50% baseline and a one-sample binomial p-value, plus the pooled estimate. It
reads both current and pre-rename summaries via
`core.result_schema.load_summary`, so it works across every `results/` directory
in this repo.

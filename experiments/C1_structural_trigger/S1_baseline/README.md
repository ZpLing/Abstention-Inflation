# S1 — Baseline (no "Unknown" option)

The model is given the prompt with the **original** label set:

* **TFQs (FLD, FOLIO)** — `True | False`
* **MCQs (ARC, MedQA, MMLU, LogiQA)** — `A | B | C | D`

S1 establishes the per-(model, dataset) baseline accuracy that S2 is then
compared against. There is no Abs Rate to report under S1: the option to
abstain has been removed.

## Run

S1 shares its runner with S2; the YAML chooses which settings to enable.
Two equivalent entry points:

```bash
# (a) Standalone S1 only
python experiments/C1_structural_trigger/S1_baseline/run.py \
    --config configs/C1_structural_trigger/<model>_<dataset>.yaml

# (b) Via the central dispatcher (S1 always runs as part of main_experiment)
python main.py --config configs/C1_structural_trigger/<model>_<dataset>.yaml
```

A minimal config sets `main_experiment.settings: ["S1"]`. Reported results in
the paper come from running all of `["S1", "S2"]` in one pass so the per-item
S1↔S2 comparison stays paired.

## Outputs

Each (dataset, model) cell writes `results/<run-name>/ab_summary_<dataset>_<model>.json`
with `label_acc_s1` plus per-item predictions in `predictions_s1[]`.

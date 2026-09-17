# S2 — "Unknown" Option Added

Identical to S1 except the third option (verb "Unknown" for FLD /
"Uncertain" for FOLIO / fifth letter "E. Unknown" for MCQs) is
appended to the option set. S2 is the experiment that surfaces
Abstention Inflation: on TFQs it drops accuracy by ~15.75pp on
average and pushes Abs Rate to 32.9pp; on MCQs the effect is
within noise.

S2 is always run paired with S1 so per-item McNemar tests are
available downstream. The default `run.py` enforces
`settings: ["S1", "S2"]` when the YAML omits the field.

## Run

```bash
python experiments/C1_structural_trigger/S2_unknown_option_added/run.py \
    --config configs/C1_structural_trigger/<model>_<dataset>.yaml
```

Reported in the paper's Table 1 (Main Results).

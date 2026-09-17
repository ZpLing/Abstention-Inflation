# S9 — Stability

Two sub-experiments, both designed to rule out the "Abstention Inflation is
just stochastic noise" counter-hypothesis.

| File | Purpose | Output |
| ---- | ------- | ------ |
| `run_S9_persistence.py` | Re-issue the S2 prompt three more times per Abstention Inflation sample at T=0.5. Records whether each re-draw returns Unknown. Reported as the 0/1/2/3-consistent-abstain histogram in Figure 6 (left). | `results/<run>/s9_persistence_<dataset>_<model>.json` |
| `run_S9_truly_unknown.py` | Run S1 + S2 + S3 on the truly-Unknown subsets (`dataset/FLD_unknown.json`, `dataset/FOLIO_unknown.json`). Reported as Figure 6 (right): Abs Rate on truly-Unknown vs. answerable items. | `results/supplementary/<dataset>_<model>.json` |

## Run

```bash
# (a) Persistence across repeats
python experiments/C4_stable_bias/S9_stability/run_S9_persistence.py \
    --config configs/C4_stable_bias/persistence_<model>.yaml

# (b) Truly-Unknown perception (uses the supplementary_experiment task)
python main.py --config configs/C4_stable_bias/truly_unknown_<model>.yaml
```

The truly-Unknown runner is `core.s9_truly_unknown_runner.SupplementaryRunner`;
it loads `dataset/<name>_unknown.json` directly so no special data prep is
required. The persistence runner re-uses the S2 prompts but sets `T=0.5`
and issues three additional draws per Abstention Inflation sample
identified in a prior S2 run.

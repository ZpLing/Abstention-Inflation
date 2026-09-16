# C4 — Abstention Inflation is a stable bias resulting from instruction tuning, not stochastic noise

Three settings, one per failure mode of the "it's just noise" counter-hypothesis:
it does not wash out across redraws, it does not track the sampling knobs, and
it does not come from where the option sits in the list.

| Setting | Folder | What it shows |
| ------- | ------ | ------------- |
| **S9** Stability | `S9_stability/` | (a) Persistence: 52.4 % of Abstention Inflation samples re-abstain on all three independent draws at T=0.5 — 4.2× the Bernoulli baseline. (b) Truly-Unknown perception: Abs Rate on `FLD_unknown` / `FOLIO_unknown` is 2.29× that on answerable items, ruling out indiscriminate abstention. |
| **S10** Factor Analysis | `S10_factor_analysis/` | Temperature, FLD step-count difficulty, model size, and Base→IT alignment — the Gemma family sweep shows the IT lift in Abs Rate is independent of size. |
| **S11** Positional Biases | `S11_option_position/` | The abstain verb moves to the first, second or third slot of the S2 prompt while the question format is held fixed. Abs Rate barely moves, so the trigger is the option's presence and not its position. Slot three is the S2 ordering itself. |

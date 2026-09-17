# C4 — Abstention Inflation is a stable bias resulting from instruction tuning, not stochastic noise

Two settings, one per failure mode of the "it's just noise" counter-hypothesis.

| Setting | Folder | What it shows |
| ------- | ------ | ------------- |
| **S9** Stability | `S9_stability/` | (a) Persistence: 52.4 % of Abstention Inflation samples re-abstain on all three independent draws at T=0.5 — 4.2× the Bernoulli baseline. (b) Truly-Unknown perception: Abs Rate on `FLD_unknown` / `FOLIO_unknown` is 2.29× that on answerable items, ruling out indiscriminate abstention. |
| **S10** Factor Analysis | `S10_factor_analysis/` | Temperature, FLD step-count difficulty, model size, and Base→IT alignment — the Gemma family sweep shows the IT lift in Abs Rate is independent of size. |

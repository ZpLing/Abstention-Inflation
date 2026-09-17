## OLMo-3 staged-checkpoint logit-lens — output-layer abstention prior

Metric: `Δlog P(Unknown) = log P(Unknown) − log[P(PROVED)+P(DISPROVED)]`, logit-lens over 33 layers, FLD, N=600 (answerable N=300, AIR N=112). Larger = stronger abstention representation. S2 = "Unknown" option present; S1 = binary (no "Unknown"). Bold = staged Instruct pipeline.

### Table 1 — Abstention prior across alignment stages

| Checkpoint (recipe) | Output-layer Δlog P(Unk) (S2) | Peak (layer) | % of final Instruct | Format effect Δ(S2−S1) |
|---|---|---|---|---|
| Base — no alignment | +8.88 | +8.88 (L32) | 62% | +5.51 |
| **Instruct-SFT — SFT** | +12.51 | +14.93 (L30) | 87% | +11.63 |
| **Instruct-DPO — SFT+DPO** | +13.45 | +15.51 (L30) | 94% | +12.92 |
| **Instruct — SFT+DPO+RLVR** | +14.33 | +16.15 (L30) | 100% | +13.18 |
| RL-Zero — RLVR on base (no SFT/DPO) | +0.26 | +2.01 (L27) | 2% | +0.57 |

### Table 2 — Prior is indiscriminate: AIR (answerable) vs CAR (genuinely-unknown)

Output-layer `log P(Unknown)` (S2). AIR = should be answered but model abstains (N=112); CAR = genuinely unknown, abstention correct (N=132). A faithful model would separate them; the injected prior does not.

| Checkpoint | AIR log P(Unk) | CAR log P(Unk) | AIR−CAR gap |
|---|---|---|---|
| Base | +2.06 | +1.59 | +0.47 |
| Instruct-SFT | +15.22 | +15.61 | -0.40 |
| Instruct-DPO | +16.87 | +17.56 | -0.69 |
| Instruct | +17.34 | +17.96 | -0.62 |
| RL-Zero | -21.77 | -22.20 | +0.42 |


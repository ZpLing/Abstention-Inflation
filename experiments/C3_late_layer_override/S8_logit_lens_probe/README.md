# S8 — Logit-Lens Representation Probe

Probes per-layer `log P("Unknown")` across all 33 transformer layers of
**OLMo-3-7B** in three variants — Base (pretrained), Instruct (SFT on
Base), RL-Zero (RL on Base) — under both S1 and S2 prompts, on
FLD + FOLIO.

The paper reports a sharp late-layer rise in `Δ log P(Unknown)` between S1
and S2, with **Instruct most affected, Base intermediate, RL-Zero nearly
inert** — direct evidence that the override is installed by instruction
tuning and operates in the final transformer layers.

S8 is a multi-stage pipeline; run the scripts in numeric order.

## Pipeline

| Step | Script | Description |
| ---- | ------ | ----------- |
| 1 | `01_download_OLMo3.py` | Download the three OLMo-3-7B variants (Base, Instruct, RL-Zero) from HuggingFace. |
| 2 | `02_collect_samples.py` | Subsample 500 FLD + 500 FOLIO items (paper's S8 input set) from the bundled dataset/. |
| 3 | `03_run_OLMo_inference.py` | Run S1 and S2 prompts under each variant, capturing **hidden states layer-by-layer** at the position immediately preceding the answer token. |
| 4 | `04_compute_logit_lens.py` | Project each layer's hidden state through the model's tied embedding matrix and record `log P(Unknown)` per layer. |
| 5 | `05_compute_wrong_prediction_baseline.py` | Same probe on Wrong-Prediction samples (gold ≠ Unknown, prediction wrong under S1) — the comparison group for the right-panel chart. |
| 6 | `06_suppression_detect.py` | Companion analysis: detect "suppression" patterns where mid-layer evidence for True/False is overridden by late-layer Unknown. |

## Run

```bash
cd software
python experiments/C3_late_layer_override/S8_logit_lens_probe/01_download_OLMo3.py
python experiments/C3_late_layer_override/S8_logit_lens_probe/02_collect_samples.py
python experiments/C3_late_layer_override/S8_logit_lens_probe/03_run_OLMo_inference.py --variant instruct
python experiments/C3_late_layer_override/S8_logit_lens_probe/03_run_OLMo_inference.py --variant base
python experiments/C3_late_layer_override/S8_logit_lens_probe/03_run_OLMo_inference.py --variant rl_zero
python experiments/C3_late_layer_override/S8_logit_lens_probe/04_compute_logit_lens.py
python experiments/C3_late_layer_override/S8_logit_lens_probe/05_compute_wrong_prediction_baseline.py
```

Requires a GPU. The plot script `plots/fig_S8_logit_lens.py`
(= old `c3_plot.py`) produces Figure 5 in the paper.

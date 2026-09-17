# S4 — Word Content Ablation

Replace the third option ("Unknown") with semantically distinct surface
words while holding every other prompt element constant. Tests whether the
trigger for Abstention Inflation is the **specific word** "Unknown"
(semantic) or the **mere presence of an extra slot** (structural).

| Variant       | Replacement word(s)                | Run script                       |
| ------------- | ---------------------------------- | -------------------------------- |
| Synonyms      | "Indeterminate", "I don't know"    | `run_S4_synonyms.py`             |
| Random words  | "Triangular", "Cerulean"           | `run_S4_random_words.py`         |
| All-model    | Synonyms across all 3 frontier LLMs | `run_S4_synonyms_all_models.py` |

The paper's headline S4 result: pairwise tests across all six (model,
dataset) cells under each replacement variant are n.s. with σ ≤ 1.5 pp —
i.e., the trigger is structural, not semantic.

## Run

```bash
# Random words ("Triangular", "Cerulean")
python experiments/C1_structural_trigger/S4_word_content_ablation/run_S4_random_words.py \
    --config configs/C1_structural_trigger/S4_random_words.yaml

# Synonyms ("Indeterminate", "I don't know")
python experiments/C1_structural_trigger/S4_word_content_ablation/run_S4_synonyms.py \
    --config configs/C1_structural_trigger/S4_synonyms.yaml
```

## Outputs

JSON files in `results/<run-name>/s4_<variant>_<dataset>_<model>.json` with
per-item predictions and abstention counts. The plotting script
`plots/fig_S4_word_content_ablation.py` (= old `plot_b_section.py`)
consumes these to produce Figure 2 in the paper.

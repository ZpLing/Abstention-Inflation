# LLM Abstention Can Be a Prompt Artifact, in Addition to Genuine Uncertainty

Code and datasets for the *Abstention Inflation* paper. The repository is laid
out to mirror the paper: four claims (**C1–C4**), each realised by a group of
experimental settings (**S1–S11**).

```
main.py                 one dispatcher; `--config <yaml>` picks the runner
core/                   everything shared: client, prompts, parsers, metrics, runners
experiments/            one folder per claim, one subfolder per setting
configs/                YAMLs grouped by claim
dataset/                the eight datasets used in the paper, one unified schema
results/                run outputs (git-ignored)
tools/                  dataset preparation, figure scripts, maintenance (git-ignored)
```

## Quick start

```bash
pip install -r requirements.txt
cp configs/secrets.template.yaml secrets.yaml   # then fill in api_key / base_url

# S1 + S2 + S3 + the S5 rerun, DeepSeek-V4-Flash on FLD and FOLIO
python main.py --config configs/C1_structural_trigger/DeepSeek_R1_FLD_FOLIO.yaml
```

Every setting can also be launched from its own folder; each one has a README
with the exact command and the expected output path.

## Repository layout

```
main.py                     dispatcher; --config selects the runner
core/                       shared infrastructure -- prompts, parser, metrics,
                            loaders. Nothing here is setting-specific.
  runners/                  one runner per setting: a file here answers "how
                            was S6 collected", a file above it answers "how is
                            any answer parsed"
experiments/                one folder per claim, one subfolder per setting
  C1_structural_trigger/      S1  S2  S3  S4
  C2_deny_yet_capable/        S5  S6
  C3_later_layer_override/    S7  S8
  C4_stable_bias/             S9  S10  S11
configs/                    one YAML per (model, dataset) cell, named for the
                            settings it collects
reporting/                  rebuilds the reported numbers from results/
dataset/                    the eight benchmark files, one schema
```

Every rate is scored on the keep-set `core.result_schema.paired_keep_ids`
defines: the items both settings of a paired contrast answered. An item is
dropped only when a setting returned nothing usable -- an exhausted retry, a
content-filter refusal, a decoding collapse. A response that declines to commit
is kept, because refusing to commit is the behaviour under study.

## Paper → code map

### C1 — the trigger is the structural presence of an extra option (§4.1)

| Setting | What it does | Entry point | Config |
|---|---|---|---|
| **S1** Baseline | original label set, no extra option | `experiments/C1_structural_trigger/S1_baseline/run.py` | `configs/C1_structural_trigger/*` |
| **S2** "Unknown" Option Added | appends the abstain option | `experiments/C1_structural_trigger/S2_unknown_option_added/run.py` | same |
| **S3** Question Format Ablation | TFQ re-rendered as A/B/C letters | `experiments/C1_structural_trigger/S3_question_format_ablation/run.py` | same |
| **S4** Word Content Ablation | abstain word → synonym / random word | `experiments/C1_structural_trigger/S4_word_content_ablation/run_S4_{synonyms,random_words,mcq_random_words}.py` | `configs/C1_structural_trigger/S4_random_words*.yaml` |

Table 1 comes from S1/S2 (`build_table1_accuracy.py`), Figure 3 from S4
(`tools/figures/plot_fig3_S4_word_ablation.py`).

### C2 — the model denies it can answer, even when it can (§4.2)

| Setting | What it does | Entry point |
|---|---|---|
| **S5** w/o "Unknown" Option Rerun | multi-turn follow-up that removes the option and forces a commitment | `experiments/C2_deny_yet_capable/S5_without_unknown_option_rerun/{run,analyze_S5}.py` |
| **S6** Self-Diagnosis | model attributes its abstention to (A) incapability or (B) truly-Unknown | `experiments/C2_deny_yet_capable/S6_self_diagnosis/run_S6.py` |

S5 is computed by `ABRunner` itself (`run_s5_rerun: true`) and stored in the
`s5_rerun` block of each summary — there is no separate S5 pass. Figure 4 is
`tools/figures/plot_fig4_S5_rerun_and_S6_selfdiag.py`.

### C3 — a later-layer output override (§4.3)

| Setting | What it does | Entry point |
|---|---|---|
| **S7** Reasoning Traces Evaluation | F1 of generated vs annotated traces + DeBERTa NLI probe | `experiments/C3_later_layer_override/S7_reasoning_traces_evaluation/` |
| **S8** Logit-Lens Representation Probe | OLMo-3-7B Base / Instruct / RL-Zero, 33 layers | `experiments/C3_later_layer_override/S8_logit_lens_representation_probe/` (numbered `01_`…`07_` pipeline) |

S7 and S8 add no new prompts — both reuse S1 and S2 verbatim.
Figure 5 = `plot_fig5_S7_trace_invariance.py`, Figure 6 = S8 step `07_`.

### C4 — a stable bias installed by instruction tuning (§4.4)

| Setting | What it does | Entry point |
|---|---|---|
| **S9** Stability | 3 re-draws at *T*=0.5; Abs Rate on truly-Unknown samples | `experiments/C4_stable_bias/S9_stability/run_S9_{persistence,truly_unknown}.py` |
| **S10** Factor Analysis | difficulty (FLD step count), temperature, model size, alignment | `experiments/C4_stable_bias/S10_factor_analysis/` |
| **S11** Positional Biases | the abstain verb moves to slot 1 / 2 / 3 of the S2 prompt | `experiments/C4_stable_bias/S11_positional_biases/run_S11_positional_biases.py` |

### Appendix

| Appendix | Contents | Location |
|---|---|---|
| C Additional Quantitative Analyses | Abs Rate–ΔAcc regression over the 18 cells of Table 1 | `reporting/abs_rate_dacc_regression.py` |
| E Mitigation | calibration suffix, stimulation+reflection | `core/appendix_mitigation_runner.py` |
| E Remedy R2 | contrastive self-consistency override (post-hoc, no API calls) | `core/appendix_remedy_r2_self_consistency.py` |

### Reproducing the reported numbers

`reporting/` rebuilds what the paper states, from `results/` alone:

| Script | Produces |
|---|---|
| `build_table1.py` | Table 1, all 24 cells, on the paired keep-set |
| `abs_rate_dacc_regression.py` | the App. C regressions |
| `audit_parser_provenance.py` | which parser tier each reported label came from |

Every rate in the paper is scored on the keep-set that
`core.result_schema.paired_keep_ids` defines: the items both settings of a
paired contrast answered. An item is dropped only when a setting returned
nothing usable (an exhausted retry, a content-filter refusal, a decoding
collapse); a response that declines to commit is kept, because that is the
behaviour under study. Reading a summary's stored `metrics` block instead is
not equivalent for the MCQ cells, whose block predates that rule.

## Datasets

`dataset/` holds all eight files in one schema — no per-dataset loader:

| File | Type | Use |
|---|---|---|
| `FLD.json`, `FOLIO.json` | TFQ | S1–S11 (500 answerable items each) |
| `FLD_unknown.json`, `FOLIO_unknown.json` | TFQ | S9 truly-Unknown subset (300 each) |
| `ARC.json`, `MMLU.json`, `MedQA.json`, `LogiQA.json` | MCQ | S1, S2 (500 each) |

Each record carries `id / source / task_type / question / context / options /
answer_idx`; `answer_idx == -1` marks a truly-Unknown item. `core.dataset_loader`
is the only reader. Regenerating them from the original releases is
`tools/data_prep/`.

## Metrics

`core/metrics.py`, matching §3.5:

* `label_acc` — Acc, label-prediction accuracy (Eq. 1).
* `abs_rate` — Abs Rate, share of items answered with the abstain option (Eq. 2).
  Under S4 the abstain slot carries a different word; the parser maps it to the
  same token, so Abs Rate measures the changed word too.
* `label_macro_f1`, `mean_trace_set_f1` / `mean_trace_bertscore_f1` — the S7
  trace layer (set-F1 for FLD, BERTScore-F1 for FOLIO / ARC / MedQA).
* `correct_abstention_rate` — the S9 mirror on truly-Unknown items.

## Result files and the pre-rename schema

Runs write `results/<dir>/ab_summary_<dataset>_<model>.json`:

```jsonc
{
  "schema": "paper-s1-s10/v1",
  "metrics":  { "S1": {...}, "S2": {...}, "S3": {...}, "S5": {...} },
  "per_sample": [ { "id", "answer_idx", "pred_s1", "pred_s2", "pred_s3_format", "raw_*" } ],
  "s5_rerun":   [ { "sample_id", "answer_idx", "pred_s5_rerun", "raw_s5_rerun" } ]
}
```

The experiments were run before the settings were renamed to S1–S11, so files
already on disk use the older names — and two of them meant something different:

| pre-rename key | actually holds | current name |
|---|---|---|
| `pred_s1`, `pred_s2` | unchanged | `pred_s1`, `pred_s2` |
| `pred_s5` | TFQ rendered MCQ-style | `pred_s3_format` (paper **S3**) |
| `pred_s4` / `air_followups` | rerun on abstaining samples | `pred_s5_rerun` / `s5_rerun` (paper **S5**) |
| `pred_s3` | S2 + calibration suffix (App. E) | `pred_calibration_suffix` |
| `n_air_s2`, `n_air` | count of items answered `Unknown` | `n_abstention_inflation` |
| `AIR`, `air`, `AIR_S2` | the metric itself | `abs_rate` (the paper's **Abs Rate**) |
| `sample_type: "AIR"` / `"non_AIR"` | one such item / not | `"ai"` / `"non_ai"` |

The code used to call both the metric and the sample category `AIR`; the paper
(arXiv:2507.16199) has a separate term for each — **Abs Rate** for the metric,
**Abstention Inflation sample** for the item — so the two are now named apart.
For files read with a bare `json.load`, `core.result_schema.get_field()` and
`canonical_sample_type()` resolve the legacy spellings individually.

Read summaries through `core.result_schema.load_summary()` and both eras come
back in the current namespace. Old YAML configs keep working too: the legacy
`run_tasks` names and block keys are aliased in `main.py` and
`core/config_loader.py`.

## Reproducing a number end to end

```bash
# S5: accuracy once the "Unknown" option is removed (paper §4.2.1, ~64% pooled)
python experiments/C2_deny_yet_capable/S5_without_unknown_option_rerun/analyze_S5.py \
    --datasets FLD FOLIO
```

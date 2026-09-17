<h1 align="center">LLM Abstention Can Be a Prompt Artifact,<br>in Addition to Genuine Uncertainty</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2507.16199"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2507.16199-b31b1b.svg"></a>
  <a href="https://arxiv.org/abs/2507.16199"><img alt="EMNLP 2026" src="https://img.shields.io/badge/EMNLP%202026-Main%20Conference-2c6fbb"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-3776ab">
</p>

<p align="center">
  <b>Accepted to the EMNLP 2026 Main Conference.</b><br>
  <a href="https://arxiv.org/abs/2507.16199">arXiv:2507.16199</a>
</p>

Adding an "Unknown" option to a True/False question makes a model abstain on
questions it can answer. We call this **Abstention Inflation** and show it is a
structural property of the prompt rather than an expression of uncertainty.

Pooled over 3 models × 2 TFQ datasets at 500 items each, offering the option
costs **19.8 accuracy points** and pushes abstention to **34.5%**. The same
manipulation on 4-option MCQs, over the same models and the same 500-item
scale, costs **1.2 points**.

| Format | n (paired) | Acc without the option | Acc with it | Abs Rate |
| --- | ---: | ---: | ---: | ---: |
| TFQ (FLD, FOLIO) | 2,993 | 79.4% | **59.5%** | **34.5%** |
| MCQ (ARC, MedQA, MMLU, LogiQA) | 5,965 | 84.5% | 83.3% | 2.3% |

The abstention is not uncertainty: removing the option again recovers **63.6%**
accuracy on exactly the items that abstained (S5), the models attribute those
abstentions to the question being unanswerable **95–100%** of the time (S6), and
the effect survives re-sampling, re-wording, and moving the option's position.

## Installation

```bash
git clone https://github.com/ZpLing/EMNLP2026_Abstention-Inflation.git
cd EMNLP2026_Abstention-Inflation
pip install -r requirements.txt
cp configs/secrets.template.yaml secrets.yaml   # fill in api_key / base_url
```

`secrets.yaml` is git-ignored. Credentials resolve in `core/config_loader.py`:
the experiment YAML, then `secrets.yaml`, then an extensionless `config` file at
the repo root, later winning.

## Quick start

```bash
# S1 + S2 + S3 and the S5 rerun, DeepSeek-V4-Flash on FLD and FOLIO
python main.py --config configs/C1_structural_trigger/S1_S3_TFQ_n500_DeepSeek_V4_Flash.yaml

# Rebuild Table 1 from whatever is in results/
python reporting/build_table1.py
```

## Repository layout

```
main.py                      dispatcher; --config selects the runner
core/                        shared infrastructure — prompts, parser, metrics,
                             loaders. Nothing here is setting-specific.
  runners/                   one runner per setting: a file here answers "how
                             was S6 collected", a file above it answers "how is
                             any answer parsed"
experiments/                 one folder per claim, one subfolder per setting
  C1_structural_trigger/       S1  S2  S3  S4
  C2_deny_yet_capable/         S5  S6
  C3_later_layer_override/     S7  S8
  C4_stable_bias/              S9  S10  S11
configs/                     one YAML per (model, dataset) cell, named for the
                             settings it collects
reporting/                   rebuilds the reported numbers from results/
dataset/                     the eight benchmark files, one schema
```

C3 has no `configs/` entry: S7 scores traces that are already on disk and S8
runs a local checkpoint, so neither reaches the gateway. Both take their
arguments on the command line.

## Paper → code map

### C1 — the trigger is the structural presence of an extra option (§4.1)

| Setting | What it does | Entry point |
| --- | --- | --- |
| **S1** Baseline | original label set, no extra option | `core/runners/ab_runner.py` |
| **S2** "Unknown" Option Added | the manipulation under study | same pass as S1 |
| **S3** Question Format Ablation | TFQ re-rendered as A/B/C letters | same pass (TFQ only) |
| **S4** Word Content Ablation | abstain word → synonym or random word | `experiments/C1_structural_trigger/S4_word_content_ablation/` |

### C2 — the model denies it can answer, even when it can (§4.2)

| Setting | What it does | Entry point |
| --- | --- | --- |
| **S5** w/o "Unknown" Option Rerun | remove the option, force a commitment | `core/runners/ab_runner.py` (`run_s5_rerun`) |
| **S6** Self-Diagnosis | the model attributes its own abstention | `experiments/C2_deny_yet_capable/S6_self_diagnosis/` |

### C3 — a later-layer output override (§4.3)

| Setting | What it does | Entry point |
| --- | --- | --- |
| **S7** Reasoning Traces Evaluation | DeBERTa NLI probe over whole traces | `experiments/C3_later_layer_override/S7_reasoning_traces_evaluation/` |
| **S8** Logit-Lens Representation Probe | OLMo-3-7B, 33 layers, steps `01_`…`06_` | `experiments/C3_later_layer_override/S8_logit_lens_representation_probe/` |

### C4 — a stable bias installed by instruction tuning (§4.4)

| Setting | What it does | Entry point |
| --- | --- | --- |
| **S9** Stability | 3 re-draws at default temperature; truly-Unknown mirror | `experiments/C4_stable_bias/S9_stability/` |
| **S10** Factor Analysis | difficulty, temperature, model size, alignment | `experiments/C4_stable_bias/S10_factor_analysis/` |
| **S11** Positional Biases | the abstain verb moves to slot 1 / 2 / 3 | `experiments/C4_stable_bias/S11_positional_biases/` |

### Appendix

| Appendix | Contents | Location |
| --- | --- | --- |
| C Additional Quantitative Analyses | Abs Rate–ΔAcc regression over the 18 cells of Table 1 | `reporting/abs_rate_dacc_regression.py` |
| E Mitigation | calibration suffix, stimulation + reflection | `core/runners/appendix_mitigation_runner.py` |
| E Remedy R2 | contrastive self-consistency override (post-hoc, no API calls) | `core/runners/appendix_remedy_r2_self_consistency.py` |

## Reproducing the reported numbers

`reporting/` rebuilds what the paper states, from `results/` alone. Nothing
there issues an API call.

| Script | Produces |
| --- | --- |
| `build_table1.py` | Table 1 — 24 cells, three metric rows per model |
| `abs_rate_dacc_regression.py` | the App. C regressions over the same 18 cells |
| `audit_parser_provenance.py` | which parser tier each reported label came from |

### One denominator

Every rate is scored on the set `core.result_schema.paired_keep_ids` returns:
the items that **both** settings of a paired contrast answered. An item is
dropped only when a setting returned nothing usable — an exhausted retry, a
content-filter refusal, a decoding collapse. A response that declines to commit
is kept, because refusing to commit is the behaviour under study, not a missing
measurement.

### One source per setting

Read the canonical summary for a setting, not a second run of the same prompt.
The S11 slot-C cell reproduces `build_judge_s2_prompt` byte for byte, but it is
a separate run: an analysis that pairs it with a standalone S1 sweep will
disagree with Table 1.

## Datasets

`dataset/` holds all eight files in one schema — no per-dataset loader:

| File | Type | Use |
| --- | --- | --- |
| `FLD.json`, `FOLIO.json` | TFQ | S1–S11 (500 answerable items each: 250 True + 250 False) |
| `FLD_unknown.json`, `FOLIO_unknown.json` | TFQ | S9 truly-Unknown subset (300 each) |
| `ARC.json`, `MMLU.json`, `MedQA.json`, `LogiQA.json` | MCQ | S1, S2 (500 each) |

Each record carries `id / source / task_type / question / context / options /
answer_idx`; `answer_idx == -1` marks a truly-Unknown item.
`core.dataset_loader` is the only reader.

## Labels

Predictions are letter strings: `A`–`D` (or `A`/`B` for TFQ), `UNKNOWN` for an
abstention, `UNPARSEABLE` when extraction failed. Parsing is deterministic and
tiered (strict exact match → lenient match); there is no LLM judge anywhere in
the pipeline.

## Citation

```bibtex
@misc{ling2026llmabstentionpromptartifact,
      title={LLM Abstention Can Be a Prompt Artifact, in Addition to Genuine Uncertainty}, 
      author={Zipeng Ling and Shuliang Liu and Yuehao Tang and Junqi Yang and Shenghong Fu and Seonil Son and Chen Huang and Kejia Huang and Yao Wan and Zhichao Hou and Xuming Hu},
      year={2026},
      eprint={2507.16199},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.16199}, 
}
```

## Contact

Questions and issues are welcome. For anything the issue tracker does not
cover, reach out to **zpling0816@gmail.com**.

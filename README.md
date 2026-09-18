<h1 align="center">LLM Abstention Can Be a Prompt Artifact,<br>in Addition to Genuine Uncertainty</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2507.16199"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2507.16199-b31b1b.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-3776ab">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
</p>


Adding an "Unknown" option to a True/False question makes a model abstain on
questions it can answer. We call this **Abstention Inflation**.


| Format | Samples | Accuracy without the “Unknown” option | Accuracy with the "Unknown" Option | Δ |
| --- | ---: | ---: | ---: | ---: |
| True-False Questions | 1,000 | 79.4% | **59.5%** | **−19.9%** |
| Multiple-Choice Questions | 2,000 | 84.5% | 83.3% | −1.2% |

The abstention does not result from uncertainty: removing the option again recovers **63.6%**
accuracy on exactly the samples that previously abstained.

## Installation

```bash
git clone https://github.com/ZpLing/Abstention-Inflation.git
cd Abstention-Inflation
pip install -r requirements.txt
cp configs/API_Config.template.yaml API_Config.yaml   # fill in your key
```

`API_Config.yaml` is git-ignored and holds the only credentials this code
reads. Everything else — which model, which datasets, where results land — is
in the experiment YAML.

## Quick start

Everything runs through `main.py`. The first argument is the setting; `all` is
the only way to run everything.

```bash
python main.py --list                 # every setting, its parts, and what --model means there
python main.py all                    # collect every gateway setting on every reported cell
python main.py all --stage analyze    # print every setting's numbers
python main.py S2                     # one setting: S2 with its S1 pair, 3 models x 6 datasets
```

## Repository layout

```
.
├── main.py                       dispatcher: the setting comes first, then
│                                 --part, --model, --dataset
├── infra/                        the method: prompts, parser, metrics, the
│                                 keep-set rule, and paired_pass, which runs
│                                 S1/S2/S3/S5 over one sample list
├── loader/                       everything that reads from disk: the config,
│                                 the datasets, the sample-fetch interface
├── experiments/                  run_S<n>_<setting>.py collects a setting,
│                                 analyze_S<n>.py reports it
│   ├── C1_structural_trigger/
│   │   ├── S1_baseline/
│   │   ├── S2_unknown_option_added/
│   │   ├── S3_question_format_ablation/
│   │   └── S4_word_content_ablation/
│   ├── C2_deny_yet_capable/
│   │   ├── S5_without_unknown_option_rerun/
│   │   └── S6_self_diagnosis/
│   ├── C3_later_layer_override/
│   │   ├── S7_reasoning_traces_evaluation/
│   │   └── S8_logit_lens_representation_probe/
│   └── C4_stable_bias/
│       ├── S9_stability/
│       ├── S10_factor_analysis/
│       └── S11_positional_biases/
├── configs/                      one YAML per (model, dataset) cell, named
│                                 for the settings it collects
├── dataset/                      the eight benchmark files, one schema
└── results/                      one folder per setting, named S<n>_<setting>
                                  to match the runner above; the main experiment
                                  is S1_baseline/ S2_unknown_option/
                                  S3_question_format/ S5_rerun/, each split by
                                  {tfq,mcq}/<model-slug>/ (S3 is TFQ-only and
                                  skips that level); S4, S9 and S10 keep their two
                                  halves as subfolders of one setting folder. Every file carries a
                                  "setting" field. infra/result_schema.py holds
                                  the registry (SETTING_DIRS) and load_cell(),
                                  which joins a cell's settings by item id.
```

**Note:** C3 has no `configs/` entry: S7 scores traces that are already on disk and S8
runs a local checkpoint, so neither reaches the gateway. `main.py S7` and
`main.py S8 --part …` pass their arguments on the command line.

## Running the settings

Arguments are read top-down: **setting → `--part` → `--model` → `--dataset`**.
The setting is required, and `all` at that level is the only way to run
everything. Below it, a level left out means every value the paper reports for
that setting; `all` at any level says the same explicitly.

```bash
python main.py S2 --model gemini-3.1-flash-lite --dataset FLD    # one cell
python main.py S4 --part random_words --model deepseek-v4-flash  # one half of a two-part setting
python main.py S9 --part persistence --model gpt-5.4-nano --dataset FOLIO
python main.py S3 --stage analyze                                # numbers only, nothing collected
python main.py S2 --model qwen3-max                              # any model the gateway serves
```

Any model the gateway serves works. A model outside the three the paper reports
borrows `gpt-5.4-nano`'s YAML for its credentials, keeps its own name and writes
under its own slug (`results/<setting>/…/<model_slug>/`). S1, S2, S3 and S5 come
out of one paired pass per cell; asking for one of them collects what it needs,
and S1 is always collected with S2.

| Setting | `--part` | `--model` ranges over | `--dataset` |
| --- | --- | --- | --- |
| S1 Baseline | — | gateway model | FLD FOLIO ARC MedQA MMLU LogiQA |
| S2 Unknown option added | — | gateway model | FLD FOLIO ARC MedQA MMLU LogiQA |
| S3 Question format ablation | — | gateway model | FLD FOLIO |
| S4 Word content ablation | `synonyms` `random_words` | gateway model | FLD FOLIO |
| S5 Without-Unknown rerun | — | gateway model | FLD FOLIO ARC MedQA MMLU LogiQA |
| S6 Self-diagnosis | — | gateway model | FLD FOLIO |
| S7 Reasoning traces | — | gateway model whose traces are scored | FLD FOLIO |
| S8 Logit lens | `download` `inference` `logit_lens` | Olmo-3-7B checkpoint key: `base` `sft` `instruct` `rl_zero` | FLD |
| S9 Stability | `perception` `persistence` | gateway model | FLD FOLIO |
| S10 Factor analysis | `temperature` `temperature_local` `size_alignment` `difficulty` | gateway model, or a checkpoint tag for the local parts | FLD FOLIO |
| S11 Positional biases | — | gateway model | FLD FOLIO |

`python main.py --list` prints the same table with each part's default models.
Follow-up settings read cells of earlier ones (S4, S6, S7 and S9 persistence
read the S1/S2/S5 cells); `main.py` checks they are on disk and names the
command that collects them if not.

### Settings that load a checkpoint

S7, S8 and the local S10 parts need `torch` and `transformers` and a model on
disk. `all` prints the command for each and moves on; they run when named.

```bash
python main.py S7                                             # NLI probe over the stored S1/S2 traces
python main.py S8 --part download                             # Olmo-3-7B checkpoints into models/
python main.py S8 --part inference                            # S1/S2 answers of the instruct checkpoint on FLD
python main.py S8 --part logit_lens                           # the base, sft and rl_zero probes the paper reports
python main.py S10 --part size_alignment --model gemma-4-E4B-it --model-path <checkout>
python main.py S10 --part temperature_local --model Olmo-3-7B-Instruct --model-path <checkout>
```

The local sweeps load one checkpoint per call, so `--model` names the tag and
`--model-path` its checkout; without a path the command is printed and skipped.

### Smoke run

`--limit` caps the items per cell, `--results-root` keeps the output away from
`results/`, and `--dry-run` prints every command without calling anything.

```bash
python main.py all --dry-run
python main.py all --model gpt-5.4-nano --limit 4 --results-root /tmp/ai_smoke
python main.py all --model gpt-5.4-nano --stage analyze --results-root /tmp/ai_smoke
```

### One YAML at a time

`--config` runs a single experiment YAML, the way `main.py` always has. The
YAML's `run_tasks` picks the runner; `--model`, `--dataset`, `--limit` and
`--results-root` still apply on top.

```bash
python main.py --config configs/C1_structural_trigger/S1_S3_TFQ_DeepSeek_V4_Flash.yaml
```

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

Questions and issues are welcome, please reach out to **zpling0816@gmail.com**.

## License

This project is released under the [MIT License](LICENSE).

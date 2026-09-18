<h1 align="center">LLM Abstention Can Be a Prompt Artifact,<br>in Addition to Genuine Uncertainty</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2507.16199"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2507.16199-b31b1b.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-3776ab">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
</p>


Adding an "Unknown" option to a True/False question makes a model abstain on
questions it can answer. We call this **Abstention Inflation**.


| Format | Samples | Accuracy without the “Unknown” option | Accuracy with the "Unknown" Option | Abs Rate |
| --- | ---: | ---: | ---: | ---: |
| True-False Questions | 1,000 | 79.4% | **59.5%** | **34.5%** |
| Multiple-Choice Questions | 2,000 | 84.5% | 83.3% | 2.3% |

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

```bash
# S1 + S2 + S3 and the S5 rerun, DeepSeek-V4-Flash on FLD and FOLIO
python main.py --config configs/C1_structural_trigger/S1_S3_TFQ_DeepSeek_V4_Flash.yaml

# Every setting's analysis prints its own numbers
python experiments/C1_structural_trigger/S2_unknown_option_added/analyze_S2.py
```

## Repository layout

```
.
├── main.py                       dispatcher; --config selects the runner
├── infra/                        the method: prompts, parser, metrics, the
│                                 keep-set rule, and paired_pass, which runs
│                                 S1/S2/S3/S5 over one sample list
├── loader/                       everything that reads from disk: the config,
│                                 the datasets, the sample-fetch interface
├── experiments/                  run_S<n>_<setting>.py collects a setting,
│                                 analyze_S<n>.py reports it
│   ├── C1_structural_trigger/
│   │   ├── S1_baseline/          run_S1_baseline.py + run.py
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
runs a local checkpoint, so neither reaches the gateway. Both take their
arguments on the command line.

## Running everything

The API settings come out of `main.py`; the rest own their runner. Every
command is run from the repo root and writes under `results/`.

### 1. Main experiments — S1, S2, S3 and the S5 rerun

One pass per (model, dataset) cell. S1/S2/S3 and S5 share it because the paper
compares them per sample, and scoring them from separate runs would compare
two different subsets of the dataset.

```bash
for m in DeepSeek_V4_Flash GPT_5_4_nano Gemini_3_1_Flash_Lite; do
  python main.py --config configs/C1_structural_trigger/S1_S3_TFQ_$m.yaml
  for ds in ARC MedQA MMLU LogiQA; do
    python main.py --config configs/C1_structural_trigger/S1_S2_${ds}_$m.yaml
  done
done
```

### 2. S4 — word content ablation

```bash
python experiments/C1_structural_trigger/S4_word_content_ablation/run_S4_synonyms.py
for m in DeepSeek_V4_Flash GPT_5_4_nano Gemini_3_1_Flash_Lite; do
  python experiments/C1_structural_trigger/S4_word_content_ablation/run_S4_random_words.py \
      --config configs/C1_structural_trigger/S4_random_words_$m.yaml
done
```

### 3. S6 — self-diagnosis

```bash
for m in DeepSeek_V4_Flash GPT_5_4_nano Gemini_3_1_Flash_Lite; do
  python main.py --config configs/C2_deny_yet_capable/S6_$m.yaml
done
```

### 4. S7 — reasoning traces

Reads the traces step 1 stored; needs a GPU but no API key.

```bash
python experiments/C3_later_layer_override/S7_reasoning_traces_evaluation/run_S7_reasoning_traces_evaluation.py
```

### 5. S8 — logit lens

Six numbered steps on a local OLMo-3-7B checkout.

```bash
S8=experiments/C3_later_layer_override/S8_logit_lens_representation_probe
python $S8/run_S8_step1_download.py                 # or bring your own checkout
python $S8/run_S8_step2_collect_samples.py
python $S8/run_S8_step3a_inference.py           --model_path <checkpoint>
python $S8/run_S8_step3b_base_baseline.py      --model_path <checkpoint>
python $S8/run_S8_step4_logit_lens.py
python $S8/run_S8_step5_wrong_prediction_baseline.py --model_path <checkpoint>
python $S8/run_S8_step6_suppression_detect.py
```

### 6. S9 — stability

```bash
for m in DeepSeek_V4_Flash GPT_5_4_nano Gemini_3_1_Flash_Lite; do
  python main.py --config configs/C4_stable_bias/S9_Perception_Unknown_labeled_Samples_$m.yaml
done

# the three re-draws, per (model, dataset)
python experiments/C4_stable_bias/S9_stability/run_S9_persistence_across_repeats.py \
    --summary results/S2_unknown_option/tfq/gpt_5.4_nano/FLD_gpt-5.4-nano.json \
    --dataset FLD --model gpt-5.4-nano --n_repeats 3 \
    --out results/S9_stability/Persistence_Across_Repeats/FLD_gpt-5.4-nano.json
```

### 7. S10 — factor analysis

Temperature, on the two checkpoints whose temperature the endpoint actually
applies. The local sweep takes one temperature per call, and the flags below
are the ones the reported cells were produced with -- the defaults would give
n=200 and truncate at 1024 tokens.

```bash
python experiments/C4_stable_bias/S10_factor_analysis/run_S10_temperature.py \
    --model gemini-3.1-flash-lite

for T in 0.0 0.3 0.7 1.0 1.5 2.0; do
  python experiments/C4_stable_bias/S10_factor_analysis/run_S10_local_sweep.py \
      --model_path <olmo-3-7b-instruct> --model_tag olmo3-instruct \
      --use_chat_template --settings S2 --n_per_class 250 \
      --max_new_tokens 8192 --batch_size 8 --top_k 20 --temperature $T \
      --out_dir results/S10_factor_analysis/temperature/olmo_topk20
done
```

Size and alignment: four Gemma sizes × {base, it}, at T=0. Pass
`--use_chat_template` for the `-it` checkpoints and leave it off for the base
ones -- that is the only difference between the two arms.

```bash
python experiments/C4_stable_bias/S10_factor_analysis/run_S10_local_sweep.py \
    --model_path <gemma-4-E4B-it> --model_tag gemma-4-E4B-it --use_chat_template \
    --n_per_class 250 --max_new_tokens 3072 --batch_size 8 \
    --out_dir results/S10_factor_analysis/size_alignment
```

### 8. S11 — positional biases

```bash
python experiments/C4_stable_bias/S11_positional_biases/run_S11_positional_biases.py \
    --model all --positions A B C --unified-labels
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

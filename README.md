<h1 align="center">LLM Abstention Can Be a Prompt Artifact,<br>in Addition to Genuine Uncertainty</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2507.16199"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2507.16199-b31b1b.svg"></a>
  <a href="https://arxiv.org/abs/2507.16199"><img alt="EMNLP 2026" src="https://img.shields.io/badge/EMNLP%202026-Main%20Conference-2c6fbb"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-3776ab">
</p>

<p align="center"><b>Accepted to the EMNLP 2026 Main Conference.</b></p>

Adding an "Unknown" option to a True/False question makes a model abstain on
questions it can answer. We call this **Abstention Inflation** and show it is a
structural property of the prompt rather than an expression of uncertainty.

Pooled over 3 models × 2 TFQ datasets at 500 items each, offering the option
costs **19.8 accuracy points** and pushes abstention to **34.5%**. The same
manipulation on 4-option MCQs, over the same models and the same 500-item
scale, costs **1.2 points**.

| Format | Items | Acc without the option | Acc with it | Abs Rate |
| --- | ---: | ---: | ---: | ---: |
| TFQ (FLD, FOLIO) | 3,000 | 79.4% | **59.5%** | **34.5%** |
| MCQ (ARC, MedQA, MMLU, LogiQA) | 6,000 | 84.5% | 83.3% | 2.3% |

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

## Running everything

The API settings come out of `main.py`; the rest own their runner. Every
command is run from the repo root and writes under `results/`.

### 1. Main experiments — S1, S2, S3 and the S5 rerun

One pass per (model, dataset) cell. S1/S2/S3 and S5 share it because the paper
compares them per item, and scoring them from separate runs would compare
different samples.

```bash
for m in DeepSeek_V4_Flash GPT_5_4_nano Gemini_3_1_Flash_Lite; do
  python main.py --config configs/C1_structural_trigger/S1_S3_TFQ_n500_$m.yaml
  for ds in ARC MedQA MMLU LogiQA; do
    python main.py --config configs/C1_structural_trigger/S1_S2_${ds}_n500_$m.yaml
  done
done
```

### 2. S4 — word content ablation

```bash
python experiments/C1_structural_trigger/S4_word_content_ablation/run_S4_synonyms_all_models.py
for m in DeepSeek_V4_Flash GPT_5_4_nano Gemini_3_1_Flash_Lite; do
  python experiments/C1_structural_trigger/S4_word_content_ablation/run_S4_random_words.py \
      --config configs/C1_structural_trigger/S4_random_words_$m.yaml
done
```

### 3. S6 — self-diagnosis

```bash
for m in DeepSeek_V4_Flash GPT_5_4_nano Gemini_3_1_Flash_Lite; do
  python main.py --config configs/C2_deny_yet_capable/S6_n500_$m.yaml
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
cd experiments/C3_later_layer_override/S8_logit_lens_representation_probe
python 01_download_OLMo3.py                 # or point --model_path at your own
python 02_collect_samples.py
python 03_run_OLMo_inference.py  --model_path <checkpoint>
python 03b_run_OLMo_base_baseline.py --model_path <checkpoint>
python 04_compute_logit_lens.py
python 05_compute_wrong_prediction_baseline.py --model_path <checkpoint>
python 06_suppression_detect.py
```

### 6. S9 — stability

```bash
for m in DeepSeek_V4_Flash GPT_5_4_nano Gemini_3_1_Flash_Lite; do
  python main.py --config configs/C4_stable_bias/S9_truly_unknown_n300_$m.yaml
done

# the three re-draws, per (model, dataset)
python experiments/C4_stable_bias/S9_stability/run_S9_persistence.py \
    --summary results/tfq_n500/nano/ab_summary_FLD_gpt-5.4-nano.json \
    --dataset FLD --model gpt-5.4-nano --n_repeats 3 \
    --out results/persistence_n500/s9_persistence_FLD_gpt-5.4-nano.json
```

### 7. S10 — factor analysis

```bash
# temperature, on the two checkpoints whose temperature the endpoint applies
python experiments/C4_stable_bias/S10_factor_analysis/run_S10_temperature_api.py \
    --model gemini-3.1-flash-lite
python experiments/C4_stable_bias/S10_factor_analysis/run_S10_local_hf_sweep.py \
    --model_path <olmo-instruct> --out_dir results/s10_temp_olmo_topk20

# size and alignment, four Gemma sizes x {base, it}
python experiments/C4_stable_bias/S10_factor_analysis/run_S10_local_hf_sweep.py \
    --model_path <gemma-checkpoint> --out_dir results/s10_gemma_n500
```

### 8. S11 — positional biases

```bash
python experiments/C4_stable_bias/S11_positional_biases/run_S11_positional_biases.py \
    --model all --positions A B C --unified-labels
```

### 9. Rebuild the reported numbers

Pure post-processing over `results/`; no API calls.

```bash
python reporting/build_table1.py                  # Table 1, all 24 cells
python reporting/abs_rate_dacc_regression.py      # the App. C regressions
python -m core.runners.appendix_remedy_r2_self_consistency   # App. E's R2
```

Each analysis script under `experiments/` prints the numbers for its own
setting; run it with `--help` to see what it takes.

### One denominator

Every rate is scored on the set `core.result_schema.paired_keep_ids` returns:
the items that **both** settings of a paired contrast answered. An item is
dropped only when a setting returned nothing usable — an exhausted retry, a
content-filter refusal, a decoding collapse. A response that declines to commit
is kept, because refusing to commit is the behaviour under study, not a missing
measurement.

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

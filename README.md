<h1 align="center">LLM Abstention Can Be a Prompt Artifact,<br>in Addition to Genuine Uncertainty</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2507.16199"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2507.16199-b31b1b.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-3776ab">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
</p>

Adding an "Unknown" option to a True/False question makes a model abstain on
questions it can answer. We call this **Abstention Inflation**.

| Format | Samples | Accuracy without the "Unknown" option | Accuracy with the "Unknown" option | Δ |
| --- | ---: | ---: | ---: | ---: |
| True-False Questions | 1,000 | 79.4% | **59.5%** | **−19.9%** |
| Multiple-Choice Questions | 2,000 | 84.5% | 83.3% | −1.2% |

## Installation

```bash
git clone https://github.com/ZpLing/Abstention-Inflation.git
cd Abstention-Inflation
pip install -r requirements.txt
cp configs/API_Config.template.yaml API_Config.yaml   # fill in your key
```

`API_Config.yaml` is git-ignored and holds the API key used to run this code.
The experiment configuration is in the YAMLs under `configs/`.

## Quick start

Everything runs through `main.py`. The first argument is the setting; `all` is
the only way to run everything.

```bash
python main.py --list                 # overview of every setting
python main.py all                    # collect every gateway setting on every reported cell
python main.py all --stage analyze    # print every setting's numbers
python main.py S2                     # one setting: S2 with its S1 pair, 3 models x 6 datasets
```

## Repository layout

```
.
├── main.py
├── infra/
├── loader/
├── experiments/
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
├── configs/
├── dataset/
└── results/
```

**Note:** C3 has no `configs/` entry: S7 scores traces that are already on disk and S8
runs a local checkpoint, so neither reaches the gateway. `main.py S7` and
`main.py S8` pass their arguments on the command line.

## Running the settings

```bash
python main.py S2 --model gemini-3.1-flash-lite --dataset FLD           # one cell
python main.py S4 --sub-setting random_words --model deepseek-v4-flash  # one sub-setting of S4
python main.py S9 --sub-setting persistence --model gpt-5.4-nano --dataset FOLIO  # one sub-setting, one cell
python main.py S3 --stage analyze                                       # analysis of experimental results
python main.py S2 --model qwen3-max                                     # any model available through your API
```

### Settings that load a checkpoint

```bash
python main.py S7                                        # NLI probe over the stored S1/S2 traces
python main.py S8                                        # download, inference, then the probes
python main.py S8 --model sft                            # probe the SFT checkpoint only
python main.py S8 --model sft --model-path xxxx          # load that checkpoint from your own directory
python main.py S10 --sub-setting size_alignment --model gemma-4-E4B-it --model-path xxxx
python main.py S10 --sub-setting temperature_local --model Olmo-3-7B-Instruct --model-path xxxx
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

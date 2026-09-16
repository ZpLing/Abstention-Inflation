# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Overview

Research code for the *Abstention Inflation* paper — "LLM Abstention Can Be a
Prompt Artifact, in Addition to Genuine Uncertainty". Adding an "Unknown"
option to a True/False question makes models abstain on questions they can
answer; the paper argues this is a structural prompt artifact rather than an
expression of uncertainty.

The repository is organised to mirror the paper: four claims (C1–C4), each
realised by a group of experimental settings (S1–S10). `README.md` holds the
full paper↔code map — read it before changing anything structural.

## Setup

```bash
pip install -r requirements.txt
cp configs/secrets.template.yaml secrets.yaml   # fill in api_key / base_url
```

Credentials resolve in `core/config_loader.py`, in increasing precedence:
the experiment YAML, then `secrets.yaml`, then an extensionless `config` file
at the repo root (gateway schema: `llm.{api_key,base_url,model}`). Both
credential files are git-ignored.

## Running experiments

```bash
python main.py --config configs/C1_structural_trigger/DeepSeek_R1_FLD_FOLIO.yaml
```

`run_tasks` in the YAML selects the runner: `main_experiment` (S1/S2/S3 + the
S5 rerun), `s6_self_diagnosis`, `s9_truly_unknown`, `s10_model_sweep`,
`appendix_mitigation`. Settings without a YAML driver (S4, S7, S8, S9
persistence, S10 temperature/difficulty) have their own scripts under
`experiments/Cx_*/Sy_*/`, each with a README.

## Architecture

Fully async: `main.py` calls `asyncio.run()` once; everything below is `async`.

```
main.py                      dispatcher
core/
  config_loader.py           YAML + secrets merge; block_key/get_block resolve
                             pre-rename config-block names
  dataset_loader.py          the only dataset reader — dataset/<name>.json
  label_scheme.py            per-dataset verbs / framing / instructions
  prompts.py                 prompt builders, named after the paper's settings
  llm_handler.py             async OpenAI-compatible client, temperature 0.0
  evaluator.py               deterministic output parser (strict → lenient)
  metrics.py                 Acc, Abs Rate, macro-F1, trace F1
  result_schema.py           canonical summary schema + pre-rename reader
  ab_runner.py               S1 / S2 / S3 + the S5 rerun
  s6_*, s9_*, s10_*          one runner per remaining setting
  appendix_*                 App. E mitigation and the R1/R2 remedies
experiments/Cx_claim/Sy_setting/   entry points + per-setting READMEs
```

## Conventions that matter

* **Setting names follow the paper.** The experiments were run under older
  names and two of them shifted meaning: the old `S5` is the paper's **S3**
  (question format ablation) and the old `S4` is the paper's **S5** (w/o
  "Unknown" rerun); the old `S3` (calibration suffix) is not a paper setting at
  all and now lives in the App. E mitigation code. `core/result_schema.py`
  documents the full mapping and normalises old result files on read — always
  load summaries through `load_summary()` rather than `json.load`.
* **Prompt strings are frozen.** Every builder in `core/prompts.py` reproduces
  the exact string that generated the reported numbers. Rename or refactor
  freely, but do not edit prompt text.
* **The abstain option is appended at prompt-build time.** The on-disk dataset
  is never modified.
* **`tools/`, `results/`, `archive/` and `paper/` are git-ignored** —
  local helpers, run outputs, retired code and the LaTeX source.

## Labels

Predictions are letter strings: `A`–`D` (or `A`/`B` for TFQ), `UNKNOWN` for an
abstention, `UNPARSEABLE` when extraction failed. `answer_idx == -1` marks a
truly-Unknown item (the S9 subset); every other setting filters to
`answer_idx >= 0`.

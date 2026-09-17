# Appendix

Analyses and controls that sit outside the ten reported settings.

## `C_quantitative_analyses/` — App. C

Post-hoc analyses over existing result files; no API calls.

| Script | Paper |
|---|---|
| `abs_rate_dacc_regression.py` | App. C "Abs Rate–ΔAcc Linear Regression" |
| `dataset_category_breakdown.py` | App. C "Dataset Modulation of Abstention Inflation" |

## App. E — Mitigation

The mitigation experiments live in `core/` because each is a runner rather than
a one-off script:

| Module | Intervention |
|---|---|
| `core/appendix_mitigation_runner.py` | two-stage stimulation + reflection on the abstaining samples |
| `core/appendix_remedy_r1_logit_calibration.py` | subtract the mean abstain-token logprob at the final-answer position |
| `core/appendix_remedy_r2_self_consistency.py` | override an abstention with the S1 answer when S1 committed (post-hoc, zero API calls) |

```bash
python main.py --config <yaml with run_tasks: ['appendix_mitigation']>
python -m core.appendix_remedy_r2_self_consistency --results_root results
```

The calibration-suffix prompt (S2 plus a note explaining when "Unknown" is
appropriate) is also part of App. E; enable it by adding
`calibration_suffix` to `main_experiment.settings`.

## `unreported_controls/`

Robustness checks run during review that no reported number depends on. Kept
because they are cheap to re-run and answer recurring questions:

| Script | Question it answers |
|---|---|
| `run_positional_bias.py`, `analyze_positional_bias.py` | does the abstain option's *position* in the list matter? |
| `run_no_cot_ablation.py`, `run_no_cot_vs_paper_baseline.py` | does the effect survive without CoT prompting? |
| `run_open_ended_abstention.py` | does an abstention *affordance* in free-form generation inflate abstention? (SelfAware) |
| `run_mcq_compound_option.py` | does a partially-true compound option behave like the abstain slot? |
| `run_prompt_stability_decomposition.py` | which items flip, and from what baseline state? |
| `probe_1d/` | 1-D structural probe over hidden states (AID → letter probe → attribution → ablation) |

Several of these read datasets that are not part of the paper (FEVER, BBH,
StrategyQA, SelfAware); those files were retired to `archive/legacy_datasets/`
and need restoring before the scripts will run.

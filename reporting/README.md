# Reporting

Rebuilds the numbers the paper states, from `results/` alone. Nothing here
issues an API call; all three scripts are pure post-processing, so they can be
re-run at any time to check a reported value against the data behind it.

| Script | Produces |
| --- | --- |
| `build_table1.py` | Table 1 — 24 cells, three metric rows per model |
| `abs_rate_dacc_regression.py` | the App. C regressions over the same 18 cells |
| `audit_parser_provenance.py` | which parser tier each reported label came from |

## One denominator

Every rate is scored on the set `core.result_schema.paired_keep_ids` returns:
the items that both settings of a paired contrast answered. An item is dropped
only when a setting returned nothing usable — an exhausted retry, a
content-filter refusal, a decoding collapse. A response that declines to commit
is kept, because a refusal to commit is the behaviour under study, not a
missing measurement.

This matters more than it sounds. The TFQ and MCQ summaries were written by
different code paths: the TFQ one records its own `n_scored` on this rule, the
MCQ one carries a `metrics` block computed over all 500 rows with an empty API
reply counted as a wrong answer. Reading each family's stored block puts two
denominators in one table, which is how Table 1 once reported a MedQA accuracy
2.3 points below the cells it was compared against.

## One source per setting

Read the canonical summary for a setting, not a second run of the same prompt.
The S11 slot-C cell reproduces `build_judge_s2_prompt` byte for byte, but it is
a separate run: analyses that paired it with a standalone S1 sweep reported an
S1 accuracy of 75.0% against a table that said 78.4%.

## Figures

The plotting scripts are not part of the release. They live under `tools/`,
which is git-ignored, and write into `figures/`.

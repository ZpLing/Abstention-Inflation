# S3 — Question Format Ablation

TFQ items (FLD, FOLIO) are re-rendered with MCQ-style letter
labels: **A = True, B = False, C = Unknown**. Every other surface
element (facts/premises, hypothesis, CoT instruction) is held
constant. This isolates whether the surface *question format*
(verb-coded TFQ vs. letter-coded MCQ) drives Abstention Inflation,
independently of the actual option set.

The manipulation moves Abs Rate by at most 5.9 points in any of the
6 cells (pooled mean 2.7), against a 31.6-point S1→S2 jump — format
alone cannot explain Abstention Inflation. There is no outlier: the
`DeepSeek-V4-Flash × FLD` cell the paper once flagged came from a Table 1
value (Acc 20.0 / Abs 25.0) that matches no result file and implies a
below-chance non-abstention accuracy; re-measured it is 45.2 / 39.8.

The condition was also re-run because the original prompt appended a
calibration note ("Select C. Unknown ONLY if …") that S2 does not
carry, which made it two manipulations rather than one. Use
`run_S3.py`; `core.prompts.build_judge_s3_format_prompt` is
now byte-for-byte the S2 prompt apart from the letter rendering, and
the superseded string is kept as
`build_judge_s3_format_prompt_calibrated`.

## Run

```bash
# current: clean prompt, full 500 samples, per-sample records
python experiments/C1_structural_trigger/S3_question_format_ablation/run_S3.py \
    --max-workers 30 --max-tokens 32768              # DeepSeek needs 16384
python experiments/C1_structural_trigger/S3_question_format_ablation/report_S3_clean.py
python experiments/C1_structural_trigger/S3_question_format_ablation/update_table1_s3.py
python experiments/C1_structural_trigger/S3_question_format_ablation/update_appendix_s3.py

# superseded ABRunner path
python experiments/C1_structural_trigger/S3_question_format_ablation/run.py \
    --config configs/C1_structural_trigger/<model>_<dataset>.yaml
```

`report_S3_clean.py` gates a cell on endpoint refusals, replies that hit
the generation cap before stating an answer, and whether those dropped
replies skew in proof depth. `--max-tokens 32768` is rejected outright by
the DeepSeek deployment; 16384 is its ceiling.

The analysis script `analyze_S3_vs_S2.py` re-scores S3 against S2
on the same items (paired McNemar reported in the paper).

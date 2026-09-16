# S11 — Positional Biases

Does abstention depend on *where* the "Unknown" option sits in the list, rather
than on its presence? The manipulation is the order of the three verbs in the
S2 prompt — not a re-rendering of the item as a letter-coded MCQ, which would
change the question format as well and confound S11 with S3.

| Slot | Prompt line | Note |
| ---- | ----------- | ---- |
| **A** (first)  | `Output one of: Unknown \| True \| False` | |
| **B** (second) | `Output one of: True \| Unknown \| False` | |
| **C** (third)  | `Output one of: True \| False \| Unknown` | byte-identical to `build_judge_s2_prompt` |

The slot letters name the position for file names and the CLI; the model never
sees a letter and still answers with a verb, parsed by the same
`Evaluator.parse_judge_tiered` the S1/S2 cells use. Because slot C *is* the S2
prompt, the paper's "the third-position condition matches the original S2
ordering" holds by construction rather than by resemblance.

| File | Purpose | Output |
| ---- | ------- | ------ |
| `run_S11_option_position.py` | Query the three slots on FLD / FOLIO. Retries transient failures, excludes content-filter refusals from the denominator, and gates a cell as `complete` only when it is safe to publish. | `results/positional_bias_n500/summary_unknown_{A,B,C}_<dataset>_<model>.json` |
| `analyze_S11_option_position.py` | Pool the cells, run the paired McNemar tests between positions, and emit the table behind Figure `positional_bias_control.pdf`. | stdout + `results/positional_bias_n500/positional_bias_report.md` |

`run_S11_option_position.py` also owns the plumbing S1 and S3 borrow — the
full-dataset loader, the unified True/False/Unknown scheme, and the
content-filter accounting — so all three settings are scored on one code path.

## Run

```bash
python experiments/C4_stable_bias/S11_option_position/run_S11_option_position.py \
    --model all --dataset all --positions A B C --full-dataset --unified-labels

python experiments/C4_stable_bias/S11_option_position/analyze_S11_option_position.py
```

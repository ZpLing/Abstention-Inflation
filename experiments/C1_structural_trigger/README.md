# C1 — Abstention Inflation is triggered by the structural presence of an extra option, not by genuine uncertainty

Four experimental settings, all driven by the same `ABRunner` over the
unified dataset format (`dataset/<name>.json`).

| Setting | Folder | Manipulation |
| ------- | ------ | ------------ |
| **S1** Baseline | `S1_baseline/` | Standard prompt, original label set (True/False for TFQs; A/B/C/D for MCQs). |
| **S2** "Unknown" Option Added | `S2_unknown_option_added/` | Append `Unknown` (TFQs) / `E. Unknown` (MCQs) to the option set. |
| **S3** Question Format Ablation | `S3_question_format_ablation/` | TFQ items re-rendered as MCQ-style letters (A = True, B = False, C = Unknown). |
| **S4** Word Content Ablation | `S4_word_content_ablation/` | Replace "Unknown" with synonyms (Indeterminate, "I don't know") and random words (Triangular, Cerulean). |

S1, S2, S3 all flow through `python main.py --config <yaml>` — the YAML
selects which of `["S1", "S2", "S3"]` to enable. S4 has its own scripts
because it varies the *third option label* rather than the option set itself.

See each subfolder's `README.md` and `run.py` for the exact configs and
expected outputs.

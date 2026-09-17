# C2 — Abstention Inflation makes the model deny it can answer, even when it can

Two settings, both operating on the **Abstention Inflation samples** produced
by S2 (= items whose gold label is True/False but the model chose Unknown).

| Setting | Folder | What it measures |
| ------- | ------ | ---------------- |
| **S5** w/o "Unknown" Option Rerun | `S5_without_unknown_rerun/` | Force the model to commit to True/False (multi-turn). Recovery rate well above chance ⇒ the answer is recoverable. |
| **S6** Self-Diagnosis | `S6_self_diagnosis/` | Ask the model whether its abstention was (A) subjective incapability or (B) truly-unknown. 95-100% pick B even though S5 just recovered the answer. |

Together: a model can answer (S5) yet sincerely denies it can (S6) — the
introspective gap that C2 names.

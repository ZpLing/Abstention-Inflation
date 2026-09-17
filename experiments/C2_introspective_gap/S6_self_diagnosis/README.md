# S6 — Self-Diagnosis

Paper §4.2.2, Figure 4 (right).

On the same Abstention Inflation samples used by S5, the model is asked to
attribute its own abstention to either

* **A** — subjective incapability ("it was too difficult, I gave up"), or
* **B** — the sample being objectively unanswerable.

Multi-turn: the S2 conversation is replayed, then the A/B question is appended
(`core.prompts.build_{judge,mcq}_s6_selfdiag_prompt`). The paper finds 95–100%
of these abstentions attributed to **B** — even though S5 just showed the model
answers correctly when the option is removed. That contradiction is the
introspective gap C2 names.

## Run

`run_S6_self_diagnosis.py` is the script that produced the reported numbers. It
consumes existing S1/S2 result files (no new S1/S2 calls) and queries only the
abstaining samples.

```bash
python experiments/C2_introspective_gap/S6_self_diagnosis/run_S6_self_diagnosis.py \
    --config configs/C2_introspective_gap/S6_DeepSeek_R1.yaml
```

`run.py` is a thin alternative that dispatches through `main.py` and drives
`core.s6_self_diagnosis_legacy_runner` — an earlier variant that additionally
cross-tabulates the A/B answer against S5 correctness. Prefer
`run_S6_self_diagnosis.py` when reproducing the paper.

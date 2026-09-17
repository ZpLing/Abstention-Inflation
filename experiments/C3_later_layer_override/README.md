# C3 — Abstention Inflation is a later-layer output override

Reasoning trace and mid-layer representations preserve the correct answer;
only the later layers introduce the "Unknown" override.

| Setting | Folder | What it does |
| ------- | ------ | ------------ |
| **S7** Reasoning Traces Evaluation | `S7_reasoning_traces_evaluation/` | F1T between generated traces under S1 vs S2 (invariance) + DeBERTa NLI probe on traces of Abstention Inflation samples (42.3 % NLI-recoverable; 22.3 % gold-label-stated). |
| **S8** Logit-Lens Representation Probe | `S8_logit_lens_representation_probe/` | Per-layer `log P(Unknown)` across 33 transformer layers for OLMo-3-7B {Base, Instruct, RL-Zero} on FLD + FOLIO. |

Both settings only re-use S1 / S2 prompts; **no new model calls** for S7
(it parses the Reasoning blocks of existing S1 / S2 transcripts). S8 has
its own inference loop because logit-lens needs raw hidden states, not just
the decoded text.

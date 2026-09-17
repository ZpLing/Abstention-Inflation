# C3 — no configs

S7 and S8 take no YAML. Both read data that is already on disk and run a local
model, so there is no gateway to configure:

* **S7** scores the S1 and S2 reasoning traces stored in
  `results/tfq_n500/` with a DeBERTa NLI encoder. Its only arguments are the
  batch size and how much of each trace to read.
* **S8** runs OLMo-3-7B from a local checkpoint through a numbered pipeline
  (`01_`…`07_`), each step taking its paths on the command line.

The directory is kept so the four claims line up here the way they do in
`experiments/` and in the paper.

# Configs

One YAML per (model, dataset) cell, grouped by the claim it serves. The file
names the gateway model, the datasets, and a `run_tasks` list that selects the
runner in `core/runners/`:

| `run_tasks` entry | Runner | Settings collected |
| --- | --- | --- |
| `main_experiment` | `ab_runner.ABRunner` | S1, S2, S3 and the S5 rerun, in one paired pass |
| `s6_self_diagnosis` | `s6_self_diagnosis_runner` | S6 |
| `s9_truly_unknown` | `s9_truly_unknown_runner` | S9's truly-Unknown mirror |
| `s10_model_sweep` | `s10_model_sweep_runner` | S10's size and alignment sweep |
| `appendix_mitigation` | `appendix_mitigation_runner` | App. E's post-hoc baseline |

S1, S2, S3 and S5 share a config because they share a pass: the paper compares
them per item, and scoring them from separate runs would compare different
samples of the dataset.

Settings with no entry here own their runner outright — S4, S7, S8, S9's
persistence re-draws, S10's temperature and difficulty sweeps, and S11 — and
take their arguments on the command line. See each setting's README.

Credentials never live here. `configs/secrets.template.yaml` shows the shape;
copy it to `secrets.yaml` (git-ignored) and fill in `api_key` / `base_url`.

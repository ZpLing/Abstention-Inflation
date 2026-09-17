# S10 — Factor Analysis

Probes the sensitivity of Abstention Inflation to **four** factors. Each
factor lives in its own script; results feed into Figures 7 and 8 of the
paper.

| Factor | Scripts | Reports |
| ------ | ------- | ------- |
| Temperature (T = 0, 0.3, 0.7, 1.0, 1.5, 2.0) | `run_S10_temperature_sweep.py`, `run_S10_temperature_sweep_generic.py`, `analyze_S10_temperature.py` | Fig. 7 right: Abs Rate flat across T ⇒ not stochastic. |
| Task difficulty (FLD proof depth, 1–9 steps) | `analyze_S10_difficulty.py` | Fig. 7 left: Abs Rate rises with depth (Spearman ρ ≈ +0.39). |
| Model size & alignment (Gemma family) | `run_S10_local_hf_sweep.py`, `run_S10_Gemma_sweep.sh`, configs `gemma_*` | Fig. 8 A/B/C: IT lifts Abs Rate at every size; IT Abs Rate is flat across size. |

## Run — temperature sweep

```bash
python experiments/C4_stable_bias/S10_factor_analysis/run_S10_temperature_sweep.py \
    --config configs/C4_stable_bias/temp_sweep_<model>.yaml
python experiments/C4_stable_bias/S10_factor_analysis/analyze_S10_temperature.py
```

## Run — difficulty (no new API calls)

`analyze_S10_difficulty.py` is a pure post-hoc analysis: it reads
`results/<run>/ab_summary_FLD_<model>.json` (produced by S2) and uses the
`depth` field bundled in `dataset/FLD.json` to compute Abs Rate per step
bin.

```bash
python experiments/C4_stable_bias/S10_factor_analysis/analyze_S10_difficulty.py
```

## Run — Gemma alignment & size sweep

```bash
# Single Gemma scale (one config = one Base / IT pair)
python experiments/C4_stable_bias/S10_factor_analysis/run_S10_local_hf_sweep.py \
    --config configs/C4_stable_bias/gemma_E2B_it.yaml

# Whole sweep (vLLM server lifecycle handled by the shell script)
bash experiments/C4_stable_bias/S10_factor_analysis/run_S10_Gemma_sweep.sh
```

The shell script expects the env vars `VLLM`, `PYTHON`, `EXPDIR`, `MODELS`
to be set (see the script header for placeholder hints) and a vLLM-served
endpoint reachable at `http://<VLLM_HOST>:8081/v1` (port overridable via
`VLLM_PORT`).

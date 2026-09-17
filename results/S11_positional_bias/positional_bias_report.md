# Positional Bias Control: Unknown Option Position (n=500, unified True/False/Unknown)

Abstention rate with the `Unknown` option at positions A, B, and C on FLD and FOLIO. All cells use the same 500 balanced samples (250 True + 250 False) and the unified True/False/Unknown label set; C is measured directly (not taken from the main table). Rows are labeled with the **actual API model id** that was queried.

## Abstention Rate by Position

| Model (API id) | Benchmark | A: Unknown | B: Unknown | C: Unknown | Max–Min (pp) |
|---|---|---:|---:|---:|---:|
| `deepseek-v4-flash` | FLD | 47.6% | 46.0% | 47.6% | 1.6 pp |
| `deepseek-v4-flash` | FOLIO | 19.6% | 19.4% | 19.2% | 0.4 pp |
| `gpt-5.4-nano` | FLD | 64.3% | 67.5% | 68.4% | 4.1 pp |
| `gpt-5.4-nano` | FOLIO | 22.0% | 25.0% | 24.6% | 3.0 pp |
| `gemini-3.1-flash-lite` | FLD | 40.0% | 36.4% | 37.8% | 3.6 pp |
| `gemini-3.1-flash-lite` | FOLIO | 17.6% | 16.2% | 16.0% | 1.6 pp |

*Abs Rate is computed over the valid subset (n_valid ≥ 496/500). A small number of FLD prompts are deterministically refused by the API content filter ("Sensitive word detected") and are excluded from the denominator.*

*Model ids are the exact strings sent to the API. If the paper uses different display names (e.g. "DeepSeek-V4-Flash", "Gemini-3.1-Flash-Lite"), map them deliberately — do not assume `deepseek-r1-distill-llama-8b` or `gemini-3.1-flash-lite` equal those names.*

## Mean Abstention Rate

| Unknown position | Mean Abs Rate |
|---|---:|
| A | 35.2% |
| B | 35.1% |
| C | 35.6% |

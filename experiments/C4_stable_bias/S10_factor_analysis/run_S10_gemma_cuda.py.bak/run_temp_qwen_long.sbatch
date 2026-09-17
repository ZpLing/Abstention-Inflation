#!/bin/bash
#SBATCH --job-name=s10qwnl
#SBATCH --account=aip-zhouyang
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=%x-%j.out
# S10(a) temperature sweep on Qwen3.5-9B, re-run with a larger generation
# budget and head truncation. Two separate defects in the 8192-token sweep:
#
#   Budget. Qwen3.5 thinks out loud, and on FLD it spends the whole budget
#   doing so: even the items that DID produce a `Final answer:` line used 82%
#   of 8192 tokens, and over half of all items were cut off mid-derivation.
#   Raising the cap is the only fix -- the reasoning is coherent, just
#   unfinished. FOLIO is not budget-bound (its finished items use 21%), but it
#   is re-run at the same settings so one curve is not spliced from two regimes.
#
#   Truncation. As with OLMo, T>=1.5 decoded into noise with no head cutoff:
#   trusted share 4.4% and 0.2%. top_k=20 is the vendor's own recommendation
#   for reasoning tasks on this model card, alongside the top_p=0.95 already
#   used here.
#
# 24576 leaves ~4k of the 32768-token context for the prompt. --time is two
# days rather than one: at 8192 the FLD cells already ran ~8h, and the cells
# that were being truncated will now run to completion.
#
# --export: DS, TEMP   (MNT, BS, TOPP, TOPK optional)
set -e
BASE=$HOME/scratch/abstention
export HF_HOME=$BASE/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd $BASE/software
echo "=== $(date) Qwen3.5-9B / $DS T=$TEMP mnt=${MNT:-24576} top_p=${TOPP:-0.95} top_k=${TOPK:-20} on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$BASE/venv/bin/python experiments/C4_stable_bias/S10_factor_analysis/run_S10_gemma_cuda.py \
    --model_path "$BASE/models/Qwen3.5-9B" --model_tag "Qwen3.5-9B" \
    --use_chat_template \
    --batch_size "${BS:-8}" --max_new_tokens "${MNT:-24576}" \
    --n_per_class 250 --settings S2 \
    --temperature "$TEMP" --top_p "${TOPP:-0.95}" --top_k "${TOPK:-20}" \
    --datasets $DS --out_dir "$BASE/results/s10_temp_qwen_mnt24576_topk20"
echo "=== $(date) DONE Qwen3.5-9B / $DS T=$TEMP ==="

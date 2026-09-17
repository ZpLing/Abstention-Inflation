#!/bin/bash
#SBATCH --job-name=s10olmk
#SBATCH --account=aip-zhouyang
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out
# S10(a) temperature sweep on OLMo-3-7B-Instruct, re-run under head truncation.
#
# The first sweep passed no top_k at all -- the runner hard-coded top_k=0, which
# disables the filter -- and above T=1 the model stopped writing English: only
# 13-20% of its words at T>=1.5 came from the vocabulary it uses at T=0, and no
# item at T=2.0 emitted a `Final answer:` line at all. Nucleus alone, at the
# checkpoint's own top_p=0.95, did not hold it together. top_k=20 is the
# truncation a served endpoint normally applies on the caller's behalf, which
# is why Gemini stays at 1.2% unparseable at T=2.0; passing it makes the local
# sweep comparable to the API one.
#
# The whole grid is re-run, not only the two broken cells: mixing a truncated
# and an untruncated regime inside one temperature curve would confound the
# very axis being measured.
#
# --export: DS, TEMP   (MNT, BS, TOPK optional)
set -e
BASE=$HOME/scratch/abstention
export HF_HOME=$BASE/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd $BASE/software
echo "=== $(date) olmo3-instruct / $DS T=$TEMP mnt=${MNT:-8192} top_k=${TOPK:-20} on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$BASE/venv/bin/python experiments/C4_stable_bias/S10_factor_analysis/run_S10_gemma_cuda.py \
    --model_path "$HOME/scratch/olmo3/instruct" --model_tag "olmo3-instruct" \
    --use_chat_template \
    --batch_size "${BS:-8}" --max_new_tokens "${MNT:-8192}" \
    --n_per_class 250 --settings S2 \
    --temperature "$TEMP" --top_k "${TOPK:-20}" \
    --datasets $DS --out_dir "$BASE/results/s10_temp_olmo_topk20"
echo "=== $(date) DONE olmo3-instruct / $DS T=$TEMP ==="

#!/usr/bin/env bash
# S10 Alignment & Size sweep: brings up vLLM for each Gemma checkpoint
# (Base + IT pairs at four scales), runs the matching YAML config, and
# tears the server down before moving on.
#
# Fill in the env-var paths below for your local setup, then:
#   bash run_S10_Gemma_sweep.sh
set -e

# === Local environment (override as needed) ==================================
: "${VLLM:?Set VLLM to the vllm binary path, e.g. /opt/envs/vllm/bin/vllm}"
: "${PYTHON:?Set PYTHON to the python interpreter, e.g. /opt/envs/exp/bin/python3.11}"
: "${EXPDIR:?Set EXPDIR to the software/ root holding configs/, main.py, ...}"
: "${MODELS:?Set MODELS to the directory holding the downloaded Gemma checkpoints}"
LOG="${LOG:-./gemma_exp_run.log}"

exec >> "$LOG" 2>&1
echo "====== $(date) START ======"

if [ -n "${TORCH_GATE:-/tmp/torch_upgrade.log}" ]; then
    echo "Waiting for torch upgrade (gate: ${TORCH_GATE})..."
    until grep -q 'DONE' "${TORCH_GATE}" 2>/dev/null; do sleep 15; done
    echo "torch upgrade done."
fi

wait_for_download() {
    local name=$1 flag=$2
    if [ ! -f "$flag" ]; then
        echo "Waiting for $name download (flag: $flag)..."
        until [ -f "$flag" ]; do sleep 30; done
    fi
    echo "$name ready."
}

run_model() {
    local MODEL_NAME=$1
    local MODEL_PATH=$2
    local CONFIG=$3
    local EXTRA_ARGS=${4:-}

    echo "--- [$(date +%H:%M)] Starting vLLM: $MODEL_NAME ---"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" $VLLM serve "$MODEL_PATH" \
        --served-model-name "$MODEL_NAME" \
        --port "${VLLM_PORT:-8081}" --max-model-len 4096 \
        $EXTRA_ARGS \
        >> "/tmp/vllm_${MODEL_NAME}.log" 2>&1 &
    local VLLM_PID=$!

    local ready=0
    for i in $(seq 1 30); do
        sleep 10
        if curl -sf "http://localhost:${VLLM_PORT:-8081}/health" > /dev/null 2>&1; then
            echo "vLLM ready (${i}0s)"; ready=1; break
        fi
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo "ERROR: vLLM died. Check /tmp/vllm_${MODEL_NAME}.log"; exit 1
        fi
    done
    [ $ready -eq 0 ] && { echo "ERROR: vLLM timeout"; kill $VLLM_PID 2>/dev/null; exit 1; }

    echo "--- Running: $CONFIG ---"
    cd "$EXPDIR"
    $PYTHON main.py --config "configs/$CONFIG"

    echo "--- Done: $MODEL_NAME, stopping vLLM ---"
    kill $VLLM_PID 2>/dev/null
    sleep 5
}

# === Six checkpoints, sequenced by download readiness ========================

run_model "gemma-4-E2B-it" "$MODELS/gemma-4-E2B-it" "C4_stable_bias/gemma_E2B_it.yaml"
run_model "gemma-4-E2B"    "$MODELS/gemma-4-E2B"    "C4_stable_bias/gemma_E2B_base.yaml"

wait_for_download "gemma-4-E4B-it" "/tmp/dl_gemma_e4b_it_done"
run_model "gemma-4-E4B-it" "$MODELS/gemma-4-E4B-it" "C4_stable_bias/gemma_E4B_it.yaml"

wait_for_download "gemma-4-E4B" "/tmp/dl_gemma_e4b_done"
run_model "gemma-4-E4B" "$MODELS/gemma-4-E4B" "C4_stable_bias/gemma_E4B_base.yaml"

wait_for_download "gemma-4-26B-A4B-it-AWQ" "/tmp/dl_gemma_26b_it_done"
run_model "gemma-4-26B-A4B-it" "$MODELS/gemma-4-26B-A4B-it-AWQ" \
    "C4_stable_bias/gemma_26B_A4B_it.yaml" "--quantization awq"

wait_for_download "gemma-4-31B-it-AWQ" "/tmp/dl_gemma_31b_it_done"
run_model "gemma-4-31B-it" "$MODELS/gemma-4-31B-it-AWQ" \
    "C4_stable_bias/gemma_31B_it.yaml" "--quantization awq"

echo "====== $(date) ALL DONE ======"

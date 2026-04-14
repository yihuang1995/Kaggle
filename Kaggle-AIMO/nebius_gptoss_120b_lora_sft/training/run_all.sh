#!/usr/bin/env bash
# run_all.sh — End-to-end orchestration: setup → download → train → eval
#
# Run each section manually or run the full script:
#   bash run_all.sh
#
# To run just one step:
#   bash run_all.sh setup
#   bash run_all.sh download
#   bash run_all.sh smoke_test
#   bash run_all.sh phase1
#   bash run_all.sh merge
#   bash run_all.sh phase2
#   bash run_all.sh eval_baseline
#   bash run_all.sh eval_phase1
#   bash run_all.sh eval_phase2
#   bash run_all.sh compare

set -e

NEBIUS_DIR="$HOME/nebius_training"
DATA_DIR="$HOME/data"
MODEL_DIR="$HOME/models"
BASE_MODEL="$MODEL_DIR/gpt-oss-120b"
MERGED_MODEL="$MODEL_DIR/gpt-oss-120b-phase1-merged"
PHASE1_ADAPTER="$HOME/lora_phase1"
PHASE2_ADAPTER="$HOME/lora_phase2"
REF_PHASE1="$HOME/sft_ref_phase1"
REF_PHASE2="$HOME/sft_ref_phase2"
REF_VENV="$HOME/venv-gptoss-ref"
HF_TOKEN=$(cat "$NEBIUS_DIR/hf_key.txt")
VLLM_PORT=8000
GPU_ID="${GPU_ID:-0}"
NPROC="${NPROC:-8}"

source ~/venv/bin/activate

# ── Helper: start vLLM server ──────────────────────────────────────────────────
start_vllm() {
    local model_path=$1
    echo "Starting vLLM server for $model_path..."
    python -m vllm.entrypoints.openai.api_server \
        --model "$model_path" \
        --dtype auto \
        --kv-cache-dtype fp8_e4m3 \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.92 \
        --port $VLLM_PORT &
    VLLM_PID=$!
    echo "vLLM PID: $VLLM_PID"
    # Wait for server to be ready
    echo "Waiting for vLLM server to start..."
    for i in $(seq 1 60); do
        if curl -s "http://localhost:$VLLM_PORT/v1/models" > /dev/null 2>&1; then
            echo "vLLM server ready."
            return
        fi
        sleep 5
    done
    echo "ERROR: vLLM server did not start within 5 minutes"
    exit 1
}

stop_vllm() {
    if [ -n "$VLLM_PID" ]; then
        echo "Stopping vLLM (PID $VLLM_PID)..."
        kill $VLLM_PID 2>/dev/null || true
        wait $VLLM_PID 2>/dev/null || true
        VLLM_PID=""
    fi
}

# ── Step 1: Setup ──────────────────────────────────────────────────────────────
step_setup() {
    echo "=== STEP: Environment Setup ==="
    bash "$NEBIUS_DIR/setup_nebius.sh"
}

# ── Step 2: Download model ─────────────────────────────────────────────────────
step_download() {
    echo "=== STEP: Download GPT-OSS 120B ==="
    mkdir -p "$MODEL_DIR"
    export HF_HUB_ENABLE_HF_TRANSFER=1
    export HF_TOKEN
    hf auth login --token "$HF_TOKEN"
    hf download openai/gpt-oss-120b \
        --local-dir "$BASE_MODEL"
    echo "Model downloaded to $BASE_MODEL"
}

# ── Step 3: Smoke test (verify pipeline before full training) ──────────────────
step_smoke_test() {
    echo "=== STEP: Smoke Test (reference model, 8-GPU model-parallel) ==="
    source "$REF_VENV/bin/activate"
    torchrun --standalone --nproc_per_node="$NPROC" \
        "$NEBIUS_DIR/sft_train_reference.py" --phase 1 --smoke-test
    echo "Smoke test passed!"
}

# ── Step 4: Phase 1 training ───────────────────────────────────────────────────
step_phase1() {
    echo "=== STEP: Phase 1 Training (reference model, 8-GPU model-parallel) ==="
    # Run in screen so it survives SSH disconnection
    screen -dmS phase1 bash -c "
        source $REF_VENV/bin/activate
        torchrun --standalone --nproc_per_node=$NPROC $NEBIUS_DIR/sft_train_reference.py --phase 1 \
            2>&1 | tee ~/logs/phase1.log
        echo 'Phase 1 done' > ~/logs/phase1_done.flag
    "
    echo "Phase 1 started in screen session 'phase1'"
    echo "Monitor: screen -r phase1   or   tail -f ~/logs/phase1.log"
    echo "Waiting for completion..."
    while [ ! -f ~/logs/phase1_done.flag ]; do sleep 60; done
    echo "Phase 1 complete!"
}

# ── Step 5: Merge LoRA ─────────────────────────────────────────────────────────
step_merge() {
    echo "=== STEP: Merge Phase 1 LoRA into base model ==="
    python "$NEBIUS_DIR/merge_lora.py" \
        --base "$BASE_MODEL" \
        --adapter "$PHASE1_ADAPTER" \
        --output "$MERGED_MODEL"
    echo "Merged model saved to $MERGED_MODEL"
}

# ── Step 6: Phase 2 training ───────────────────────────────────────────────────
step_phase2() {
    echo "=== STEP: Phase 2 Training (resume reference checkpoint) ==="
    mkdir -p ~/logs
    screen -dmS phase2 bash -c "
        source $REF_VENV/bin/activate
        torchrun --standalone --nproc_per_node=$NPROC $NEBIUS_DIR/sft_train_reference.py \
            --phase 2 --resume-from $REF_PHASE1 --output-dir $REF_PHASE2 \
            2>&1 | tee ~/logs/phase2.log
        echo 'Phase 2 done' > ~/logs/phase2_done.flag
    "
    echo "Phase 2 started in screen session 'phase2'"
    echo "Monitor: screen -r phase2   or   tail -f ~/logs/phase2.log"
    while [ ! -f ~/logs/phase2_done.flag ]; do sleep 60; done
    echo "Phase 2 complete!"
}

# ── Step 7: Eval baseline ──────────────────────────────────────────────────────
step_eval_baseline() {
    echo "=== STEP: Eval Baseline (500 problems) ==="
    start_vllm "$BASE_MODEL"
    python "$NEBIUS_DIR/eval_sft.py" --model "$BASE_MODEL" --tag baseline
    stop_vllm
}

# ── Step 8: Eval Phase 1 ──────────────────────────────────────────────────────
step_eval_phase1() {
    echo "=== STEP: Eval Phase 1 merged model (500 problems) ==="
    start_vllm "$MERGED_MODEL"
    python "$NEBIUS_DIR/eval_sft.py" --model "$MERGED_MODEL" --tag phase1
    stop_vllm
}

# ── Step 9: Eval Phase 2 ──────────────────────────────────────────────────────
step_eval_phase2() {
    echo "=== STEP: Eval Phase 2 model (500 problems) ==="
    # Merge phase 2 adapter first
    PHASE2_MERGED="$MODEL_DIR/gpt-oss-120b-phase2-merged"
    if [ ! -d "$PHASE2_MERGED" ]; then
        python "$NEBIUS_DIR/merge_lora.py" \
            --base "$MERGED_MODEL" \
            --adapter "$PHASE2_ADAPTER" \
            --output "$PHASE2_MERGED"
    fi
    start_vllm "$PHASE2_MERGED"
    python "$NEBIUS_DIR/eval_sft.py" --model "$PHASE2_MERGED" --tag phase2
    stop_vllm
}

# ── Step 10: Compare ──────────────────────────────────────────────────────────
step_compare() {
    echo "=== STEP: Compare Results ==="
    python "$NEBIUS_DIR/eval_sft.py" --compare
}

# ── Main dispatcher ────────────────────────────────────────────────────────────
mkdir -p ~/logs

case "${1:-all}" in
    setup)         step_setup ;;
    download)      step_download ;;
    smoke_test)    step_smoke_test ;;
    phase1)        step_phase1 ;;
    merge)         step_merge ;;
    phase2)        step_phase2 ;;
    eval_baseline) step_eval_baseline ;;
    eval_phase1)   step_eval_phase1 ;;
    eval_phase2)   step_eval_phase2 ;;
    compare)       step_compare ;;
    all)
        step_setup
        step_download
        step_smoke_test
        step_eval_baseline
        step_phase1
        step_merge
        step_eval_phase1
        step_phase2
        step_eval_phase2
        step_compare
        ;;
    *)
        echo "Unknown step: $1"
        echo "Valid steps: setup download smoke_test phase1 merge phase2 eval_baseline eval_phase1 eval_phase2 compare all"
        exit 1
        ;;
esac

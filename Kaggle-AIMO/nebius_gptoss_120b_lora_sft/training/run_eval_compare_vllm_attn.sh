#!/usr/bin/env bash

set -euo pipefail

NEBIUS_DIR="${NEBIUS_DIR:-$HOME/nebius_training}"
VENV_DIR="${VENV_DIR:-$HOME/venv}"
BASE_MODEL="${BASE_MODEL:-$HOME/models/gpt-oss-120b}"
ADAPTER_DIR="${ADAPTER_DIR:-$HOME/peft_sft_phase2_attn}"
RESULTS_DIR="${RESULTS_DIR:-$HOME/eval_results}"
LOG_DIR="${LOG_DIR:-$HOME/logs}"
PORT="${PORT:-8000}"
NPROC="${NPROC:-8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_LORA_RANK="${MAX_LORA_RANK:-64}"
BASE_MODEL_NAME="${BASE_MODEL_NAME:-baseline_vllm}"
POST_MODEL_NAME="${POST_MODEL_NAME:-sft_attn}"
BASE_TAG="${BASE_TAG:-baseline_vllm_attn}"
POST_TAG="${POST_TAG:-phase2_attn_vllm}"
N_PROBLEMS="${N_PROBLEMS:-0}"

source "$VENV_DIR/bin/activate"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

VLLM_PID=""

cleanup() {
  if [[ -n "${VLLM_PID}" ]]; then
    kill "$VLLM_PID" >/dev/null 2>&1 || true
    wait "$VLLM_PID" >/dev/null 2>&1 || true
    VLLM_PID=""
  fi
}

trap cleanup EXIT

wait_for_server() {
  local expected=$1
  local url="http://127.0.0.1:${PORT}/v1/models"

  for _ in $(seq 1 240); do
    if curl -sf "$url" | grep -q "$expected"; then
      return 0
    fi
    sleep 5
  done

  echo "Timed out waiting for vLLM model '$expected' on $url" >&2
  return 1
}

start_vllm() {
  local log_file=$1
  shift

  cleanup

  python -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL" \
    --served-model-name "$BASE_MODEL_NAME" \
    --tensor-parallel-size "$NPROC" \
    --dtype auto \
    --quantization mxfp4 \
    --kv-cache-dtype fp8_e4m3 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --port "$PORT" \
    "$@" \
    >"$log_file" 2>&1 &

  VLLM_PID=$!
}

run_eval() {
  local model_name=$1
  local tag=$2
  local n_args=()

  if [[ "$N_PROBLEMS" != "0" ]]; then
    n_args=(--n "$N_PROBLEMS")
  fi

  python "$NEBIUS_DIR/eval_sft.py" \
    --url "http://127.0.0.1:${PORT}/v1" \
    --model "$model_name" \
    --tag "$tag" \
    "${n_args[@]}"
}

echo "=== Baseline vLLM eval ==="
start_vllm "$LOG_DIR/vllm_eval_baseline.log"
wait_for_server "$BASE_MODEL_NAME"
run_eval "$BASE_MODEL_NAME" "$BASE_TAG"

echo "=== Post-SFT attention-only LoRA vLLM eval ==="
start_vllm "$LOG_DIR/vllm_eval_post.log" \
  --enable-lora \
  --max-loras 1 \
  --max-lora-rank "$MAX_LORA_RANK" \
  --lora-modules "${POST_MODEL_NAME}=${ADAPTER_DIR}"
wait_for_server "$POST_MODEL_NAME"
run_eval "$POST_MODEL_NAME" "$POST_TAG"

echo "=== Compare saved results ==="
python "$NEBIUS_DIR/eval_sft.py" --compare

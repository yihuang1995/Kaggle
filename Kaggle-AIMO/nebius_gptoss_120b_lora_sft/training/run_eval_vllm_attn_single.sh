#!/usr/bin/env bash

set -euo pipefail

NEBIUS_DIR="${NEBIUS_DIR:-$HOME/nebius_training}"
VENV_DIR="${VENV_DIR:-$HOME/venv}"
BASE_MODEL="${BASE_MODEL:-$HOME/models/gpt-oss-120b}"
ADAPTER_DIR="${ADAPTER_DIR:-$HOME/peft_sft_phase1_attn}"
RESULTS_DIR="${RESULTS_DIR:-$HOME/eval_results}"
LOG_DIR="${LOG_DIR:-$HOME/logs}"
PORT="${PORT:-8000}"
NPROC="${NPROC:-8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_LORA_RANK="${MAX_LORA_RANK:-64}"
MODEL_NAME="${MODEL_NAME:-sft_attn_step1000}"
TAG="${TAG:-phase1_step1000_attn_vllm}"
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

run_eval() {
  local n_args=()

  if [[ "$N_PROBLEMS" != "0" ]]; then
    n_args=(--n "$N_PROBLEMS")
  fi

  python "$NEBIUS_DIR/eval_sft.py" \
    --url "http://127.0.0.1:${PORT}/v1" \
    --model "$MODEL_NAME" \
    --tag "$TAG" \
    "${n_args[@]}"
}

python -m vllm.entrypoints.openai.api_server \
  --model "$BASE_MODEL" \
  --served-model-name "$MODEL_NAME" \
  --tensor-parallel-size "$NPROC" \
  --dtype auto \
  --quantization mxfp4 \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --port "$PORT" \
  --enable-lora \
  --max-loras 1 \
  --max-lora-rank "$MAX_LORA_RANK" \
  --lora-modules "${MODEL_NAME}=${ADAPTER_DIR}" \
  >"$LOG_DIR/vllm_eval_${TAG}.log" 2>&1 &

VLLM_PID=$!

wait_for_server "$MODEL_NAME"
run_eval

#!/usr/bin/env bash
# switch_to_phase2.sh — wait for phase 1 step 1000, save cleanly, start phase 2 for 10h.
set -euo pipefail

LOG_P1=~/logs/phase1_ref_lora_1024.log
LOG_P2=~/logs/phase2_ref_lora.log
CKPT_DIR=~/sft_ref_lora_phase1
VENV=~/venv-gptoss-ref

echo "[switch] $(date -u) — watching for step 1000 in $LOG_P1 ..."

# ── 1. Wait for step 1000 log line ──────────────────────────────────────────
while true; do
    if grep -qP "^step=01000 " "$LOG_P1" 2>/dev/null; then
        echo "[switch] $(date -u) — step=01000 found in log."
        break
    fi
    sleep 15
done

# ── 2. Wait for checkpoint at step 1000 to be written ───────────────────────
echo "[switch] Waiting for checkpoint save..."
while true; do
    if grep -qP "Saved checkpoint.*at step 1000" "$LOG_P1" 2>/dev/null; then
        echo "[switch] $(date -u) — checkpoint confirmed at step 1000."
        break
    fi
    sleep 5
done

# ── 3. Verify checkpoint files exist ────────────────────────────────────────
if [[ ! -f "$CKPT_DIR/trainable_state.pt" ]]; then
    echo "[ERROR] trainable_state.pt missing at $CKPT_DIR — aborting." >&2
    exit 1
fi
if [[ ! -f "$CKPT_DIR/trainer_state.json" ]]; then
    echo "[ERROR] trainer_state.json missing at $CKPT_DIR — aborting." >&2
    exit 1
fi
echo "[switch] Checkpoint files verified:"
ls -lh "$CKPT_DIR/"

# ── 4. Kill phase 1 gracefully ───────────────────────────────────────────────
echo "[switch] $(date -u) — sending SIGTERM to phase 1 processes..."
pkill -TERM -f "sft_train_reference_lora.py" || true
sleep 10
# Force-kill any stragglers
pkill -KILL -f "sft_train_reference_lora.py" 2>/dev/null || true
sleep 3
echo "[switch] Phase 1 processes stopped."

# ── 5. Start phase 2 with 10-hour timeout ───────────────────────────────────
echo "[switch] $(date -u) — starting phase 2 (10-hour timeout)..."
cd ~/nebius_training
source "$VENV/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# timeout 36000s = 10 hours; sends SIGTERM → last 100-step checkpoint is preserved
timeout 36000 torchrun --standalone --nproc_per_node=8 \
    sft_train_reference_lora.py \
    --phase 2 \
    --resume-from "$CKPT_DIR" \
    --max-length 1024 \
    2>&1 | tee "$LOG_P2"

EXIT=$?
if [[ $EXIT -eq 124 ]]; then
    echo "[switch] $(date -u) — 10-hour limit reached, phase 2 stopped cleanly."
else
    echo "[switch] $(date -u) — phase 2 finished (exit $EXIT)."
fi

#!/bin/bash
set -e
source ~/venv/bin/activate
cd ~/nebius_training

echo "=== Phase 1 (2 epochs, DeepSpeed ZeRO-3, 8×H100) ==="
torchrun --nproc_per_node=8 sft_train.py --phase 1

echo "=== Merging Phase 1 LoRA ==="
python merge_lora.py \
    --base ~/models/gpt-oss-120b \
    --adapter ~/lora_phase1 \
    --output ~/models/gpt-oss-120b-phase1-merged

echo "=== Phase 2 (1 epoch, DeepSpeed ZeRO-3, 8×H100) ==="
torchrun --nproc_per_node=8 sft_train.py --phase 2 \
    --base-model ~/models/gpt-oss-120b-phase1-merged

echo "=== All done ==="

#!/usr/bin/env bash
# setup_nebius.sh — One-time environment setup for Nebius H200
# Run: bash setup_nebius.sh

set -e

echo "=== Installing system packages ==="
sudo apt-get update -qq
sudo apt-get install -y python3-full python3-venv git screen htop nvtop

echo "=== Creating venv ==="
python3 -m venv ~/venv
source ~/venv/bin/activate

echo "=== Installing PyTorch (CUDA 12.4) ==="
pip install --upgrade pip
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "=== Installing training stack ==="
pip install \
    "transformers>=4.57.0" \
    datasets \
    peft \
    trl \
    accelerate \
    scipy \
    openai-harmony

echo "=== Installing vLLM and inference tools ==="
pip install vllm==0.11.2 openai

echo "=== Installing HuggingFace tools ==="
pip install huggingface_hub hf_transfer "huggingface_hub[cli]"

echo "=== Verifying GPU ==="
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {props.name} ({props.total_memory/1e9:.0f} GB)')
"

echo ""
echo "=== Setup complete. To activate venv: source ~/venv/bin/activate ==="

#!/usr/bin/env bash
# Environment setup for Nebius H100 server
# Run once after SSHing in for the first time.

# Install python3-full if not already present
sudo apt-get install -y python3-full python3-venv

# Create a venv (do this once)
python3 -m venv ~/venv

# Activate it (do this every time you SSH in)
source ~/venv/bin/activate

# Install dependencies
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets peft trl accelerate scipy
pip install huggingface_hub hf_transfer "huggingface_hub[cli]"
pip install vllm==0.11.2
pip install openai-harmony

# Login to HuggingFace (do this once)
hf auth login

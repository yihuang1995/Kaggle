"""
Math Model Downloader for Nebius H100 Server

Usage:
  python download_models.py --model list
  python download_models.py --model gpt-oss-120b
  python download_models.py --model all_single_gpu   # only models fitting 1x H100
  python download_models.py --model qwen3-32b
"""

import os
import argparse
import subprocess
import sys

# ── activate fast parallel downloads ──────────────────────────────────────────
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# ── model registry ─────────────────────────────────────────────────────────────
MODELS = {
    "gpt-oss-120b": {
        "repo": "openai/gpt-oss-120b",
        "dir":  "/data/models/gpt-oss-120b",
        "size": "~65 GB",
        "gpus": 1,
        "note": "Requires HF login + license acceptance at hf.co/openai/gpt-oss-120b",
    },
    "deepseek-r1-0528": {
        "repo": "deepseek-ai/DeepSeek-R1-0528",
        "dir":  "/data/models/DeepSeek-R1-0528",
        "size": "~720 GB",
        "gpus": 8,
        "note": "Very large — needs 8x H100. Plan ~2-3 hrs download time.",
    },
    "deepseek-r1-distill-8b": {
        "repo": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        "dir":  "/data/models/DeepSeek-R1-0528-Qwen3-8B",
        "size": "~16 GB",
        "gpus": 1,
        "note": "Best small distill. Runs on a single 40GB GPU.",
    },
    "qwen3-235b-thinking": {
        "repo": "Qwen/Qwen3-235B-A22B-Thinking-2507",
        "dir":  "/data/models/Qwen3-235B-A22B-Thinking-2507",
        "size": "~480 GB",
        "gpus": 8,
        "note": "State-of-the-art open math model. Needs 8x H100.",
    },
    "qwen3-32b": {
        "repo": "Qwen/Qwen3-32B",
        "dir":  "/data/models/Qwen3-32B",
        "size": "~65 GB",
        "gpus": 1,
        "note": "Best single-GPU math model after gpt-oss-120b.",
    },
    "qwen2.5-math-72b": {
        "repo": "Qwen/Qwen2.5-Math-72B-Instruct",
        "dir":  "/data/models/Qwen2.5-Math-72B-Instruct",
        "size": "~145 GB",
        "gpus": 2,
        "note": "Dedicated math specialist. Needs 2x H100.",
    },
    "phi4-reasoning": {
        "repo": "microsoft/Phi-4-reasoning",
        "dir":  "/data/models/Phi-4-reasoning",
        "size": "~28 GB",
        "gpus": 1,
        "note": "Efficient 14B math model. Fits comfortably on 1x H100.",
    },
    "openmath-nemotron-14b": {
        "repo": "nvidia/OpenMath-Nemotron-14B",
        "dir":  "/data/models/OpenMath-Nemotron-14B",
        "size": "~28 GB",
        "gpus": 1,
        "note": "AIMO2 Kaggle winning model base.",
    },
    "llama-nemotron-ultra": {
        "repo": "nvidia/Llama-Nemotron-Ultra-253B-v1",
        "dir":  "/data/models/Llama-Nemotron-Ultra-253B",
        "size": "~500 GB",
        "gpus": 8,
        "note": "NVIDIA's flagship reasoning model.",
    },
    "qwq-32b": {
        "repo": "Qwen/QwQ-32B",
        "dir":  "/data/models/QwQ-32B",
        "size": "~65 GB",
        "gpus": 1,
        "note": "Strong CoT reasoning. Single H100.",
    },
}

# ── models that fit on a single H100 80GB ──────────────────────────────────────
SINGLE_GPU_MODELS = [k for k, v in MODELS.items() if v["gpus"] == 1]


def check_hf_auth():
    """Verify HuggingFace login token is present."""
    token_path = os.path.expanduser("~/.cache/huggingface/token")
    if not os.path.exists(token_path):
        print("⚠️  Not logged in to HuggingFace.")
        print("   Run: huggingface-cli login")
        sys.exit(1)
    print("✅ HuggingFace token found.")


def check_disk_space(path, required_gb):
    """Warn if disk space looks tight."""
    result = subprocess.run(
        ["df", "-BG", path], capture_output=True, text=True
    )
    lines = result.stdout.strip().split("\n")
    if len(lines) >= 2:
        available = int(lines[1].split()[3].replace("G", ""))
        if available < required_gb * 1.2:
            print(f"⚠️  Low disk space: {available}GB available, "
                  f"need ~{required_gb}GB. Proceed anyway? (y/n)")
            if input().lower() != "y":
                sys.exit(1)


def download_model(key):
    """Download a single model using huggingface-cli."""
    if key not in MODELS:
        print(f"❌ Unknown model '{key}'. Available: {list(MODELS.keys())}")
        sys.exit(1)

    m = MODELS[key]
    print(f"\n{'='*60}")
    print(f"📥 Downloading: {key}")
    print(f"   Repo  : {m['repo']}")
    print(f"   Size  : {m['size']}")
    print(f"   GPUs  : {m['gpus']}x H100 required for inference")
    print(f"   Note  : {m['note']}")
    print(f"   Dest  : {m['dir']}")
    print(f"{'='*60}\n")

    os.makedirs(m["dir"], exist_ok=True)

    gb_est = int(m["size"].replace("~", "").replace(" GB", ""))
    check_disk_space("/data", gb_est)

    cmd = [
        "huggingface-cli", "download",
        m["repo"],
        "--local-dir", m["dir"],
        "--local-dir-use-symlinks", "False",
    ]

    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"\n✅ Successfully downloaded {key} to {m['dir']}")
    else:
        print(f"\n❌ Download failed for {key} (exit code {result.returncode})")
        sys.exit(1)


def list_models():
    print("\n📋 Available models:\n")
    print(f"{'Key':<30} {'Repo':<45} {'Size':<10} {'GPUs'}")
    print("-" * 95)
    for key, m in MODELS.items():
        print(f"{key:<30} {m['repo']:<45} {m['size']:<10} {m['gpus']}x H100")
    print(f"\n  'all_single_gpu' downloads: {SINGLE_GPU_MODELS}")


def main():
    parser = argparse.ArgumentParser(description="Download math LLMs for Nebius H100")
    parser.add_argument("--model", type=str, required=True,
                        help="Model key, 'all_single_gpu', or 'list'")
    parser.add_argument("--skip-auth-check", action="store_true",
                        help="Skip HuggingFace auth check")
    args = parser.parse_args()

    if args.model == "list":
        list_models()
        return

    if not args.skip_auth_check:
        check_hf_auth()

    if args.model == "all_single_gpu":
        print(f"Downloading all single-GPU models: {SINGLE_GPU_MODELS}")
        for key in SINGLE_GPU_MODELS:
            download_model(key)
    else:
        download_model(args.model)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
merge_lora.py — Merge a LoRA adapter into the base model and save full weights.

Required before Phase 2 training so Phase 2 starts from Phase 1's learned weights.

Usage:
  python merge_lora.py \
      --base  ~/models/gpt-oss-120b \
      --adapter ~/lora_phase1 \
      --output ~/models/gpt-oss-120b-phase1-merged
"""

import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, Mxfp4Config
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base",    required=True, help="Base model path")
    parser.add_argument("--adapter", required=True, help="LoRA adapter directory")
    parser.add_argument("--output",  required=True, help="Output path for merged model")
    args = parser.parse_args()

    print(f"Base model : {args.base}")
    print(f"LoRA adapter: {args.adapter}")
    print(f"Output     : {args.output}")

    print("\nLoading base model with the GPT-OSS MXFP4 training config...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        attn_implementation="eager",
        torch_dtype="auto",
        quantization_config=Mxfp4Config(dequantize=True),
        use_cache=True,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)

    print("Loading LoRA adapter and merging it into the base model...")
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()

    print(f"Saving merged model to {args.output}...")
    Path(args.output).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size="10GB")
    tokenizer.save_pretrained(args.output)

    print(f"\nDone. Merged model saved to {args.output}")
    print("Now run Phase 2:")
    print(f"  CUDA_VISIBLE_DEVICES=0 python sft_train.py --phase 2 --base-model {args.output}")


if __name__ == "__main__":
    main()

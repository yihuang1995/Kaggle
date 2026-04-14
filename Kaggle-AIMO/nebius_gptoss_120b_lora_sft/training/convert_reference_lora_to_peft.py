#!/usr/bin/env python3
"""
Convert a custom reference-LoRA checkpoint into a PEFT LoRA adapter.

This is primarily for vLLM serving. The custom reference trainer stores LoRA
weights in `trainable_state.pt` with GPT-OSS reference-model names
(`block.<n>.attn.qkv`, `block.<n>.attn.out`, `block.<n>.mlp.gate`). PEFT/vLLM
expects Hugging Face GPT-OSS names (`self_attn.q_proj`, etc.).

The attention-only export is the safest vLLM-compatible option because GPT-OSS
router modules are not standard PEFT LoRA targets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

Q_DIM = 64 * 64
K_DIM = 8 * 64
V_DIM = 8 * 64


def build_peft_config(base_model: str, rank: int, alpha: float, include_router: bool) -> dict:
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if include_router:
        target_modules.append("router")

    return {
        "alpha_pattern": {},
        "auto_mapping": None,
        "base_model_name_or_path": str(Path(base_model).expanduser()),
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": rank,
        "rank_pattern": {},
        "revision": None,
        "target_modules": target_modules,
        "task_type": "CAUSAL_LM",
    }


def convert_state(src_state: dict[str, torch.Tensor], include_router: bool) -> dict[str, torch.Tensor]:
    peft_state: dict[str, torch.Tensor] = {}
    layer_ids = sorted({int(k.split(".")[1]) for k in src_state if k.startswith("block.")})

    for i in layer_ids:
        prefix = f"base_model.model.model.layers.{i}"

        qkv_a = src_state[f"block.{i}.attn.qkv.lora_a"]
        qkv_b = src_state[f"block.{i}.attn.qkv.lora_b"]

        peft_state[f"{prefix}.self_attn.q_proj.lora_A.weight"] = qkv_a.clone()
        peft_state[f"{prefix}.self_attn.q_proj.lora_B.weight"] = qkv_b[:Q_DIM, :].clone()

        peft_state[f"{prefix}.self_attn.k_proj.lora_A.weight"] = qkv_a.clone()
        peft_state[f"{prefix}.self_attn.k_proj.lora_B.weight"] = qkv_b[Q_DIM:Q_DIM + K_DIM, :].clone()

        peft_state[f"{prefix}.self_attn.v_proj.lora_A.weight"] = qkv_a.clone()
        peft_state[f"{prefix}.self_attn.v_proj.lora_B.weight"] = qkv_b[Q_DIM + K_DIM:Q_DIM + K_DIM + V_DIM, :].clone()

        peft_state[f"{prefix}.self_attn.o_proj.lora_A.weight"] = src_state[f"block.{i}.attn.out.lora_a"].clone()
        peft_state[f"{prefix}.self_attn.o_proj.lora_B.weight"] = src_state[f"block.{i}.attn.out.lora_b"].clone()

        if include_router:
            peft_state[f"{prefix}.mlp.router.lora_A.weight"] = src_state[f"block.{i}.mlp.gate.lora_a"].clone()
            peft_state[f"{prefix}.mlp.router.lora_B.weight"] = src_state[f"block.{i}.mlp.gate.lora_b"].clone()

    return peft_state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="Source custom checkpoint dir")
    ap.add_argument("--output", required=True, help="Output PEFT adapter dir")
    ap.add_argument("--base-model", default="~/models/gpt-oss-120b")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=32.0)
    ap.add_argument(
        "--include-router",
        action="store_true",
        help="Also export router LoRA weights. Attention-only is safer for vLLM.",
    )
    args = ap.parse_args()

    src = Path(args.adapter).expanduser()
    dst = Path(args.output).expanduser()
    dst.mkdir(parents=True, exist_ok=True)

    payload = torch.load(src / "trainable_state.pt", map_location="cpu", weights_only=True)
    src_state = payload["model"]
    peft_state = convert_state(src_state, include_router=args.include_router)

    torch.save(peft_state, dst / "adapter_model.bin")
    (dst / "adapter_config.json").write_text(
        json.dumps(
            build_peft_config(
                base_model=args.base_model,
                rank=args.lora_rank,
                alpha=args.lora_alpha,
                include_router=args.include_router,
            ),
            indent=2,
        )
    )

    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        src_file = src / name
        if src_file.exists():
            (dst / name).write_bytes(src_file.read_bytes())

    total = sum(v.numel() for v in peft_state.values())
    mode = "attention+router" if args.include_router else "attention-only"
    print(f"Saved {len(peft_state)} tensors ({total/1e6:.1f}M params) to {dst} [{mode}]")


if __name__ == "__main__":
    main()

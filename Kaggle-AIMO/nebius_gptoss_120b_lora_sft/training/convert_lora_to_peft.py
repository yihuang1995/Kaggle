#!/usr/bin/env python3
"""
convert_lora_to_peft.py — Convert sft_train_reference_lora checkpoint to PEFT format.

The custom LoRA targets (attn_qkv, attn_out, mlp_gate) are mapped to HuggingFace
parameter names so vLLM can serve them via --enable-lora.

QKV split: combined (5120, r) lora_b → q_proj (4096,r) + k_proj (512,r) + v_proj (512,r)

Usage:
  python convert_lora_to_peft.py \
      --adapter ~/sft_ref_lora_phase2 \
      --output  ~/peft_sft_phase2
"""
import argparse, json
from pathlib import Path
import torch

Q_DIM = 64 * 64   # num_attention_heads * head_dim = 4096
K_DIM = 8  * 64   # num_key_value_heads * head_dim = 512
V_DIM = 8  * 64   # 512

def main(args):
    src  = Path(args.adapter)
    dst  = Path(args.output)
    dst.mkdir(parents=True, exist_ok=True)

    payload = torch.load(src / "trainable_state.pt", map_location="cpu", weights_only=True)
    state   = payload["model"]
    lora_rank   = args.lora_rank
    lora_alpha  = args.lora_alpha

    peft_state = {}

    # Detect number of layers
    layer_ids = sorted({int(k.split(".")[1]) for k in state if k.startswith("block.")})
    print(f"Converting {len(layer_ids)} layers, rank={lora_rank}, alpha={lora_alpha}")

    for i in layer_ids:
        prefix = f"base_model.model.model.layers.{i}"

        # ── attn qkv → split into q / k / v ──────────────────────────────────
        a = state[f"block.{i}.attn.qkv.lora_a"]   # (r, 2880)
        b = state[f"block.{i}.attn.qkv.lora_b"]   # (5120, r)

        peft_state[f"{prefix}.self_attn.q_proj.lora_A.weight"] = a.clone()
        peft_state[f"{prefix}.self_attn.q_proj.lora_B.weight"] = b[:Q_DIM, :].clone()

        peft_state[f"{prefix}.self_attn.k_proj.lora_A.weight"] = a.clone()
        peft_state[f"{prefix}.self_attn.k_proj.lora_B.weight"] = b[Q_DIM:Q_DIM+K_DIM, :].clone()

        peft_state[f"{prefix}.self_attn.v_proj.lora_A.weight"] = a.clone()
        peft_state[f"{prefix}.self_attn.v_proj.lora_B.weight"] = b[Q_DIM+K_DIM:, :].clone()

        # ── attn out → o_proj ─────────────────────────────────────────────────
        peft_state[f"{prefix}.self_attn.o_proj.lora_A.weight"] = \
            state[f"block.{i}.attn.out.lora_a"].clone()   # (r, 4096)
        peft_state[f"{prefix}.self_attn.o_proj.lora_B.weight"] = \
            state[f"block.{i}.attn.out.lora_b"].clone()   # (2880, r)

        # ── mlp gate → router ─────────────────────────────────────────────────
        peft_state[f"{prefix}.mlp.router.lora_A.weight"] = \
            state[f"block.{i}.mlp.gate.lora_a"].clone()   # (r, 2880)
        peft_state[f"{prefix}.mlp.router.lora_B.weight"] = \
            state[f"block.{i}.mlp.gate.lora_b"].clone()   # (128, r)

    # Save weights
    torch.save(peft_state, dst / "adapter_model.bin")

    # Save adapter config
    cfg = {
        "alpha_pattern":              {},
        "auto_mapping":               None,
        "base_model_name_or_path":    str(Path(args.base_model).expanduser()),
        "bias":                       "none",
        "fan_in_fan_out":             False,
        "inference_mode":             True,
        "init_lora_weights":          True,
        "lora_alpha":                 lora_alpha,
        "lora_dropout":               0.0,
        "modules_to_save":            None,
        "peft_type":                  "LORA",
        "r":                          lora_rank,
        "rank_pattern":               {},
        "revision":                   None,
        "target_modules":             ["q_proj", "k_proj", "v_proj", "o_proj", "router"],
        "task_type":                  "CAUSAL_LM",
    }
    (dst / "adapter_config.json").write_text(json.dumps(cfg, indent=2))

    # Copy tokenizer from adapter dir if present
    for fname in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]:
        if (src / fname).exists():
            (dst / fname).write_bytes((src / fname).read_bytes())

    total = sum(v.numel() for v in peft_state.values())
    print(f"Saved {len(peft_state)} tensors ({total/1e6:.1f}M params) to {dst}")
    print(f"adapter_config.json written.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter",    required=True)
    ap.add_argument("--output",     required=True)
    ap.add_argument("--base-model", default="~/models/gpt-oss-120b")
    ap.add_argument("--lora-rank",  type=int,   default=16)
    ap.add_argument("--lora-alpha", type=float, default=32.0)
    main(ap.parse_args())

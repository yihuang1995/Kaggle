#!/usr/bin/env python3
"""
LoRA-style SFT for GPT-OSS 120B on the official `gpt_oss.torch` backend.

This keeps the stable 8-GPU reference-model loading path, but freezes the base
weights and trains low-rank adapters on top of the attention and router
projections.

Launch:
  torchrun --standalone --nproc_per_node=8 sft_train_reference_lora.py --phase 1
  torchrun --standalone --nproc_per_node=8 sft_train_reference_lora.py --phase 1 --smoke-test
  torchrun --standalone --nproc_per_node=8 sft_train_reference_lora.py --phase 2 \
      --resume-from ~/sft_ref_lora_phase1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer


BASE_MODEL_PATH = os.path.expanduser("~/models/gpt-oss-120b")
DATA_DIR = Path(os.path.expanduser("~/data"))
PHASE1_DATA = DATA_DIR / "train_part1.jsonl"
PHASE2_DATA = DATA_DIR / "train_part2.jsonl"
PHASE1_OUT = os.path.expanduser("~/sft_ref_lora_phase1")
PHASE2_OUT = os.path.expanduser("~/sft_ref_lora_phase2")

PHASE_CONFIG = {
    1: {
        "data_path": PHASE1_DATA,
        "output_dir": PHASE1_OUT,
        "num_epochs": 2,
        "learning_rate": 1e-4,
    },
    2: {
        "data_path": PHASE2_DATA,
        "output_dir": PHASE2_OUT,
        "num_epochs": 1,
        "learning_rate": 5e-5,
    },
}

DEFAULT_ADAPTER_TARGETS = ("attn_qkv", "attn_out", "mlp_gate")
MAX_LENGTH = 1536
SMOKE_MAX_LENGTH = 512
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 5
SAVE_STEPS = 100
SEED = 42


def rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def is_rank0() -> bool:
    return rank() == 0


def log(msg: str) -> None:
    if is_rank0():
        print(msg, flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed() -> torch.device:
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    rk = int(os.environ.get("RANK", "0"))
    if ws > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=ws,
            rank=rk,
        )

    local_rank = int(os.environ.get("LOCAL_RANK", rk))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if dist.is_initialized():
        warmup = torch.ones(1, device=device)
        dist.all_reduce(warmup)
        torch.cuda.synchronize(device)

    return device


def import_reference_model():
    repo_root = Path(os.path.expanduser("~/gpt-oss"))
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from gpt_oss.torch.model import ModelConfig, Transformer  # pylint: disable=import-error
    from gpt_oss.torch.weights import Checkpoint  # pylint: disable=import-error

    return ModelConfig, Transformer, Checkpoint


def load_reference_model(base_model: str, device: torch.device):
    ModelConfig, Transformer, Checkpoint = import_reference_model()

    config_path = Path(base_model) / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        raw_config = json.load(f)

    rope_scaling = raw_config.get("rope_scaling", {})
    config = ModelConfig(
        num_hidden_layers=raw_config["num_hidden_layers"],
        num_experts=raw_config.get("num_local_experts", raw_config.get("num_experts", 128)),
        experts_per_token=raw_config.get(
            "experts_per_token",
            raw_config.get("num_experts_per_tok", 4),
        ),
        vocab_size=raw_config["vocab_size"],
        hidden_size=raw_config["hidden_size"],
        intermediate_size=raw_config["intermediate_size"],
        swiglu_limit=raw_config.get("swiglu_limit", 7.0),
        head_dim=raw_config["head_dim"],
        num_attention_heads=raw_config["num_attention_heads"],
        num_key_value_heads=raw_config["num_key_value_heads"],
        sliding_window=raw_config.get("sliding_window", 128),
        initial_context_length=raw_config.get(
            "initial_context_length",
            rope_scaling.get("original_max_position_embeddings", 4096),
        ),
        rope_theta=float(raw_config.get("rope_theta", 150000.0)),
        rope_scaling_factor=float(rope_scaling.get("factor", 32.0)),
        rope_ntk_alpha=float(rope_scaling.get("beta_slow", 1.0)),
        rope_ntk_beta=float(rope_scaling.get("beta_fast", 32.0)),
    )

    model = Transformer(config=config, device=device)
    model.train()

    my_rank = rank()
    per_rank_intermediate_size = config.intermediate_size // world_size()
    checkpoint_reader = Checkpoint(base_model, device)

    for name, param in model.named_parameters():
        loaded_tensor = load_hf_checkpoint_tensor(checkpoint_reader, name)
        if "mlp1" in name:
            loaded_tensor = loaded_tensor[
                :,
                my_rank * 2 * per_rank_intermediate_size : (my_rank + 1)
                * 2
                * per_rank_intermediate_size,
                ...,
            ]
        elif "mlp2_weight" in name:
            loaded_tensor = loaded_tensor[
                ...,
                my_rank * per_rank_intermediate_size : (my_rank + 1)
                * per_rank_intermediate_size,
            ]
        param.data.copy_(loaded_tensor)

    return model


def load_hf_checkpoint_tensor(checkpoint_reader, param_name: str) -> torch.Tensor:
    if param_name == "embedding.weight":
        return checkpoint_reader._get_tensor("model.embed_tokens.weight")
    if param_name == "norm.scale":
        return checkpoint_reader._get_tensor("model.norm.weight")
    if param_name == "unembedding.weight":
        return checkpoint_reader._get_tensor("lm_head.weight")

    match = re.match(r"block\.(\d+)\.(.+)", param_name)
    if not match:
        raise KeyError(f"Unrecognized parameter name: {param_name}")

    layer_idx = int(match.group(1))
    suffix = match.group(2)
    prefix = f"model.layers.{layer_idx}"

    if suffix == "attn.sinks":
        return checkpoint_reader._get_tensor(f"{prefix}.self_attn.sinks")
    if suffix == "attn.norm.scale":
        return checkpoint_reader._get_tensor(f"{prefix}.input_layernorm.weight")
    if suffix == "attn.qkv.weight":
        q = checkpoint_reader._get_tensor(f"{prefix}.self_attn.q_proj.weight")
        k = checkpoint_reader._get_tensor(f"{prefix}.self_attn.k_proj.weight")
        v = checkpoint_reader._get_tensor(f"{prefix}.self_attn.v_proj.weight")
        return torch.cat([q, k, v], dim=0)
    if suffix == "attn.qkv.bias":
        q = checkpoint_reader._get_tensor(f"{prefix}.self_attn.q_proj.bias")
        k = checkpoint_reader._get_tensor(f"{prefix}.self_attn.k_proj.bias")
        v = checkpoint_reader._get_tensor(f"{prefix}.self_attn.v_proj.bias")
        return torch.cat([q, k, v], dim=0)
    if suffix == "attn.out.weight":
        return checkpoint_reader._get_tensor(f"{prefix}.self_attn.o_proj.weight")
    if suffix == "attn.out.bias":
        return checkpoint_reader._get_tensor(f"{prefix}.self_attn.o_proj.bias")
    if suffix == "mlp.norm.scale":
        return checkpoint_reader._get_tensor(f"{prefix}.post_attention_layernorm.weight")
    if suffix == "mlp.gate.weight":
        return checkpoint_reader._get_tensor(f"{prefix}.mlp.router.weight")
    if suffix == "mlp.gate.bias":
        return checkpoint_reader._get_tensor(f"{prefix}.mlp.router.bias")
    if suffix == "mlp.mlp1_weight":
        return checkpoint_reader._get_mxfp4_tensor(
            f"{prefix}.mlp.experts.gate_up_proj_blocks",
            f"{prefix}.mlp.experts.gate_up_proj_scales",
            dtype=torch.bfloat16,
        )
    if suffix == "mlp.mlp1_bias":
        return checkpoint_reader._get_tensor(f"{prefix}.mlp.experts.gate_up_proj_bias")
    if suffix == "mlp.mlp2_weight":
        return checkpoint_reader._get_mxfp4_tensor(
            f"{prefix}.mlp.experts.down_proj_blocks",
            f"{prefix}.mlp.experts.down_proj_scales",
            dtype=torch.bfloat16,
        )
    if suffix == "mlp.mlp2_bias":
        return checkpoint_reader._get_tensor(f"{prefix}.mlp.experts.down_proj_bias")

    raise KeyError(f"No HF checkpoint mapping for {param_name}")


def load_and_format(data_path: Path, tokenizer, smoke_test: bool, max_length: int):
    dataset = load_dataset("json", data_files=str(data_path), split="train")
    if smoke_test:
        dataset = dataset.select(range(min(16, len(dataset))))

    log(f"Loaded {len(dataset):,} samples from {data_path}")

    def encode_example(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        return {
            "input_ids": tokenized["input_ids"],
            "seq_len": len(tokenized["input_ids"]),
        }

    dataset = dataset.map(encode_example, remove_columns=dataset.column_names, num_proc=4)
    dataset = dataset.filter(lambda example: example["seq_len"] >= 2, num_proc=4)
    return dataset


def iter_batches(dataset, batch_size: int, seed: int):
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        yield [dataset[idx]["input_ids"] for idx in batch_indices]


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / max(rank, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        device = base.weight.device
        dtype = base.weight.dtype
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=device, dtype=dtype))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=device, dtype=dtype))

        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_x = self.dropout(x).to(self.lora_a.dtype)
        update = F.linear(F.linear(lora_x, self.lora_a), self.lora_b)
        return base_out + update * self.scale


def attach_lora_adapter(parent: nn.Module, attr_name: str, rank: int, alpha: float, dropout: float):
    module = getattr(parent, attr_name)
    if isinstance(module, LoRALinear):
        return module
    wrapped = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
    setattr(parent, attr_name, wrapped)
    return wrapped


def configure_trainable_params(model, args):
    for _, param in model.named_parameters():
        param.requires_grad = False

    adapter_targets = tuple(
        part.strip() for part in args.adapter_targets.split(",") if part.strip()
    )
    attached = []

    for layer_idx, block in enumerate(model.block):
        if "attn_qkv" in adapter_targets:
            attach_lora_adapter(block.attn, "qkv", args.lora_rank, args.lora_alpha, args.lora_dropout)
            attached.append(f"block.{layer_idx}.attn.qkv")
        if "attn_out" in adapter_targets:
            attach_lora_adapter(block.attn, "out", args.lora_rank, args.lora_alpha, args.lora_dropout)
            attached.append(f"block.{layer_idx}.attn.out")
        if "mlp_gate" in adapter_targets:
            attach_lora_adapter(block.mlp, "gate", args.lora_rank, args.lora_alpha, args.lora_dropout)
            attached.append(f"block.{layer_idx}.mlp.gate")

    if args.train_norms:
        for name, param in model.named_parameters():
            if name.endswith(".norm.scale") or name == "norm.scale":
                param.requires_grad = True

    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    frozen = sum(param.numel() for _, param in model.named_parameters() if not param.requires_grad)
    trainable_count = sum(param.numel() for _, param in trainable)

    log(f"Adapter targets  : {', '.join(adapter_targets)}")
    log(f"LoRA rank/alpha : {args.lora_rank}/{args.lora_alpha}")
    log(f"LoRA dropout    : {args.lora_dropout}")
    log(f"Train norms     : {args.train_norms}")
    log(f"Attached adapters: {len(attached)}")
    log(f"Trainable params: {trainable_count:,}")
    log(f"Frozen params   : {frozen:,}")
    if is_rank0():
        preview = "\n".join(f"  - {name}" for name, _ in trainable[:16])
        print(f"Trainable tensors (preview):\n{preview}", flush=True)
    return [param for _, param in trainable]


def named_trainable_state(model):
    return {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def load_trainable_state(model, checkpoint_dir: str | None, device: torch.device) -> int:
    if not checkpoint_dir:
        return 0

    state_path = Path(checkpoint_dir) / "trainable_state.pt"
    meta_path = Path(checkpoint_dir) / "trainer_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing checkpoint state: {state_path}")

    payload = torch.load(state_path, map_location=device)
    model_state = payload["model"]
    missing = []

    name_to_param = dict(model.named_parameters())
    for name, value in model_state.items():
        if name not in name_to_param:
            missing.append(name)
            continue
        name_to_param[name].data.copy_(value.to(device=device, dtype=name_to_param[name].dtype))

    if missing:
        raise KeyError(f"Checkpoint contains unknown tensors: {missing[:5]}")

    start_step = 0
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        start_step = int(meta.get("global_step", 0))

    log(f"Loaded trainable checkpoint from {checkpoint_dir} at step {start_step}")
    return start_step


def save_checkpoint(model, tokenizer, output_dir: str, global_step: int, phase: int):
    if not is_rank0():
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "phase": phase,
            "global_step": global_step,
            "model": named_trainable_state(model),
        },
        output_path / "trainable_state.pt",
    )
    tokenizer.save_pretrained(output_path)
    with (output_path / "trainer_state.json").open("w", encoding="utf-8") as f:
        json.dump({"phase": phase, "global_step": global_step}, f, indent=2)

    log(f"Saved checkpoint to {output_dir} at step {global_step}")


def forward_logits(model, input_ids: torch.Tensor, use_checkpointing: bool) -> torch.Tensor:
    hidden = model.embedding(input_ids)
    for block in model.block:
        if use_checkpointing and any(param.requires_grad for param in block.parameters()):
            hidden = checkpoint(block, hidden, use_reentrant=False)
        else:
            hidden = block(hidden)
    hidden = model.norm(hidden)
    return model.unembedding(hidden)


def compute_loss(model, input_ids: list[int], device: torch.device, use_checkpointing: bool) -> torch.Tensor:
    tokens = torch.tensor(input_ids, dtype=torch.int32, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = forward_logits(model, tokens, use_checkpointing)
    logits = logits[:-1].float()
    labels = tokens[1:].to(torch.long)
    return F.cross_entropy(logits, labels)


def cosine_lr(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return float(step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train(args):
    device = init_distributed()
    set_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True

    phase_cfg = PHASE_CONFIG[args.phase]
    output_dir = args.output_dir or phase_cfg["output_dir"]
    max_length = args.max_length or (SMOKE_MAX_LENGTH if args.smoke_test else MAX_LENGTH)
    grad_accum = args.grad_accum
    epochs = 1 if args.smoke_test else phase_cfg["num_epochs"]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dataset = load_and_format(phase_cfg["data_path"], tokenizer, args.smoke_test, max_length)
    steps_per_epoch = math.ceil(len(dataset) / grad_accum)
    total_steps = max(steps_per_epoch * epochs, 1)
    warmup_steps = max(int(total_steps * WARMUP_RATIO), 1)

    log(f"Loading reference GPT-OSS checkpoint from {args.base_model} on {world_size()} GPUs...")
    model = load_reference_model(args.base_model, device)

    trainable_params = configure_trainable_params(model, args)
    if not trainable_params:
        raise RuntimeError("No trainable parameters selected.")

    start_step = load_trainable_state(model, args.resume_from, device)

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate or phase_cfg["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY,
        fused=True,
    )

    global_step = start_step
    optimizer.zero_grad(set_to_none=True)

    log("")
    log("=" * 72)
    log(f"Phase {args.phase} reference LoRA SFT")
    log(f"  Base model   : {args.base_model}")
    log(f"  Data         : {phase_cfg['data_path']}")
    log(f"  Output       : {output_dir}")
    log(f"  Epochs       : {epochs}")
    log(f"  Max length   : {max_length}")
    log(f"  Grad accum   : {grad_accum}")
    log(f"  Warmup steps : {warmup_steps}")
    log("=" * 72)
    log("")

    running_loss = 0.0
    micro_step = 0

    for epoch in range(epochs):
        epoch_seed = SEED + epoch
        for batch_idx, batch in enumerate(iter_batches(dataset, batch_size=1, seed=epoch_seed), start=1):
            loss = compute_loss(model, batch[0], device, use_checkpointing=not args.disable_checkpointing)
            loss = loss / grad_accum
            loss.backward()
            running_loss += loss.item() * grad_accum
            micro_step += 1

            if micro_step % grad_accum != 0:
                continue

            lr_scale = cosine_lr(global_step, total_steps, warmup_steps)
            current_lr = (args.learning_rate or phase_cfg["learning_rate"]) * lr_scale
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % LOGGING_STEPS == 0:
                avg_loss = running_loss / LOGGING_STEPS
                log(
                    f"step={global_step:05d} epoch={epoch + 1}/{epochs} "
                    f"loss={avg_loss:.4f} lr={current_lr:.3e} batch={batch_idx}"
                )
                running_loss = 0.0

            if global_step % SAVE_STEPS == 0 and not args.smoke_test:
                save_checkpoint(model, tokenizer, output_dir, global_step, args.phase)

    if micro_step % grad_accum != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

    save_checkpoint(model, tokenizer, output_dir, global_step, args.phase)
    log(f"Training complete at global step {global_step}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=[1, 2], required=True)
    parser.add_argument("--base-model", default=BASE_MODEL_PATH)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--disable-checkpointing", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--adapter-targets",
        default=",".join(DEFAULT_ADAPTER_TARGETS),
        help="Comma-separated subset of: attn_qkv,attn_out,mlp_gate",
    )
    parser.add_argument("--train-norms", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

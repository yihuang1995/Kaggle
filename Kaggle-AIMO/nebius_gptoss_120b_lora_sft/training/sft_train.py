#!/usr/bin/env python3
"""
sft_train.py — 2-phase LoRA fine-tuning of GPT-OSS 120B on AIMO math data.

This version follows the GPT-OSS MXFP4 fine-tuning path instead of trying to
reload the full model into BF16 under ZeRO/FSDP. The checkpoint already fits on
one H100 80GB, so we fine-tune it directly on a single visible GPU.

Launch:
  CUDA_VISIBLE_DEVICES=0 python sft_train.py --phase 1
  CUDA_VISIBLE_DEVICES=0 python sft_train.py --phase 2 \
      --base-model ~/models/gpt-oss-120b-phase1-merged

Smoke test:
  CUDA_VISIBLE_DEVICES=0 python sft_train.py --phase 1 --smoke-test
"""

import argparse
import os
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
from trl import SFTConfig, SFTTrainer


BASE_MODEL_PATH = os.path.expanduser("~/models/gpt-oss-120b")
DATA_DIR = Path(os.path.expanduser("~/data"))
PHASE1_DATA = DATA_DIR / "train_part1.jsonl"
PHASE2_DATA = DATA_DIR / "train_part2.jsonl"
PHASE1_OUT = os.path.expanduser("~/lora_phase1")
PHASE2_OUT = os.path.expanduser("~/lora_phase2")

PHASE_CONFIG = {
    1: dict(data_path=PHASE1_DATA, output_dir=PHASE1_OUT, num_epochs=2, learning_rate=1e-4),
    2: dict(data_path=PHASE2_DATA, output_dir=PHASE2_OUT, num_epochs=1, learning_rate=5e-5),
}

MAX_LENGTH = 2048
BATCH_SIZE = 1
GRAD_ACCUM = 8
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "cosine"
LOGGING_STEPS = 10
SAVE_STEPS = 200
SAVE_TOTAL = 3

LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    target_parameters=[
        "7.mlp.experts.gate_up_proj",
        "7.mlp.experts.down_proj",
        "15.mlp.experts.gate_up_proj",
        "15.mlp.experts.down_proj",
        "23.mlp.experts.gate_up_proj",
        "23.mlp.experts.down_proj",
        "31.mlp.experts.gate_up_proj",
        "31.mlp.experts.down_proj",
    ],
    bias="none",
    inference_mode=False,
)


def load_and_format(data_path: Path, tokenizer, smoke_test: bool = False) -> Dataset:
    print(f"Loading data from {data_path}...")
    ds = load_dataset("json", data_files=str(data_path), split="train")
    if smoke_test:
        ds = ds.select(range(min(100, len(ds))))
    print(f"  Loaded {len(ds):,} samples")

    def apply_template(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    return ds.map(apply_template, remove_columns=ds.column_names, num_proc=4)


def load_model_and_tokenizer(model_path: str):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading GPT-OSS with MXFP4 dequantization for training...")
    quantization_config = Mxfp4Config(dequantize=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="eager",
        dtype=torch.bfloat16,
        quantization_config=quantization_config,
        use_cache=False,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = False
    model.enable_input_require_grads()
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()
    return model, tokenizer


def train(phase: int, base_model_path: str, smoke_test: bool):
    cfg = PHASE_CONFIG[phase]

    print(f"\n{'=' * 60}")
    print(f"PHASE {phase}  (single-GPU MXFP4 LoRA)")
    print(f"  Base    : {base_model_path}")
    print(f"  Data    : {cfg['data_path']}")
    print(f"  Output  : {cfg['output_dir']}")
    print(f"  Epochs  : {cfg['num_epochs']}")
    print(f"  LR      : {cfg['learning_rate']}")
    print(f"  Batch   : {BATCH_SIZE} sample/GPU × {GRAD_ACCUM} accum = {BATCH_SIZE * GRAD_ACCUM} effective")
    print(f"  Max len : {MAX_LENGTH}")
    print(f"{'=' * 60}\n")

    model, tokenizer = load_model_and_tokenizer(base_model_path)
    dataset = load_and_format(cfg["data_path"], tokenizer, smoke_test)
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    epochs = 1 if smoke_test else cfg["num_epochs"]

    training_args = SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        bf16=True,
        fp16=False,
        tf32=True,
        optim="adamw_torch_fused",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=MAX_LENGTH,
        dataset_text_field="text",
        packing=False,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS if not smoke_test else 9999,
        save_total_limit=SAVE_TOTAL,
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    print(f"\n=== Saving Phase {phase} LoRA adapter ===")
    trainer.model.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])

    logs = [entry for entry in trainer.state.log_history if "loss" in entry]
    if logs:
        print(f"Final train loss: {logs[-1]['loss']:.4f}")
    print(f"Saved to {cfg['output_dir']}")
    print(f"Phase {phase} complete!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=[1, 2], required=True)
    parser.add_argument("--base-model", default=BASE_MODEL_PATH)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    train(args.phase, args.base_model, args.smoke_test)


if __name__ == "__main__":
    main()

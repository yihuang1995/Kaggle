#!/usr/bin/env python3
"""
sft_oss20b.py — SFT fine-tuning of GPT-OSS 20B on AIMO math data.

Usage:
  python sft_oss20b.py

Output:
  ~/lora_adapter/   — LoRA weights
  ~/sft_train.log   — training log (also piped to stdout)
"""

import os, re, torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.expanduser("~/model")
TRAIN_FILE  = os.path.expanduser("~/data/aimo_cleaned_data_v1/train.txt")
OUTPUT_DIR  = os.path.expanduser("~/lora_adapter")
SEPARATOR   = "=" * 50

# Hyperparameters
LORA_R              = 16
LORA_ALPHA          = 32
LORA_DROPOUT        = 0.05
TARGET_MODULES      = ["q_proj", "k_proj", "v_proj", "o_proj"]
MAX_SEQ_LENGTH      = 2048
NUM_EPOCHS          = 3
BATCH_SIZE          = 2
GRAD_ACCUM          = 8
LEARNING_RATE       = 2e-4
LR_SCHEDULER        = "cosine"
WARMUP_RATIO        = 0.05
WEIGHT_DECAY        = 0.01
LOGGING_STEPS       = 5
SAVE_STEPS          = 50

# ── 1. Parse training data ────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert mathematician. Solve the problem step by step, "
    "showing all reasoning. End with your final answer in \\boxed{}."
)

def parse_train(train_file):
    with open(train_file) as f:
        content = f.read()
    entries = [e.strip() for e in content.split(SEPARATOR) if e.strip()]

    samples = []
    for entry in entries:
        q_m  = re.search(r"<question>(.*?)</question>",       entry, re.DOTALL)
        r_m  = re.search(r"<reasoning>(.*?)</reasoning>",     entry, re.DOTALL)
        fa_m = re.search(r"<final_answer>(.*?)</final_answer>", entry, re.DOTALL)
        if not q_m:
            continue
        question  = q_m.group(1).strip()
        reasoning = r_m.group(1).strip()  if r_m  else ""
        answer    = fa_m.group(1).strip() if fa_m else ""

        # Build assistant response: reasoning + answer
        assistant = reasoning
        if answer:
            assistant = assistant + "\n\n" + answer if assistant else answer

        samples.append({
            "question":  question,
            "assistant": assistant,
        })

    print(f"Parsed {len(samples)} training samples")
    return samples


def format_sample(sample, tokenizer):
    """Apply chat template to produce a single training string."""
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": sample["question"]},
        {"role": "assistant", "content": sample["assistant"]},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


# ── 2. Load model & tokenizer ─────────────────────────────────────────────────

def load_model_and_tokenizer():
    print(f"Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = MAX_SEQ_LENGTH

    print(f"Loading model from {MODEL_PATH}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False   # required for gradient checkpointing

    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.0f} GB)")
    print(f"Model loaded — params: {sum(p.numel() for p in model.parameters())/1e9:.1f}B")
    return model, tokenizer


# ── 3. Apply LoRA ─────────────────────────────────────────────────────────────

def apply_lora(model):
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


# ── 4. Build dataset ──────────────────────────────────────────────────────────

def build_dataset(samples, tokenizer):
    texts = [format_sample(s, tokenizer) for s in samples]
    return Dataset.from_dict({"text": texts})


# ── 5. Train ──────────────────────────────────────────────────────────────────

def train(model, tokenizer, dataset):
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        bf16=True,
        fp16=False,
        optim="adamw_torch",
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        dataset_text_field="text",
        gradient_checkpointing=True,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("\n=== Starting training ===")
    print(f"  Samples       : {len(dataset)}")
    print(f"  Epochs        : {NUM_EPOCHS}")
    print(f"  Batch size    : {BATCH_SIZE} × {GRAD_ACCUM} accum = {BATCH_SIZE*GRAD_ACCUM} effective")
    print(f"  LR            : {LEARNING_RATE}  scheduler={LR_SCHEDULER}")
    print(f"  Max seq len   : {MAX_SEQ_LENGTH}")
    print(f"  Output        : {OUTPUT_DIR}")
    print("=" * 40)

    trainer.train()

    print("\n=== Saving LoRA adapter ===")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved to {OUTPUT_DIR}")

    # Print final loss
    logs = trainer.state.log_history
    loss_entries = [l for l in logs if "loss" in l]
    if loss_entries:
        print(f"Final train loss: {loss_entries[-1]['loss']:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    samples          = parse_train(TRAIN_FILE)
    model, tokenizer = load_model_and_tokenizer()
    model            = apply_lora(model)
    dataset          = build_dataset(samples, tokenizer)
    train(model, tokenizer, dataset)
    print("\nDone!")


if __name__ == "__main__":
    main()

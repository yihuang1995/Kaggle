#!/usr/bin/env python3
"""
eval_native.py — Pre/post SFT evaluation using native gpt_oss.torch backend.

Reuses model loading, LoRA, and distributed utilities from sft_train_reference_lora.py.

Launch (run baseline and SFT separately or together):
  torchrun --standalone --nproc_per_node=8 eval_native.py -- --tag baseline
  torchrun --standalone --nproc_per_node=8 eval_native.py -- --tag sft_phase2 \
      --adapter ~/sft_ref_lora_phase2

Compare:
  python eval_native.py -- --compare
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

# ── reuse training infrastructure ─────────────────────────────────────────────
_TRAIN_DIR = Path(__file__).parent
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))

from sft_train_reference_lora import (
    attach_lora_adapter,
    init_distributed,
    is_rank0,
    load_reference_model,
    load_trainable_state,
    log,
    rank,
    world_size,
)

# ── config ─────────────────────────────────────────────────────────────────────
BASE_MODEL  = os.path.expanduser("~/models/gpt-oss-120b")
EVAL_DATA   = Path(os.path.expanduser("~/data/eval_1000.jsonl"))
HARD_DATA   = Path(os.path.expanduser("~/data/hard_problems.jsonl"))
RESULTS_DIR = Path(os.path.expanduser("~/eval_results"))

K           = 4      # candidates per problem
TEMPERATURE = 0.8
MAX_TOKENS  = 768
DEFAULT_N   = 200
SEED        = 42

SYSTEM_PROMPT = (
    "You are an elite mathematical problem solver. "
    "The final answer must be a non-negative integer between 0 and 99999. "
    r"Place your final numerical answer inside \boxed{}, e.g., \boxed{42}. "
    r"Output exactly ONE \boxed{} at the very end of your solution."
)

# ── LoRA restore ───────────────────────────────────────────────────────────────

ADAPTER_TARGETS = ("attn_qkv", "attn_out", "mlp_gate")
TARGET_MAP = {
    "attn_qkv": ("attn", "qkv"),
    "attn_out":  ("attn", "out"),
    "mlp_gate":  ("mlp",  "gate"),
}

def restore_lora(model, adapter_dir: str, device: torch.device,
                 lora_rank: int = 16, lora_alpha: float = 32.0):
    """Attach LoRA wrappers then load saved weights (no grad needed for eval)."""
    for block in model.block:
        for target in ADAPTER_TARGETS:
            parent_attr, child_attr = TARGET_MAP[target]
            parent = getattr(block, parent_attr)
            attach_lora_adapter(parent, child_attr,
                                rank=lora_rank, alpha=lora_alpha, dropout=0.0)

    step = load_trainable_state(model, adapter_dir, device)
    log(f"Restored LoRA from {adapter_dir} (step {step})")


# ── data ───────────────────────────────────────────────────────────────────────

def load_problems(n: int, path: Path = EVAL_DATA) -> list[dict]:
    problems = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            msgs = rec["messages"]
            q    = next(m["content"] for m in msgs if m["role"] == "user")
            asst = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            problems.append({"question": q, "gold": extract_int(asst)})
    if n > 0:
        problems = random.Random(SEED).sample(problems, min(n, len(problems)))
    log(f"Loaded {len(problems):,} problems (gold: {sum(1 for p in problems if p['gold'] is not None):,})")
    return problems


def extract_int(text: str):
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    if not m: return None
    try:
        v = int(m[-1].replace(",", "").strip())
        return v if 0 <= v <= 99999 else None
    except ValueError:
        return None


# ── generation ─────────────────────────────────────────────────────────────────

@torch.inference_mode()
def generate_one(model, prompt_ids: list[int], stop_ids: set,
                 temperature: float, max_tokens: int) -> list[int]:
    tokens = list(prompt_ids)
    for _ in range(max_tokens):
        t = torch.tensor(tokens, dtype=torch.int32, device=next(model.parameters()).device)
        hidden = model.embedding(t)
        for block in model.block:
            hidden = block(hidden)
        hidden = model.norm(hidden)
        logits = model.unembedding(hidden)[-1].float()

        if temperature == 0.0:
            next_tok = int(torch.argmax(logits))
        else:
            probs   = torch.softmax(logits / temperature, dim=-1)
            next_tok = int(torch.multinomial(probs, 1))

        tokens.append(next_tok)
        if next_tok in stop_ids:
            break
    return tokens[len(prompt_ids):]


def get_stop_ids(tokenizer) -> set:
    ids = {tokenizer.eos_token_id}
    if hasattr(tokenizer, "additional_special_tokens_ids"):
        ids.update(tokenizer.additional_special_tokens_ids)
    return {i for i in ids if i is not None}


def build_prompt(tokenizer, question: str) -> list[int]:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tokenizer.encode(text, add_special_tokens=False)


# ── metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    with_gold = [r for r in results if r["gold"] is not None]
    n = len(with_gold)
    if n == 0: return {"error": "no gold"}

    pass_at_k    = sum(1 for r in with_gold if r["pass_at_k"])    / n
    majority_acc = sum(1 for r in with_gold if r["majority_ok"])  / n
    format_rate  = sum(r["fmt"] for r in results) / len(results)
    exact_at_1   = sum(r["n_correct"] for r in with_gold) / (n * K)
    return {
        "n_problems":    len(results),
        "n_with_gold":   n,
        "exact_at_1":    round(exact_at_1,   4),
        "pass_at_4":     round(pass_at_k,    4),
        "majority_at_4": round(majority_acc, 4),
        "format_rate":   round(format_rate,  4),
    }


# ── compare ────────────────────────────────────────────────────────────────────

def compare_results():
    files = sorted(RESULTS_DIR.glob("*.json"))
    if not files:
        print("No results in", RESULTS_DIR); return

    data   = {json.loads(f.read_text())["tag"]: json.loads(f.read_text())["metrics"] for f in files}
    tags   = list(data.keys())
    keys   = ["exact_at_1", "pass_at_4", "majority_at_4", "format_rate",
               "hard_pass_at_4", "hard_majority_at_4"]

    print(f"\n{'='*65}")
    print("PRE vs POST SFT — EVALUATION COMPARISON")
    print(f"{'='*65}")
    print(f"{'Metric':<22}", end="")
    for t in tags: print(f"  {t:<18}", end="")
    print()
    print("-" * 65)
    for mk in keys:
        print(f"  {mk:<20}", end="")
        for t in tags:
            v = data[t].get(mk)
            print(f"  {v*100:>6.2f}%         " if isinstance(v, float) else f"  {'N/A':<18}", end="")
        print()


# ── eval loop ──────────────────────────────────────────────────────────────────

def eval_set(model, tokenizer, problems: list[dict], label: str) -> list[dict]:
    stop_ids = get_stop_ids(tokenizer)
    results  = []
    for idx, prob in enumerate(problems):
        # broadcast problem to all ranks
        obj = [prob]
        if dist.is_initialized():
            dist.broadcast_object_list(obj, src=0)
        prob = obj[0]

        if is_rank0() and idx % 20 == 0:
            log(f"  [{idx+1}/{len(problems)}] {label} ...")

        prompt = build_prompt(tokenizer, prob["question"])
        gold   = prob["gold"]
        answers, texts = [], []
        for _ in range(K):
            gen = generate_one(model, prompt, stop_ids, TEMPERATURE, MAX_TOKENS)
            text = tokenizer.decode(gen, skip_special_tokens=True)
            texts.append(text)
            answers.append(extract_int(text))

        if is_rank0():
            valid    = [a for a in answers if a is not None]
            majority = Counter(valid).most_common(1)[0][0] if valid else None
            ok       = [a == gold for a in answers if a is not None and gold is not None]
            results.append({
                "question":  prob["question"][:80],
                "gold":      gold,
                "answers":   answers,
                "majority":  majority,
                "pass_at_k": any(ok) if ok else None,
                "majority_ok": (majority == gold) if majority is not None and gold is not None else None,
                "fmt":       len(valid) / K,
                "n_correct": sum(ok),
            })
    return results


# ── main ───────────────────────────────────────────────────────────────────────

def main(args):
    if args.compare:
        compare_results(); return

    device    = init_distributed()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_reference_model(args.base_model, device)
    model.eval()

    if args.adapter:
        restore_lora(model, args.adapter, device)

    problems = load_problems(args.n) if is_rank0() else [None] * DEFAULT_N
    if dist.is_initialized():
        cnt = [len(problems)]
        dist.broadcast_object_list(cnt, src=0)
        if not is_rank0():
            problems = [None] * cnt[0]

    log(f"\n{'='*60}")
    log(f"Eval: {args.tag}  |  adapter: {args.adapter or 'none'}  |  K={K}")
    log(f"{'='*60}\n")

    results = eval_set(model, tokenizer, problems, args.tag)

    if is_rank0():
        metrics = compute_metrics(results)

        # Hard problems
        if HARD_DATA.exists():
            hard_probs = load_problems(0, HARD_DATA)
            hard_results = eval_set(model, tokenizer, hard_probs, f"{args.tag}/hard")
            log("\nHard problem details:")
            for i, r in enumerate(hard_results):
                log(f"  [{i+1}] gold={r['gold']}  answers={r['answers']}  majority={r['majority']}")
            hm = compute_metrics(hard_results)
            metrics["hard_pass_at_4"]     = hm.get("pass_at_4", 0.0)
            metrics["hard_majority_at_4"] = hm.get("majority_at_4", 0.0)

        log(f"\n{'='*50}\nResults: {args.tag}\n{'='*50}")
        for k, v in metrics.items():
            log(f"  {k:<22}: {v*100:.2f}%" if isinstance(v, float) else f"  {k:<22}: {v}")

        RESULTS_DIR.mkdir(exist_ok=True)
        out = RESULTS_DIR / f"{args.tag}.json"
        out.write_text(json.dumps({"tag": args.tag, "adapter": args.adapter,
                                   "metrics": metrics, "results": results}, indent=2))
        log(f"\nSaved → {out}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag",        default=None)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--adapter",    default=None)
    parser.add_argument("--n",          type=int, default=DEFAULT_N)
    parser.add_argument("--compare",    action="store_true")
    args = parser.parse_args()
    if not args.compare and not args.tag:
        parser.error("--tag required unless --compare")
    main(args)

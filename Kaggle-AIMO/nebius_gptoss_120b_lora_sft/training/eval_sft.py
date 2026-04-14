#!/usr/bin/env python3
"""
eval_sft.py — Evaluate math model accuracy using vLLM.

Metrics:
  - exact@1   : best-of-k exact match (pick highest-scoring candidate)
  - pass@k    : any candidate correct (k=4, k=8)
  - majority@k: majority vote answer correct
  - format_rate: % responses containing valid \\boxed{integer}

Usage:
  # Start vLLM server first (separate terminal):
  python -m vllm.entrypoints.openai.api_server \
      --model ~/models/gpt-oss-120b \
      --dtype auto --kv-cache-dtype fp8_e4m3 \
      --max-model-len 8192 --gpu-memory-utilization 0.92 \
      --port 8000

  # Run eval (500 problems by default for speed):
  python eval_sft.py --model ~/models/gpt-oss-120b --tag baseline
  python eval_sft.py --model ~/models/gpt-oss-120b-phase1-merged --tag phase1
  python eval_sft.py --model ~/models/gpt-oss-120b-sft --tag phase2

  # Full eval (31k problems):
  python eval_sft.py --model ~/models/gpt-oss-120b --tag baseline --n 0

  # Compare all saved results:
  python eval_sft.py --compare
"""

import re
import json
import time
import argparse
import asyncio
import random
from pathlib import Path
from collections import Counter
from openai import AsyncOpenAI

# ── Config ─────────────────────────────────────────────────────────────────────
EVAL_DATA    = Path("~/data/eval_1000.jsonl").expanduser()
HARD_DATA    = Path("~/data/hard_problems.jsonl").expanduser()
RESULTS_DIR  = Path("~/eval_results").expanduser()
BASE_URL     = "http://localhost:8000/v1"
K            = 4           # candidates per problem (~30 min for 1000 problems on H200)
TEMPERATURE  = 1.0
MIN_P        = 0.02
MAX_TOKENS   = 4096
CONCURRENCY  = 32          # parallel requests to vLLM
DEFAULT_N    = 0           # 0 = all 1000 problems
SEED         = 42

SYSTEM_PROMPT = (
    "You are an elite mathematical problem solver competing at the International "
    "Mathematical Olympiad (IMO) level. Your goal is to find the correct answer "
    "through rigorous mathematical reasoning.\n\n"
    "The final answer must be a non-negative integer between 0 and 99999.\n"
    "Place your final numerical answer inside \\boxed{}, e.g., \\boxed{42}\n"
    "Output exactly ONE \\boxed{} at the very end of your solution."
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_eval_data(n: int, data_path: Path = None) -> list[dict]:
    """Load eval JSONL, extract question + gold answer."""
    data_path = data_path or EVAL_DATA
    problems = []
    with open(data_path) as f:
        for line in f:
            rec = json.loads(line)
            msgs = rec["messages"]
            question = next(m["content"] for m in msgs if m["role"] == "user")
            # Extract gold integer from assistant content \boxed{N}
            assistant = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            gold = extract_int(assistant)
            problems.append({"question": question, "gold": gold})

    if n > 0:
        random.seed(SEED)
        problems = random.sample(problems, min(n, len(problems)))

    print(f"Eval problems: {len(problems):,} (gold_available: {sum(1 for p in problems if p['gold'] is not None):,})")
    return problems


def extract_int(text: str) -> int | None:
    """Extract last \\boxed{integer} from text."""
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    if not matches:
        return None
    inner = matches[-1].replace(",", "").strip()
    try:
        val = int(inner)
        return val if 0 <= val <= 99999 else None
    except ValueError:
        return None


# ── Inference ──────────────────────────────────────────────────────────────────

async def generate_candidates(client: AsyncOpenAI, model: str, question: str) -> list[str]:
    """Generate K candidates for one problem."""
    sem = asyncio.Semaphore(1)  # semaphore managed at caller level
    tasks = []
    for _ in range(K):
        tasks.append(client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": question},
            ],
            temperature=TEMPERATURE,
            extra_body={"min_p": MIN_P},
            max_tokens=MAX_TOKENS,
        ))
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    texts = []
    for r in responses:
        if isinstance(r, Exception):
            texts.append("")
        else:
            texts.append(r.choices[0].message.content or "")
    return texts


async def eval_one(client: AsyncOpenAI, model: str, sem: asyncio.Semaphore,
                   problem: dict, idx: int, total: int) -> dict:
    async with sem:
        if idx % 50 == 0:
            print(f"  [{idx}/{total}] evaluating...", flush=True)
        candidates = await generate_candidates(client, model, problem["question"])

    gold = problem["gold"]
    answers = [extract_int(c) for c in candidates]
    valid = [a for a in answers if a is not None]

    # majority vote
    if valid:
        majority, _ = Counter(valid).most_common(1)[0]
    else:
        majority = None

    exact_matches = [a == gold for a in answers if a is not None and gold is not None]
    pass_at_k = any(exact_matches) if exact_matches else None
    majority_correct = (majority == gold) if (majority is not None and gold is not None) else None

    return {
        "question":         problem["question"][:100],
        "gold":             gold,
        "answers":          answers,
        "majority":         majority,
        "pass_at_k":        pass_at_k,
        "majority_correct": majority_correct,
        "format_rate":      len(valid) / K,
        "n_correct":        sum(exact_matches),
    }


async def run_eval(model_path: str, problems: list[dict]) -> list[dict]:
    client = AsyncOpenAI(base_url=BASE_URL, api_key="EMPTY")
    sem = asyncio.Semaphore(CONCURRENCY)

    tasks = [
        eval_one(client, model_path, sem, p, i, len(problems))
        for i, p in enumerate(problems)
    ]
    results = await asyncio.gather(*tasks)
    return list(results)


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    with_gold = [r for r in results if r["gold"] is not None]
    n = len(with_gold)
    if n == 0:
        return {"error": "no gold answers available"}

    pass_at_k      = sum(1 for r in with_gold if r["pass_at_k"]) / n
    majority_acc   = sum(1 for r in with_gold if r["majority_correct"]) / n
    format_rate    = sum(r["format_rate"] for r in results) / len(results)

    # exact@1: best single candidate (highest n_correct / K as proxy)
    # Approximated here as majority vote accuracy since we don't rerank
    exact_at_1 = majority_acc

    # pass@4 (use first 4 candidates)
    pass_at_4 = sum(
        1 for r in with_gold
        if any(a == r["gold"] for a in r["answers"][:4] if a is not None)
    ) / n

    return {
        "n_problems":   len(results),
        "n_with_gold":  n,
        "exact_at_1":   round(exact_at_1,  4),
        "pass_at_4":    round(pass_at_4,   4),
        "pass_at_8":    round(pass_at_k,   4),
        "majority_at_8":round(majority_acc,4),
        "format_rate":  round(format_rate, 4),
    }


def print_metrics(tag: str, metrics: dict):
    print(f"\n{'='*50}")
    print(f"Results: {tag}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<20}: {v*100:.2f}%")
        else:
            print(f"  {k:<20}: {v}")


# ── Compare mode ───────────────────────────────────────────────────────────────

def compare_results():
    RESULTS_DIR.mkdir(exist_ok=True)
    files = sorted(RESULTS_DIR.glob("*.json"))
    if not files:
        print("No result files found in", RESULTS_DIR)
        return

    all_metrics = {}
    for f in files:
        data = json.loads(f.read_text())
        tag = data["tag"]
        all_metrics[tag] = data["metrics"]

    metrics_keys = ["exact_at_1", "pass_at_4", "majority_at_8", "format_rate",
                    "hard_pass_at_4", "hard_majority"]
    tags = list(all_metrics.keys())

    print(f"\n{'='*70}")
    print("BEFORE vs AFTER SFT COMPARISON")
    print(f"{'='*70}")
    print(f"{'Metric':<22}", end="")
    for t in tags:
        print(f"  {t:<18}", end="")
    print()
    print("-" * 70)
    for mk in metrics_keys:
        print(f"  {mk:<20}", end="")
        for t in tags:
            v = all_metrics[t].get(mk, "N/A")
            if isinstance(v, float):
                print(f"  {v*100:>7.2f}%       ", end="")
            else:
                print(f"  {'N/A':<18}", end="")
        print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default=None, help="Model path (must match what vLLM is serving)")
    parser.add_argument("--tag",     default=None, help="Label for this run (e.g. baseline, phase1, phase2)")
    parser.add_argument("--n",       type=int, default=DEFAULT_N,
                        help="Number of problems to evaluate (0 = all 31k)")
    parser.add_argument("--url",     default=None, help="vLLM server URL")
    parser.add_argument("--compare", action="store_true", help="Print comparison table and exit")
    args = parser.parse_args()

    global BASE_URL
    if args.url:
        BASE_URL = args.url

    if args.compare:
        compare_results()
        return

    if not args.model or not args.tag:
        parser.error("--model and --tag are required (unless using --compare)")

    print(f"Evaluating: {args.tag}")
    print(f"Model     : {args.model}")
    print(f"Server    : {BASE_URL}")
    print(f"k         : {K} candidates per problem")
    print(f"Problems  : {args.n if args.n > 0 else 'all'}")

    problems = load_eval_data(args.n)
    t0 = time.time()
    results = asyncio.run(run_eval(args.model, problems))
    elapsed = time.time() - t0

    metrics = compute_metrics(results)

    # ── Evaluate hard problems separately ──────────────────────────────────────
    hard_metrics = {}
    if HARD_DATA.exists():
        print(f"\n=== Evaluating {len(list(open(HARD_DATA)))} hard+integer problems ===")
        hard_problems = load_eval_data(0, data_path=HARD_DATA)
        hard_results  = asyncio.run(run_eval(args.model, hard_problems))
        hard_metrics  = compute_metrics(hard_results)

        print(f"\n{'='*50}")
        print("Hard Problems (3 olympiad-level problems):")
        for i, r in enumerate(hard_results):
            print(f"  Problem {i+1}: gold={r['gold']}  answers={r['answers']}  "
                  f"pass={r['pass_at_k']}  majority={r['majority']}")

        metrics["hard_pass_at_4"]  = hard_metrics.get("pass_at_4", 0)
        metrics["hard_majority"]   = hard_metrics.get("majority_at_8", 0)

    print_metrics(args.tag, metrics)
    print(f"\nTotal time: {elapsed/60:.1f} min")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{args.tag}.json"
    out.write_text(json.dumps({
        "tag":          args.tag,
        "model":        args.model,
        "metrics":      metrics,
        "hard_metrics": hard_metrics,
        "results":      results,
    }, indent=2))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
eval_val_sft.py — Evaluate fine-tuned GPT-OSS 20B (+ LoRA) on the val set,
                  then compare with vanilla OSS 20B results.

Usage:
  ANTHROPIC_API_KEY=sk-ant-... python eval_val_sft.py
"""

import os, re, json, argparse, textwrap
from collections import Counter
from pathlib import Path

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
BASE_MODEL_PATH   = os.path.expanduser("~/model")
ADAPTER_PATH      = os.path.expanduser("~/lora_adapter")
VAL_FILE          = os.path.expanduser("~/data/aimo_cleaned_data_v1/val.txt")
VANILLA_FILE      = os.path.expanduser("~/eval_results_claude_judge.json")
OUTPUT_FILE       = os.path.expanduser("~/eval_results_sft_claude_judge.json")
SEPARATOR         = "=" * 50
EVAL_MODEL        = "claude-haiku-4-5-20251001"
K                 = 4
TEMPERATURE       = 0.7
MAX_TOKENS        = 2048

# ── 1. Parse val.txt ──────────────────────────────────────────────────────────

def _extract_int_from_boxed(text):
    if not text:
        return None
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if not m:
        return None
    inner = m.group(1).replace(",", "").strip()
    return inner if re.fullmatch(r"-?\d+", inner) else None

def parse_val(val_file):
    with open(val_file) as f:
        content = f.read()
    entries = [e.strip() for e in content.split(SEPARATOR) if e.strip()]
    questions = []
    for entry in entries:
        qid_m = re.search(r"<question_id>(.*?)</question_id>", entry, re.DOTALL)
        q_m   = re.search(r"<question>(.*?)</question>",       entry, re.DOTALL)
        fa_m  = re.search(r"<final_answer>(.*?)</final_answer>",entry, re.DOTALL)
        if not qid_m or not q_m:
            continue
        questions.append({
            "question_id": qid_m.group(1).strip(),
            "problem":     q_m.group(1).strip(),
            "gold_answer": _extract_int_from_boxed(fa_m.group(1).strip() if fa_m else ""),
        })
    return questions

# ── 2. vLLM inference with LoRA adapter ──────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert mathematician. Solve the problem step by step, "
    "showing all reasoning. End with your final answer in \\boxed{}."
)

def run_inference(questions, base_path, adapter_path, k, max_tokens, temperature):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)

    print(f"Loading base model + LoRA adapter...")
    llm = LLM(
        model=base_path,
        enable_lora=True,
        max_lora_rank=16,
        dtype="bfloat16",
        max_model_len=4096,
        trust_remote_code=True,
        gpu_memory_utilization=0.92,
    )

    from vllm.lora.request import LoRARequest
    lora_request = LoRARequest("sft_adapter", 1, adapter_path)

    sampling = SamplingParams(n=k, temperature=temperature, max_tokens=max_tokens)

    prompts = []
    for q in questions:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": q["problem"]},
        ]
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ))

    print(f"Generating {len(prompts)} × k={k} with SFT adapter...")
    outputs = llm.generate(prompts, sampling, lora_request=lora_request)

    return [
        [{"candidate_id": f"c{j+1}", "text": o.text} for j, o in enumerate(out.outputs)]
        for out in outputs
    ]

# ── 3. Answer extraction ──────────────────────────────────────────────────────

def extract_answer(text):
    boxed_all = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed_all:
        inner = boxed_all[-1].replace(",", "").strip()
        if re.fullmatch(r"-?\d+", inner):
            return inner
    tail = text[-500:]
    for pat in [
        r"(?:answer|result|value)\s+(?:is|=|equals)\s*[:\s]*(-?\d{1,9})\b",
        r"(?:therefore|thus|hence|so)[^.]*?=\s*(-?\d{1,9})\b",
        r"=\s*\**(-?\d{1,9})\**\s*[.$\n]",
    ]:
        m = re.search(pat, tail, re.IGNORECASE)
        if m:
            return m.group(1)
    for line in reversed([l.strip() for l in text.splitlines() if l.strip()]):
        if re.fullmatch(r"-?\d{1,9}", line):
            return line
    return "NONE"

# ── 4. Claude judge ───────────────────────────────────────────────────────────

EVAL_SYSTEM = textwrap.dedent("""\
    You are a strict evaluator for olympiad-style math problems with integer final answers.

    Evaluation rules:
    1. Extract the model's final integer answer from each candidate output.
    2. If gold_answer is provided, compare it with the extracted answer.
    3. If gold_answer is not provided, set exact_match to null.
    4. Judge whether the reasoning is mathematically valid.
    5. If the final answer is correct but reasoning is weak, mark it "suspicious".
    6. If no integer final answer can be extracted, set extracted_answer to "NONE".
    7. Focus on mathematical correctness, not writing style.
    8. Be strict and deterministic. Do not use outside knowledge.
    9. Evaluate each candidate independently.

    For each candidate return:
      candidate_id, extracted_answer (integer string or "NONE"),
      exact_match (0/1 or null), format_valid (0/1),
      reasoning_verdict ("correct"|"incorrect"|"suspicious"),
      error_type ("none"|"arithmetic"|"algebra"|"logic"|"missing_case"|
                  "unsupported_final_answer"|"format"|"other"),
      score (0-10), short_reason (one sentence).

    Return ONLY valid JSON. No markdown fences. Schema:
    {"candidates": [{...}, ...]}""")

def claude_eval(question_id, problem, gold, candidates, api_key):
    import anthropic
    lines = [f"question_id: {question_id}", f"problem: {problem}"]
    if gold:
        lines.append(f"gold_answer: {gold}")
    lines.append("")
    for c in candidates:
        lines.append(f"--- {c['candidate_id']} ---")
        lines.append(c["text"].strip())
        lines.append("")

    client = anthropic.Anthropic(api_key=api_key)
    for attempt in range(3):
        msg = client.messages.create(
            model=EVAL_MODEL, max_tokens=4096,
            system=EVAL_SYSTEM,
            messages=[{"role": "user", "content": "\n".join(lines)}],
        )
        raw = msg.content[0].text.strip()
        raw = re.sub(r"^```json\s*|^```\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        try:
            start, end = raw.index("{"), raw.rindex("}") + 1
            return json.loads(raw[start:end])["candidates"]
        except Exception as e:
            print(f"  [warn] parse failed attempt {attempt+1}: {e}")
    return heuristic_eval(candidates, gold)

def heuristic_eval(candidates, gold):
    evals = []
    for c in candidates:
        ext = extract_answer(c["text"])
        fmt = 0 if ext == "NONE" else 1
        exact = (1 if ext == gold else 0) if gold else None
        verdict = "correct" if exact == 1 else "incorrect"
        score   = 9 if exact == 1 else (4 if fmt else 1)
        evals.append({"candidate_id": c["candidate_id"], "extracted_answer": ext,
                      "exact_match": exact, "format_valid": fmt,
                      "reasoning_verdict": verdict, "error_type": "other",
                      "score": score, "short_reason": "Heuristic fallback."})
    return evals

# ── 5. Aggregate + summary ────────────────────────────────────────────────────

VERDICT_RANK = {"correct": 0, "suspicious": 1, "incorrect": 2}

def compute_aggregate(gold, evals):
    has_gold  = gold is not None
    n         = len(evals)
    extracted = [c["extracted_answer"] for c in evals]
    exact_ok  = [c for c in evals if c.get("exact_match") == 1]
    non_none  = [a for a in extracted if a != "NONE"]
    if non_none:
        from collections import Counter as C
        maj_ans, maj_cnt = C(non_none).most_common(1)[0]
    else:
        maj_ans, maj_cnt = "NONE", extracted.count("NONE")
    best   = sorted(evals, key=lambda c: (VERDICT_RANK.get(c["reasoning_verdict"], 2), -c.get("score", 0)))[0]
    scores = [c.get("score", 0) for c in evals]
    cs     = [c.get("score", 0) for c in exact_ok]
    ne     = len(exact_ok)
    return {
        "best_candidate_id":          best["candidate_id"],
        "best_extracted_answer":      best["extracted_answer"],
        "best_of_k_exact_match":      best.get("exact_match") if has_gold else None,
        "pass_at_k":                  (1 if ne > 0 else 0) if has_gold else None,
        "num_candidates":             n,
        "num_exact_match":            ne if has_gold else None,
        "exact_match_rate_at_k":      round(ne/n, 4) if has_gold else None,
        "majority_answer":            maj_ans,
        "majority_answer_count":      maj_cnt,
        "majority_answer_fraction":   round(maj_cnt/n, 4),
        "majority_vote_exact_match":  (1 if maj_ans == gold else 0) if has_gold else None,
        "num_unique_extracted_answers": len(set(extracted)),
        "unique_answer_list":         list(set(extracted)),
        "answer_histogram":           dict(Counter(extracted)),
        "num_format_valid":           sum(c["format_valid"] for c in evals),
        "format_valid_rate_at_k":     round(sum(c["format_valid"] for c in evals)/n, 4),
        "num_none_answers":           extracted.count("NONE"),
        "num_reasoning_correct":      sum(1 for c in evals if c["reasoning_verdict"] == "correct"),
        "num_reasoning_suspicious":   sum(1 for c in evals if c["reasoning_verdict"] == "suspicious"),
        "num_reasoning_incorrect":    sum(1 for c in evals if c["reasoning_verdict"] == "incorrect"),
        "reasoning_correct_rate_at_k": round(sum(1 for c in evals if c["reasoning_verdict"] == "correct")/n, 4),
        "avg_score":                  round(sum(scores)/n, 2),
        "max_score":                  max(scores),
        "correct_answer_support_count": ne if has_gold else None,
        "correct_answer_best_score":    max(cs) if cs else None,
        "correct_answer_avg_score":     round(sum(cs)/len(cs), 2) if cs else None,
    }

def compute_summary(results):
    s = {"num_questions": len(results),
         "num_questions_with_gold_answer": sum(1 for r in results if r["problem_has_gold_answer"]),
         "num_total_candidates": sum(len(r["candidates"]) for r in results),
         "num_exact_match": 0, "num_format_valid": 0,
         "num_reasoning_correct": 0, "num_reasoning_suspicious": 0,
         "num_pass_at_k": 0, "num_best_of_k_exact_match": 0,
         "num_majority_vote_exact_match": 0}
    for r in results:
        agg = r["aggregate"]
        s["num_format_valid"]         += agg["num_format_valid"]
        s["num_reasoning_correct"]    += agg["num_reasoning_correct"]
        s["num_reasoning_suspicious"] += agg["num_reasoning_suspicious"]
        if r["problem_has_gold_answer"]:
            s["num_exact_match"]               += agg["num_exact_match"] or 0
            s["num_pass_at_k"]                 += agg["pass_at_k"] or 0
            s["num_best_of_k_exact_match"]     += agg["best_of_k_exact_match"] or 0
            s["num_majority_vote_exact_match"] += agg["majority_vote_exact_match"] or 0
    return s

# ── 6. Comparison report ──────────────────────────────────────────────────────

def print_comparison(vanilla_file, sft_file):
    with open(vanilla_file) as f: van = json.load(f)
    with open(sft_file)    as f: sft = json.load(f)
    v_by = {r["question_id"]: r for r in van["results"]}
    s_by = {r["question_id"]: r for r in sft["results"]}

    sep = "=" * 100
    print(f"\n{sep}")
    print("COMPARISON: Vanilla OSS-20B  vs  SFT OSS-20B  (both judged by Claude Haiku)")
    print(sep)
    print("{:15s} {:5s} | {:>6s} {:>5s} {:>7s} {:>9s} {:>6s} | {:>6s} {:>5s} {:>7s} {:>9s} {:>6s}".format(
        "question_id","gold",
        "best","fmt%","avg_sc","reasoning","pass@k",
        "best","fmt%","avg_sc","reasoning","pass@k"))
    print("{:15s} {:5s} | {:^43s} | {:^43s}".format(
        "","","--- Vanilla OSS-20B ---","--- SFT OSS-20B ---"))
    print("-" * 100)

    for qid in v_by:
        if qid not in s_by:
            continue
        v = v_by[qid]["aggregate"]
        s = s_by[qid]["aggregate"]
        gold = "yes" if v_by[qid]["problem_has_gold_answer"] else "no"
        vr = f"{v['num_reasoning_correct']}c/{v['num_reasoning_suspicious']}s"
        sr = f"{s['num_reasoning_correct']}c/{s['num_reasoning_suspicious']}s"
        vp = str(v["pass_at_k"]) if v["pass_at_k"] is not None else "-"
        sp = str(s["pass_at_k"]) if s["pass_at_k"] is not None else "-"
        print("{:15s} {:5s} | {:>6s} {:>5.2f} {:>7.2f} {:>9s} {:>6s} | {:>6s} {:>5.2f} {:>7.2f} {:>9s} {:>6s}".format(
            qid, gold,
            str(v["best_extracted_answer"]), v["format_valid_rate_at_k"], v["avg_score"], vr, vp,
            str(s["best_extracted_answer"]), s["format_valid_rate_at_k"], s["avg_score"], sr, sp,
        ))

    print(sep)
    vs, ss = van["summary"], sft["summary"]
    v_avg = sum(r["aggregate"]["avg_score"] for r in van["results"]) / len(van["results"])
    s_avg = sum(r["aggregate"]["avg_score"] for r in sft["results"]) / len(sft["results"])
    v_tot, s_tot = vs["num_total_candidates"], ss["num_total_candidates"]

    print("\nSUMMARY")
    print("-" * 65)
    print(f"{'Metric':42s} {'Vanilla':>10s} {'SFT':>10s} {'Delta':>8s}")
    print("-" * 65)

    def row(label, v, s, pct=False):
        if pct:
            d = f"{(s-v)*100:+.1f}pp"
            print(f"{label:42s} {v:>10.1%} {s:>10.1%} {d:>8s}")
        else:
            d = f"{s-v:+d}" if isinstance(v, int) else f"{s-v:+.2f}"
            print(f"{label:42s} {str(v):>10s} {str(s):>10s} {d:>8s}")

    row("num_format_valid",           vs["num_format_valid"],        ss["num_format_valid"])
    row("format_valid_rate",          vs["num_format_valid"]/v_tot,  ss["num_format_valid"]/s_tot,  pct=True)
    row("num_reasoning_correct",      vs["num_reasoning_correct"],   ss["num_reasoning_correct"])
    row("reasoning_correct_rate",     vs["num_reasoning_correct"]/v_tot, ss["num_reasoning_correct"]/s_tot, pct=True)
    row("num_reasoning_suspicious",   vs["num_reasoning_suspicious"],ss["num_reasoning_suspicious"])
    row("num_exact_match (gold qs)",  vs["num_exact_match"],         ss["num_exact_match"])
    row("num_pass_at_k",              vs["num_pass_at_k"],           ss["num_pass_at_k"])
    row("avg_score_across_questions", v_avg,                         s_avg)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    api_key = ANTHROPIC_API_KEY
    if not api_key:
        print("WARNING: no ANTHROPIC_API_KEY — using heuristic eval")

    questions        = parse_val(VAL_FILE)
    all_candidates   = run_inference(questions, BASE_MODEL_PATH, ADAPTER_PATH,
                                     K, MAX_TOKENS, TEMPERATURE)
    results = []
    for i, (q, candidates) in enumerate(zip(questions, all_candidates)):
        print(f"[{i+1}/{len(questions)}] Evaluating {q['question_id']}  gold={q['gold_answer']}")
        evals     = claude_eval(q["question_id"], q["problem"], q["gold_answer"],
                                candidates, api_key) if api_key else heuristic_eval(candidates, q["gold_answer"])
        aggregate = compute_aggregate(q["gold_answer"], evals)
        print(f"         best={aggregate['best_extracted_answer']}  "
              f"pass@k={aggregate['pass_at_k']}  avg_score={aggregate['avg_score']}")
        results.append({"question_id": q["question_id"],
                        "problem_has_gold_answer": q["gold_answer"] is not None,
                        "candidates": evals, "aggregate": aggregate})

    summary = compute_summary(results)
    with open(OUTPUT_FILE, "w") as f:
        json.dump({"results": results, "summary": summary}, f, indent=2)
    print(f"\nSaved to {OUTPUT_FILE}")
    print(json.dumps(summary, indent=2))

    if Path(VANILLA_FILE).exists():
        print_comparison(VANILLA_FILE, OUTPUT_FILE)
    else:
        print(f"Vanilla results not found at {VANILLA_FILE} — skipping comparison")

if __name__ == "__main__":
    main()

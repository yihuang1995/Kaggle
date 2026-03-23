#!/usr/bin/env python3
"""
eval_claude_gen.py — Claude as both generator and judge on the val set.

Pipeline:
  1. Parse val.txt
  2. Claude Sonnet generates k=4 candidate answers per question
  3. Claude Haiku judges each candidate (same prompt as oss20b eval)
  4. Compute aggregate metrics
  5. Save eval_results_claude_gen_claude_judge.json
  6. Print side-by-side comparison with eval_results_claude_judge.json (oss20b)
"""

import os, re, json, time
from collections import Counter
from pathlib import Path
import anthropic

API_KEY     = os.environ.get("ANTHROPIC_API_KEY", "")
GEN_MODEL   = "claude-sonnet-4-6"
EVAL_MODEL  = "claude-haiku-4-5-20251001"
K           = 4
TEMPERATURE = 1.0   # high temp for diversity across candidates

VAL_FILE    = os.path.expanduser("~/Downloads/val.txt")
OSS_FILE    = os.path.expanduser("~/Downloads/eval_results_claude_judge.json")
OUT_FILE    = os.path.expanduser("~/Downloads/eval_results_claude_gen_claude_judge.json")
SEPARATOR   = "=" * 50

client = anthropic.Anthropic(api_key=API_KEY)

# ── Parse val.txt ─────────────────────────────────────────────────────────────

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
        raw_fa   = fa_m.group(1).strip() if fa_m else ""
        gold_int = _extract_int_from_boxed(raw_fa)
        questions.append({
            "question_id": qid_m.group(1).strip(),
            "problem":     q_m.group(1).strip(),
            "gold_answer": gold_int,
        })
    return questions

# ── Claude generation ─────────────────────────────────────────────────────────

GEN_SYSTEM = (
    "You are an expert mathematician solving olympiad competition problems. "
    "Think step by step. At the end of your solution write your final integer "
    "answer inside \\boxed{}, for example: \\boxed{42}."
)

def generate_one(problem, candidate_id, retries=5):
    for attempt in range(retries):
        try:
            msg = client.messages.create(
                model=GEN_MODEL,
                max_tokens=2048,
                temperature=TEMPERATURE,
                system=GEN_SYSTEM,
                messages=[{"role": "user", "content": f"Problem:\n{problem}"}],
            )
            return {"candidate_id": candidate_id, "text": msg.content[0].text}
        except anthropic.APIStatusError as e:
            wait = 2 ** attempt * 5
            print(f" [retry in {wait}s: {e.status_code}]", end=" ", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"Generation failed after {retries} retries")

def generate_candidates(problem, k):
    candidates = []
    for i in range(k):
        print(f"    generating c{i+1}/{k}…", end=" ", flush=True)
        cand = generate_one(problem, f"c{i+1}")
        candidates.append(cand)
        print("done")
        time.sleep(0.3)   # gentle rate limit
    return candidates

# ── Answer extraction ─────────────────────────────────────────────────────────

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

# ── Claude judge (same as eval_val.py) ───────────────────────────────────────

EVAL_SYSTEM = """\
You are a strict evaluator for olympiad-style math problems with integer final answers.

Evaluation rules:
1. Extract the model's final integer answer from each candidate output.
2. If gold_answer is provided, compare it with the extracted answer.
3. If gold_answer is not provided, set exact_match to null.
4. Judge whether the reasoning is mathematically valid.
5. If the final answer is correct but the reasoning is weak or unsupported, mark it "suspicious".
6. If no integer final answer can be extracted, set extracted_answer to "NONE".
7. Focus on mathematical correctness, not writing style.
8. Be strict and deterministic. Do not use outside knowledge.
9. Evaluate each candidate independently.
10. Select the best candidate (mathematical correctness > reasoning validity > score).

For each candidate return:
  candidate_id, extracted_answer (integer string or "NONE"),
  exact_match (0/1 or null), format_valid (0/1),
  reasoning_verdict ("correct"|"incorrect"|"suspicious"),
  error_type ("none"|"arithmetic"|"algebra"|"logic"|"missing_case"|
              "unsupported_final_answer"|"format"|"other"),
  score (0-10), short_reason (one sentence).

Return ONLY valid JSON. No markdown fences. Schema:
{"candidates": [{...}, ...]}"""

def claude_eval(question_id, problem, gold, candidates, retries=3):
    lines = [f"question_id: {question_id}", f"problem: {problem}"]
    if gold:
        lines.append(f"gold_answer: {gold}")
    lines.append("")
    for c in candidates:
        lines.append(f"--- {c['candidate_id']} ---")
        lines.append(c["text"].strip())
        lines.append("")

    for attempt in range(retries):
        msg = client.messages.create(
            model=EVAL_MODEL,
            max_tokens=4096,
            system=EVAL_SYSTEM,
            messages=[{"role": "user", "content": "\n".join(lines)}],
        )
        raw = msg.content[0].text.strip()
        # Strip markdown fences
        raw = re.sub(r"^```json\s*|^```\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        # Extract outermost JSON object robustly
        try:
            start = raw.index("{")
            end   = raw.rindex("}") + 1
            return json.loads(raw[start:end])["candidates"]
        except (ValueError, json.JSONDecodeError, KeyError) as e:
            print(f"  [warn] eval parse failed (attempt {attempt+1}): {e}")
            time.sleep(1)
    # Final fallback: heuristic scoring
    print("  [warn] all eval attempts failed, using heuristic fallback")
    return heuristic_fallback(candidates, gold)

# ── Aggregate metrics (identical to eval_val.py) ──────────────────────────────

def heuristic_fallback(candidates, gold):
    evals = []
    for c in candidates:
        extracted = extract_answer(c["text"])
        fmt_valid = 0 if extracted == "NONE" else 1
        exact = None
        if gold is not None:
            exact = 1 if extracted == gold else 0
        verdict = "correct" if exact == 1 else ("incorrect" if fmt_valid else "incorrect")
        score   = 9 if exact == 1 else (5 if fmt_valid else 1)
        evals.append({
            "candidate_id": c["candidate_id"], "extracted_answer": extracted,
            "exact_match": exact, "format_valid": fmt_valid,
            "reasoning_verdict": verdict, "error_type": "other",
            "score": score, "short_reason": "Fallback heuristic scoring.",
        })
    return evals

VERDICT_RANK = {"correct": 0, "suspicious": 1, "incorrect": 2}

def compute_aggregate(gold, evals):
    has_gold = gold is not None
    n        = len(evals)
    extracted = [c["extracted_answer"] for c in evals]
    exact_ok  = [c for c in evals if c.get("exact_match") == 1]

    non_none = [a for a in extracted if a != "NONE"]
    if non_none:
        ctr = Counter(non_none)
        maj_ans, maj_cnt = ctr.most_common(1)[0]
    else:
        maj_ans, maj_cnt = "NONE", extracted.count("NONE")

    best = sorted(evals, key=lambda c: (VERDICT_RANK.get(c["reasoning_verdict"], 2), -c["score"]))[0]
    scores        = [c["score"] for c in evals]
    correct_scores = [c["score"] for c in exact_ok]
    n_exact        = len(exact_ok)

    return {
        "best_candidate_id":          best["candidate_id"],
        "best_extracted_answer":      best["extracted_answer"],
        "best_of_k_exact_match":      best.get("exact_match") if has_gold else None,
        "pass_at_k":                  (1 if n_exact > 0 else 0) if has_gold else None,
        "num_candidates":             n,
        "num_exact_match":            n_exact            if has_gold else None,
        "exact_match_rate_at_k":      round(n_exact / n, 4) if has_gold else None,
        "majority_answer":            maj_ans,
        "majority_answer_count":      maj_cnt,
        "majority_answer_fraction":   round(maj_cnt / n, 4),
        "majority_vote_exact_match":  (1 if maj_ans == gold else 0) if has_gold else None,
        "num_unique_extracted_answers": len(set(extracted)),
        "unique_answer_list":         list(set(extracted)),
        "answer_histogram":           dict(Counter(extracted)),
        "num_format_valid":           sum(c["format_valid"] for c in evals),
        "format_valid_rate_at_k":     round(sum(c["format_valid"] for c in evals) / n, 4),
        "num_none_answers":           extracted.count("NONE"),
        "num_reasoning_correct":      sum(1 for c in evals if c["reasoning_verdict"] == "correct"),
        "num_reasoning_suspicious":   sum(1 for c in evals if c["reasoning_verdict"] == "suspicious"),
        "num_reasoning_incorrect":    sum(1 for c in evals if c["reasoning_verdict"] == "incorrect"),
        "reasoning_correct_rate_at_k": round(
            sum(1 for c in evals if c["reasoning_verdict"] == "correct") / n, 4),
        "avg_score":                  round(sum(scores) / n, 2),
        "max_score":                  max(scores),
        "correct_answer_support_count": n_exact if has_gold else None,
        "correct_answer_best_score":    max(correct_scores)                        if correct_scores else None,
        "correct_answer_avg_score":     round(sum(correct_scores)/len(correct_scores), 2) if correct_scores else None,
    }

def compute_summary(results):
    s = {
        "num_questions": len(results),
        "num_questions_with_gold_answer": sum(1 for r in results if r["problem_has_gold_answer"]),
        "num_total_candidates": sum(len(r["candidates"]) for r in results),
        "num_exact_match": 0, "num_format_valid": 0,
        "num_reasoning_correct": 0, "num_reasoning_suspicious": 0,
        "num_pass_at_k": 0, "num_best_of_k_exact_match": 0,
        "num_majority_vote_exact_match": 0,
    }
    for r in results:
        agg = r["aggregate"]
        s["num_format_valid"]        += agg["num_format_valid"]
        s["num_reasoning_correct"]   += agg["num_reasoning_correct"]
        s["num_reasoning_suspicious"] += agg["num_reasoning_suspicious"]
        if r["problem_has_gold_answer"]:
            s["num_exact_match"]              += agg["num_exact_match"] or 0
            s["num_pass_at_k"]                += agg["pass_at_k"] or 0
            s["num_best_of_k_exact_match"]    += agg["best_of_k_exact_match"] or 0
            s["num_majority_vote_exact_match"] += agg["majority_vote_exact_match"] or 0
    return s

# ── Comparison report ─────────────────────────────────────────────────────────

def print_comparison(oss_file, claude_file):
    with open(oss_file)   as f: oss   = json.load(f)
    with open(claude_file) as f: cld  = json.load(f)

    o_by_id = {r["question_id"]: r for r in oss["results"]}
    c_by_id = {r["question_id"]: r for r in cld["results"]}

    sep = "=" * 100
    print(f"\n{sep}")
    print("COMPARISON: GPT-OSS 20B (generator) vs Claude Sonnet (generator)  |  Both judged by Claude Haiku")
    print(sep)
    hdr = "{:15s} {:5s} | {:>6s} {:>5s} {:>7s} {:>9s} {:>9s} | {:>6s} {:>5s} {:>7s} {:>9s} {:>9s}".format(
        "question_id","gold",
        "best","fmt%","avg_sc","reasoning","pass@k",
        "best","fmt%","avg_sc","reasoning","pass@k")
    sub = "{:15s} {:5s} | {:^47s} | {:^47s}".format(
        "","","-- GPT-OSS 20B --","-- Claude Sonnet --")
    print(sub)
    print(hdr)
    print("-" * 100)

    for qid in o_by_id:
        o = o_by_id[qid]["aggregate"]
        c = c_by_id[qid]["aggregate"]
        gold = "yes" if o_by_id[qid]["problem_has_gold_answer"] else "no"
        o_r = f"{o['num_reasoning_correct']}c/{o['num_reasoning_suspicious']}s"
        c_r = f"{c['num_reasoning_correct']}c/{c['num_reasoning_suspicious']}s"
        o_p = str(o["pass_at_k"]) if o["pass_at_k"] is not None else "-"
        c_p = str(c["pass_at_k"]) if c["pass_at_k"] is not None else "-"
        print("{:15s} {:5s} | {:>6s} {:>5.2f} {:>7.2f} {:>9s} {:>9s} | {:>6s} {:>5.2f} {:>7.2f} {:>9s} {:>9s}".format(
            qid, gold,
            str(o["best_extracted_answer"]), o["format_valid_rate_at_k"], o["avg_score"], o_r, o_p,
            str(c["best_extracted_answer"]), c["format_valid_rate_at_k"], c["avg_score"], c_r, c_p,
        ))

    print(sep)
    print("\nSUMMARY")
    print("-" * 60)
    os_s = oss["summary"]
    cl_s = cld["summary"]
    metrics = [
        "num_questions_with_gold_answer",
        "num_total_candidates",
        "num_format_valid",
        "num_reasoning_correct",
        "num_reasoning_suspicious",
        "num_exact_match",
        "num_pass_at_k",
        "num_best_of_k_exact_match",
        "num_majority_vote_exact_match",
    ]
    print(f"{'Metric':40s} {'OSS-20B':>12s} {'Claude-Sonnet':>14s}")
    print("-" * 68)
    for m in metrics:
        print(f"{m:40s} {str(os_s[m]):>12s} {str(cl_s[m]):>14s}")

    # Avg score across all questions
    o_avg = sum(r["aggregate"]["avg_score"] for r in oss["results"]) / len(oss["results"])
    c_avg = sum(r["aggregate"]["avg_score"] for r in cld["results"]) / len(cld["results"])
    print(f"{'avg_score_across_questions':40s} {o_avg:>12.2f} {c_avg:>14.2f}")
    print(f"{'format_valid_rate_overall':40s} {os_s['num_format_valid']/os_s['num_total_candidates']:>12.2%} {cl_s['num_format_valid']/cl_s['num_total_candidates']:>14.2%}")
    print(f"{'reasoning_correct_rate_overall':40s} {os_s['num_reasoning_correct']/os_s['num_total_candidates']:>12.2%} {cl_s['num_reasoning_correct']/cl_s['num_total_candidates']:>14.2%}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    questions = parse_val(VAL_FILE)
    print(f"Loaded {len(questions)} validation questions")

    # Resume from partial results if they exist
    results = []
    if Path(OUT_FILE).exists():
        with open(OUT_FILE) as f:
            existing = json.load(f)
        results = existing.get("results", [])
        done_ids = {r["question_id"] for r in results}
        questions = [q for q in questions if q["question_id"] not in done_ids]
        print(f"Resuming: {len(results)} done, {len(questions)} remaining")

    for i, q in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}] {q['question_id']}  gold={q['gold_answer']}")

        candidates = generate_candidates(q["problem"], K)

        print(f"  evaluating with Claude Haiku…")
        evals     = claude_eval(q["question_id"], q["problem"], q["gold_answer"], candidates)
        aggregate = compute_aggregate(q["gold_answer"], evals)

        print(f"  best={aggregate['best_extracted_answer']}  "
              f"pass@k={aggregate['pass_at_k']}  avg_score={aggregate['avg_score']}")

        results.append({
            "question_id":             q["question_id"],
            "problem_has_gold_answer": q["gold_answer"] is not None,
            "candidates":              evals,
            "aggregate":               aggregate,
        })
        # Save partial results after each question (allows resuming)
        with open(OUT_FILE, "w") as f:
            json.dump({"results": results, "summary": compute_summary(results)}, f, indent=2)
        time.sleep(0.5)

    summary = compute_summary(results)
    output  = {"results": results, "summary": summary}

    with open(OUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUT_FILE}")
    print(json.dumps(summary, indent=2))

    print_comparison(OSS_FILE, OUT_FILE)

if __name__ == "__main__":
    main()

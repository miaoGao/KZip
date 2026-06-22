"""Run the KnowledgeZip inference pipeline over a dataset and report EM / F1.

Examples:
    # greedy compression (1 sample), default example data
    python run.py

    # draw 5 candidate graphs per question and select the best by reward
    python run.py --num-samples 5

    # point at your own data / model
    python run.py --data data/example.jsonl --model Qwen3-4B --base-url http://localhost:8000/v1
"""

import argparse
import json

from knowledgezip import LLM, run_question
from knowledgezip.eval import evaluate


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    p = argparse.ArgumentParser(description="KnowledgeZip inference (compress -> select -> answer).")
    p.add_argument("--data", default="data/example.jsonl", help="JSONL file of questions.")
    p.add_argument("--num-samples", type=int, default=1, help="Candidate graphs per question; >1 enables reward selection.")
    p.add_argument("--gt-only", action="store_true", help="Use only supporting documents as context.")
    p.add_argument("--limit", type=int, default=0, help="Only run the first N questions (0 = all).")
    p.add_argument("--model", default=None, help="Model name (overrides KZIP_MODEL).")
    p.add_argument("--base-url", default=None, help="OpenAI-compatible base URL (overrides KZIP_BASE_URL).")
    p.add_argument("--out", default=None, help="Optional path to write predictions as JSONL.")
    args = p.parse_args()

    llm = LLM(model=args.model, base_url=args.base_url)
    questions = load_jsonl(args.data)
    if args.limit:
        questions = questions[: args.limit]

    results = []
    for i, q in enumerate(questions, 1):
        r = run_question(llm, q, num_samples=args.num_samples, gt_only=args.gt_only)
        results.append(r)
        print(f"[{i}/{len(questions)}] Q: {q['question']}")
        print(f"          pred: {r['pred_ans']!r}  gold: {q['answer']!r}  ({len(r['triples'])} triples)")

    if args.out:
        with open(args.out, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nPredictions written to {args.out}")

    print("\n=== Results ===")
    print(json.dumps(evaluate(results), indent=2))


if __name__ == "__main__":
    main()

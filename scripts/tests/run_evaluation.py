"""
Runs the NyayaSahayak retrieval evaluation suite.
Measures Recall@5, Recall@10, Recall@20, and MRR.
"""

import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.retrieval_service import HybridRetriever, RetrievalConfig


@dataclass
class EvalResult:
    query: str
    expected_case: str
    category: str
    hit_at_5: bool
    hit_at_10: bool
    hit_at_20: bool
    rank_of_hit: int  # -1 if not found in top 20
    retrieval_time: float


def normalize_case_name(name: str) -> str:
    # Normalize a case name for fuzzy matching
    name = name.lower()
    # Remove common legal noise
    for word in ["vs", "v.", "and", "ors", "anr", "etc", "the", "&"]:
        name = name.replace(word, " ")
    # Remove punctuation and extra spaces
    name = "".join(c for c in name if c.isalnum() or c.isspace())
    name = " ".join(name.split())
    return name


def check_hit(expected_case: str, retrieved_cases: List[str]) -> int:
    expected_norm = normalize_case_name(expected_case)
    expected_tokens = set(expected_norm.split())

    for rank, case in enumerate(retrieved_cases, start=1):
        case_norm = normalize_case_name(case)
        case_tokens = set(case_norm.split())

        # Check if significant tokens overlap
        # We require at least 2 tokens to match, or 50% of expected tokens
        overlap = expected_tokens & case_tokens
        if len(overlap) >= 2 or (len(expected_tokens) > 0 and len(overlap) / len(expected_tokens) >= 0.5):
            return rank

    return -1


def run_evaluation():
    # Load eval dataset
    eval_path = PROJECT_ROOT / "scripts" / "tests" / "eval_dataset.jsonl"
    if not eval_path.exists():
        print(f"ERROR: Evaluation dataset not found at {eval_path}")
        sys.exit(1)

    eval_queries = []
    with eval_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                eval_queries.append(json.loads(line))

    print(f"Loaded {len(eval_queries)} evaluation queries.")
    print("=" * 80)

    # Initialize retriever
    print("Loading retrieval models...")
    config = RetrievalConfig()
    retriever = HybridRetriever(config).load()
    print("Models loaded.\n")

    results: List[EvalResult] = []

    for i, item in enumerate(eval_queries, start=1):
        query = item["query"]
        expected_case = item["expected_case"]
        category = item.get("category", "unknown")

        print(f"[{i}/{len(eval_queries)}] {query[:70]}...")

        start_time = time.time()
        authorities = retriever.retrieve(query=query, final_top_k=20)
        elapsed = time.time() - start_time

        retrieved_cases = [a.case_name for a in authorities]

        rank = check_hit(expected_case, retrieved_cases)

        result = EvalResult(
            query=query,
            expected_case=expected_case,
            category=category,
            hit_at_5=(1 <= rank <= 5),
            hit_at_10=(1 <= rank <= 10),
            hit_at_20=(1 <= rank <= 20),
            rank_of_hit=rank,
            retrieval_time=elapsed,
        )
        results.append(result)

        status = f"FOUND at rank {rank}" if rank > 0 else "NOT FOUND in top 20"
        print(f"    Expected: {expected_case}")
        print(f"    Result:   {status} | Time: {elapsed:.2f}s")
        print()

    # Compute metrics
    total = len(results)
    recall_at_5 = sum(1 for r in results if r.hit_at_5) / total * 100
    recall_at_10 = sum(1 for r in results if r.hit_at_10) / total * 100
    recall_at_20 = sum(1 for r in results if r.hit_at_20) / total * 100

    # Mean Reciprocal Rank
    mrr = sum(1.0 / r.rank_of_hit if r.rank_of_hit > 0 else 0.0 for r in results) / total

    avg_latency = sum(r.retrieval_time for r in results) / total

    # Print summary
    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Total queries:    {total}")
    print(f"Recall@5:         {recall_at_5:.1f}%")
    print(f"Recall@10:        {recall_at_10:.1f}%")
    print(f"Recall@20:        {recall_at_20:.1f}%")
    print(f"MRR:              {mrr:.4f}")
    print(f"Avg latency:      {avg_latency:.2f}s")
    print("=" * 80)

    # Category breakdown
    categories = set(r.category for r in results)
    print("\nCATEGORY BREAKDOWN:")
    print("-" * 50)
    for cat in sorted(categories):
        cat_results = [r for r in results if r.category == cat]
        cat_recall = sum(1 for r in cat_results if r.hit_at_10) / len(cat_results) * 100
        print(f"  {cat:25s} | Recall@10: {cat_recall:.1f}% ({len(cat_results)} queries)")

    # Failed queries
    failed = [r for r in results if not r.hit_at_20]
    if failed:
        print(f"\nFAILED QUERIES ({len(failed)}):")
        print("-" * 50)
        for r in failed:
            print(f"  Query:    {r.query[:60]}...")
            print(f"  Expected: {r.expected_case}")
            print()

    # Save results
    output_path = PROJECT_ROOT / "scripts" / "tests" / "eval_results.json"
    output_data = {
        "metrics": {
            "total_queries": total,
            "recall_at_5": round(recall_at_5, 2),
            "recall_at_10": round(recall_at_10, 2),
            "recall_at_20": round(recall_at_20, 2),
            "mrr": round(mrr, 4),
            "avg_latency_seconds": round(avg_latency, 2),
        },
        "results": [
            {
                "query": r.query,
                "expected_case": r.expected_case,
                "category": r.category,
                "hit_at_5": r.hit_at_5,
                "hit_at_10": r.hit_at_10,
                "hit_at_20": r.hit_at_20,
                "rank_of_hit": r.rank_of_hit,
                "retrieval_time": round(r.retrieval_time, 3),
            }
            for r in results
        ],
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_evaluation()
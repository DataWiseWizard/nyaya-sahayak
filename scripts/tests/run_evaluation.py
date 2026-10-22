"""
Runs the NyayaSahayak retrieval evaluation suite.

Reports Recall@5/10/20 and MRR under two matching policies:
    - lenient  : ≥2 token overlap OR ≥50% of expected tokens (current behaviour)
    - strict   : normalized full-name equality OR SequenceMatcher ratio ≥ 0.9

The strict number is the honest headline. The lenient number captures
"case-family match" and is reported as a secondary signal.
"""

import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from typing import List

import re

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.retrieval_service import HybridRetriever, RetrievalConfig


@dataclass
class EvalResult:
    query: str
    expected_case: str
    category: str
    # Lenient (existing) fields — kept for backward compatibility
    hit_at_5: bool
    hit_at_10: bool
    hit_at_20: bool
    rank_of_hit: int
    # Strict fields
    hit_at_5_strict: bool
    hit_at_10_strict: bool
    hit_at_20_strict: bool
    rank_of_hit_strict: int
    # Timing
    retrieval_time: float


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
_NOISE_WORDS = ["vs", "v.", "and", "ors", "anr", "etc", "the", "&"]
_NOISE_REGEX = re.compile(r"\b(vs?\.?|and|ors|anr|etc|the)\b")


def normalize_case_name(name: str) -> str:
    """Lenient normalization (existing behaviour)."""
    name = name.lower()
    for word in _NOISE_WORDS:
        name = name.replace(word, " ")
    name = "".join(c for c in name if c.isalnum() or c.isspace())
    name = " ".join(name.split())
    return name


def normalize_case_name_strict(name: str) -> str:
    """Strict normalization: whole-word noise removal + punctuation strip."""
    name = name.lower()
    name = _NOISE_REGEX.sub(" ", name)
    name = re.sub(r"[^a-z0-9\s]", "", name)
    name = " ".join(name.split())
    return name


# ---------------------------------------------------------------------------
# Matchers
# ---------------------------------------------------------------------------
def check_hit_lenient(expected_case: str, retrieved_cases: List[str]) -> int:
    """Existing matcher: token-overlap heuristic."""
    expected_norm = normalize_case_name(expected_case)
    expected_tokens = set(expected_norm.split())

    for rank, case in enumerate(retrieved_cases, start=1):
        case_norm = normalize_case_name(case)
        case_tokens = set(case_norm.split())
        overlap = expected_tokens & case_tokens
        if (
            len(overlap) >= 2
            or (
                len(expected_tokens) > 0
                and len(overlap) / len(expected_tokens) >= 0.5
            )
        ):
            return rank
    return -1


def check_hit_strict(
    expected_case: str,
    retrieved_cases: List[str],
    similarity_threshold: float = 0.9,
) -> int:
    """Strict matcher: full-name equality or high-similarity fuzzy match."""
    expected_norm = normalize_case_name_strict(expected_case)
    if not expected_norm:
        return -1

    for rank, case in enumerate(retrieved_cases, start=1):
        case_norm = normalize_case_name_strict(case)
        if not case_norm:
            continue
        if case_norm == expected_norm:
            return rank
        ratio = SequenceMatcher(None, expected_norm, case_norm).ratio()
        if ratio >= similarity_threshold:
            return rank
    return -1


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------
def _metrics(results: List[EvalResult], suffix: str = "") -> dict:
    total = len(results)
    hit5 = sum(1 for r in results if getattr(r, f"hit_at_5{suffix}")) / total * 100
    hit10 = sum(1 for r in results if getattr(r, f"hit_at_10{suffix}")) / total * 100
    hit20 = sum(1 for r in results if getattr(r, f"hit_at_20{suffix}")) / total * 100
    ranks = [getattr(r, f"rank_of_hit{suffix}") for r in results]
    mrr = sum(1.0 / rank if rank > 0 else 0.0 for rank in ranks) / total
    return {
        "recall_at_5": round(hit5, 2),
        "recall_at_10": round(hit10, 2),
        "recall_at_20": round(hit20, 2),
        "mrr": round(mrr, 4),
    }


def run_evaluation():
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

        rank_lenient = check_hit_lenient(expected_case, retrieved_cases)
        rank_strict = check_hit_strict(expected_case, retrieved_cases)

        results.append(
            EvalResult(
                query=query,
                expected_case=expected_case,
                category=category,
                hit_at_5=(1 <= rank_lenient <= 5),
                hit_at_10=(1 <= rank_lenient <= 10),
                hit_at_20=(1 <= rank_lenient <= 20),
                rank_of_hit=rank_lenient,
                hit_at_5_strict=(1 <= rank_strict <= 5),
                hit_at_10_strict=(1 <= rank_strict <= 10),
                hit_at_20_strict=(1 <= rank_strict <= 20),
                rank_of_hit_strict=rank_strict,
                retrieval_time=elapsed,
            )
        )

        print(
            f"    Expected: {expected_case}\n"
            f"    Lenient:  rank {rank_lenient}\n"
            f"    Strict:   rank {rank_strict} | {elapsed:.2f}s\n"
        )

    lenient = _metrics(results, suffix="")
    strict = _metrics(results, suffix="_strict")
    avg_latency = sum(r.retrieval_time for r in results) / len(results)
    # Median is more informative than mean because of cold-start outliers
    sorted_lat = sorted(r.retrieval_time for r in results)
    median_latency = sorted_lat[len(sorted_lat) // 2]

    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Total queries:    {len(results)}")
    print(f"-- Lenient matcher --")
    for k, v in lenient.items():
        print(f"  {k:16s}: {v}")
    print(f"-- Strict matcher --")
    for k, v in strict.items():
        print(f"  {k:16s}: {v}")
    print(f"Avg latency:      {avg_latency:.2f}s")
    print(f"Median latency:   {median_latency:.2f}s")
    print("=" * 80)

    # Category breakdown (strict)
    categories = sorted(set(r.category for r in results))
    print("\nCATEGORY BREAKDOWN (strict Recall@10):")
    print("-" * 50)
    for cat in categories:
        cat_results = [r for r in results if r.category == cat]
        cat_recall = sum(1 for r in cat_results if r.hit_at_10_strict) / len(cat_results) * 100
        print(f"  {cat:25s} | Recall@10: {cat_recall:.1f}% ({len(cat_results)} queries)")

    # Failures (strict)
    failed = [r for r in results if not r.hit_at_20_strict]
    if failed:
        print(f"\nSTRICT FAILURES ({len(failed)}):")
        print("-" * 50)
        for r in failed:
            print(f"  Query:    {r.query[:60]}...")
            print(f"  Expected: {r.expected_case}\n")

    # Save
    output_path = PROJECT_ROOT / "scripts" / "tests" / "eval_results.json"
    output_data = {
        "metrics": {
            "total_queries": len(results),
            **lenient,
            "avg_latency_seconds": round(avg_latency, 2),
            "median_latency_seconds": round(median_latency, 2),
        },
        "metrics_strict": {
            "total_queries": len(results),
            **strict,
        },
        "results": [asdict(r) for r in results],
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_evaluation()
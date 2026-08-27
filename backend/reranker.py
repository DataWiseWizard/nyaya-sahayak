"""
Cross-encoder re-ranking sits between retrieval (rag_pipeline.py) and
generation (model_loader.py).

Why this matters: ChromaDB's embedding similarity search is fast but coarse
— it scores the question and each chunk independently, then compares
vectors. A cross-encoder instead reads the (question, chunk) pair together
in one forward pass, which is much more accurate at judging "does this
chunk actually answer this question" — but too slow to run over the whole
vector store. So the pattern is:

    1. Embedding search retrieves a wide net (e.g. top 20 candidates)
    2. Cross-encoder re-scores just those 20 and keeps the true top few
        (e.g. top 5) to hand to the LLM

This runs fully locally/offline — no network calls at inference time (only
the first time, to download the model weights).
"""

from __future__ import annotations
from sentence_transformers import CrossEncoder

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker_instance: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    """Lazily loads the cross-encoder once and reuses it across calls."""
    global _reranker_instance
    if _reranker_instance is None:
        print(f"Loading re-ranker ({RERANKER_MODEL_NAME})...")
        _reranker_instance = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker_instance


def rerank(question: str, documents: list[str], metadatas: list[dict],
        top_k: int = 5) -> tuple[list[str], list[dict], list[float]]:
    """
    Re-scores (question, document) pairs with the cross-encoder and returns
    the top_k documents/metadatas sorted by relevance, along with their
    scores (higher = more relevant).
    """
    if not documents:
        return [], [], []

    reranker = get_reranker()
    pairs = [(question, doc) for doc in documents]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(documents, metadatas, scores), key=lambda x: x[2], reverse=True
    )
    ranked = ranked[:top_k]

    docs_out = [r[0] for r in ranked]
    metas_out = [r[1] for r in ranked]
    scores_out = [float(r[2]) for r in ranked]
    return docs_out, metas_out, scores_out
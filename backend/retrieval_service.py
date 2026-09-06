"""
This module is UI-independent. It performs:
1. Dense retrieval from ChromaDB.
2. Keyword retrieval from SQLite FTS5.
3. Reciprocal Rank Fusion.
4. Cross-encoder reranking.

It returns structured RetrievedAuthority objects that can later be used by:
- context_builder.py
- generation_service.py
- citation_verifier.py
- app.py
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer

logger = logging.getLogger("nyaya.retrieval")


STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "then",
    "else",
    "when",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "once",
    "here",
    "there",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "can",
    "will",
    "just",
    "should",
    "now",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "doing",
    "would",
    "could",
    "may",
    "might",
    "must",
    "shall",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "these",
    "those",
    "i",
    "me",
    "my",
    "we",
    "our",
    "you",
    "your",
    "he",
    "him",
    "his",
    "she",
    "her",
    "it",
    "its",
    "they",
    "them",
    "their",
}


@dataclass
class RetrievalConfig:
    # Configuration for the NyayaSahayak retrieval service.

    vault_dir: str = "vault"
    collection_name: str = "nyaya_sahayak_v1"
    keyword_db_path: str = "vault/nyaya_keyword_porter.db"

    embedding_model_name: str = "multi-qa-mpnet-base-dot-v1"
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    device: str = "cuda"

    dense_k: int = 80
    keyword_k: int = 80
    rerank_top_n: int = 40
    final_top_k: int = 8
    rrf_k: int = 60

    use_reranker: bool = True


@dataclass
class RetrievedAuthority:
    # A single retrieved judgment chunk with metadata and retrieval scores.

    chunk_id: str
    doc_id: str
    case_name: str
    judgment_date: str
    source_file: str
    chunk_index: int
    text: str

    dense_rank: Optional[int] = None
    keyword_rank: Optional[int] = None

    dense_distance: Optional[float] = None
    bm25_score: Optional[float] = None

    rrf_score: float = 0.0
    rerank_score: Optional[float] = None

    retrieval_sources: List[str] = field(default_factory=list)


def build_fts_query(query: str) -> Optional[str]:
    """
    Converts a natural-language query into a safe SQLite FTS5 MATCH query.

    Example:
        "kidnapping for ransom under Section 302 IPC"

    becomes approximately:

        ("kidnapping" OR "ransom" OR "Section" OR "302" OR "IPC"
        OR "kidnapping ransom Section 302 IPC")
    """
    tokens = re.findall(r"[A-Za-z0-9]+", query)

    cleaned_tokens: List[str] = []
    seen = set()

    for token in tokens:
        lower_token = token.lower()

        if lower_token in STOPWORDS:
            continue

        if not token.isdigit() and len(token) <= 1:
            continue

        if lower_token not in seen:
            seen.add(lower_token)
            cleaned_tokens.append(token)

    if not cleaned_tokens:
        return None

    cleaned_tokens = cleaned_tokens[:20]

    parts = [f'"{token}"' for token in cleaned_tokens]

    if len(cleaned_tokens) >= 2:
        phrase = " ".join(cleaned_tokens[:6])
        parts.append(f'"{phrase}"')

    return "(" + " OR ".join(parts) + ")"


class HybridRetriever:
    """
    Hybrid retriever combining:
    - ChromaDB dense vector search
    - SQLite FTS5 keyword search
    - Reciprocal Rank Fusion
    - Cross-encoder reranking
    """

    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()

        self._loaded = False

        self.embedder: Optional[SentenceTransformer] = None
        self.reranker: Optional[CrossEncoder] = None

        self.chroma_client: Optional[chromadb.api.ClientAPI] = None
        self.collection = None

        self.keyword_conn: Optional[sqlite3.Connection] = None

    def load(self) -> "HybridRetriever":
        """
        Loads models and connects to retrieval stores.

        This is intentionally separate from __init__ so that Streamlit can
        cache the loaded retriever with @st.cache_resource.
        """
        if self._loaded:
            return self

        logger.info("Loading embedding model: %s", self.config.embedding_model_name)
        self.embedder = SentenceTransformer(
            self.config.embedding_model_name,
            device=self.config.device,
        )

        if self.config.use_reranker:
            logger.info("Loading reranker model: %s", self.config.reranker_model_name)
            self.reranker = CrossEncoder(
                self.config.reranker_model_name,
                device=self.config.device,
            )
        else:
            self.reranker = None

        logger.info("Connecting to ChromaDB vault: %s", self.config.vault_dir)
        self.chroma_client = chromadb.PersistentClient(path=self.config.vault_dir)
        self.collection = self.chroma_client.get_collection(
            name=self.config.collection_name
        )

        keyword_db_path = Path(self.config.keyword_db_path).expanduser().resolve()

        if not keyword_db_path.exists():
            raise FileNotFoundError(
                f"SQLite keyword database not found: {keyword_db_path}. "
                "Run the keyword index builder first."
            )

        logger.info("Connecting to SQLite FTS5 database: %s", keyword_db_path)
        self.keyword_conn = sqlite3.connect(
            str(keyword_db_path),
            check_same_thread=False,
        )

        self._loaded = True
        logger.info("HybridRetriever loaded successfully.")

        return self

    def retrieve(
        self,
        query: str,
        final_top_k: Optional[int] = None,
        dense_k: Optional[int] = None,
        keyword_k: Optional[int] = None,
        rerank_top_n: Optional[int] = None,
    ) -> List[RetrievedAuthority]:

        # Retrieves ranked judgment chunks for a legal query
        self.load()

        final_top_k = final_top_k or self.config.final_top_k
        dense_k = dense_k or self.config.dense_k
        keyword_k = keyword_k or self.config.keyword_k
        rerank_top_n = rerank_top_n or self.config.rerank_top_n

        query = query.strip()

        if not query:
            return []

        start_total = time.time()

        dense_items = self._dense_search(query, dense_k)
        keyword_items = self._keyword_search(query, keyword_k)

        logger.info(
            "Retrieved %s dense candidates and %s keyword candidates.",
            len(dense_items),
            len(keyword_items),
        )

        fused_items = self._reciprocal_rank_fusion(dense_items, keyword_items)

        logger.info("Fused candidates: %s", len(fused_items))

        if self.reranker is not None:
            fused_items = self._rerank(query, fused_items, rerank_top_n)

        final_items = fused_items[:final_top_k]

        elapsed_total = time.time() - start_total
        logger.info("Total retrieval time: %.3f seconds", elapsed_total)

        return final_items

    def _dense_search(self, query: str, top_k: int) -> List[RetrievedAuthority]:
        # Performs dense vector search using ChromaDB.
        if self.embedder is None or self.collection is None:
            raise RuntimeError("Retriever is not loaded. Call load() first.")

        start = time.time()

        query_embedding = self.embedder.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        result = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        elapsed = time.time() - start
        logger.info("Dense search time: %.3f seconds", elapsed)

        items: List[RetrievedAuthority] = []

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        if not ids:
            return items

        for rank, chunk_id in enumerate(ids, start=1):
            metadata = metadatas[rank - 1] if rank - 1 < len(metadatas) else {}
            metadata = metadata or {}

            document = documents[rank - 1] if rank - 1 < len(documents) else ""
            distance = distances[rank - 1] if rank - 1 < len(distances) else None

            items.append(
                RetrievedAuthority(
                    chunk_id=str(chunk_id),
                    doc_id=str(metadata.get("doc_id", "Unknown")),
                    case_name=str(metadata.get("case_name", "Unknown")),
                    judgment_date=str(metadata.get("judgment_date", "Unknown")),
                    source_file=str(metadata.get("source_file", "Unknown")),
                    chunk_index=int(metadata.get("chunk_index", 0) or 0),
                    text=str(document or ""),
                    dense_rank=rank,
                    dense_distance=float(distance) if distance is not None else None,
                    retrieval_sources=["dense"],
                )
            )

        return items

    def _keyword_search(self, query: str, top_k: int) -> List[RetrievedAuthority]:
        # Performs keyword search using SQLite FTS5.
        if self.keyword_conn is None:
            raise RuntimeError("Retriever is not loaded. Call load() first.")

        fts_query = build_fts_query(query)

        if not fts_query:
            return []

        logger.info("FTS5 query: %s", fts_query)

        start = time.time()

        sql_with_metadata = """
            SELECT
                chunks_fts.chunk_id,
                chunks_fts.doc_id,
                chunks_fts.case_name,
                chunks_fts.judgment_date,
                chunks_fts.text,
                chunk_metadata.source_file,
                chunk_metadata.chunk_index,
                bm25(chunks_fts) AS score
            FROM chunks_fts
            LEFT JOIN chunk_metadata
                ON chunk_metadata.chunk_id = chunks_fts.chunk_id
            WHERE chunks_fts MATCH ?
            ORDER BY score
            LIMIT ?
        """

        sql_without_metadata = """
            SELECT
                chunk_id,
                doc_id,
                case_name,
                judgment_date,
                text,
                bm25(chunks_fts) AS score
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY score
            LIMIT ?
        """

        rows = []
        has_metadata = True

        try:
            rows = self.keyword_conn.execute(
                sql_with_metadata,
                (fts_query, top_k),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            logger.warning(
                "Metadata join failed, falling back to FTS-only query: %s",
                exc,
            )
            has_metadata = False
            rows = self.keyword_conn.execute(
                sql_without_metadata,
                (fts_query, top_k),
            ).fetchall()

        elapsed = time.time() - start
        logger.info("Keyword search time: %.3f seconds", elapsed)

        items: List[RetrievedAuthority] = []

        for rank, row in enumerate(rows, start=1):
            if has_metadata:
                (
                    chunk_id,
                    doc_id,
                    case_name,
                    judgment_date,
                    text,
                    source_file,
                    chunk_index,
                    score,
                ) = row
            else:
                chunk_id, doc_id, case_name, judgment_date, text, score = row
                source_file = "Unknown"
                chunk_index = 0

            items.append(
                RetrievedAuthority(
                    chunk_id=str(chunk_id),
                    doc_id=str(doc_id or "Unknown"),
                    case_name=str(case_name or "Unknown"),
                    judgment_date=str(judgment_date or "Unknown"),
                    source_file=str(source_file or "Unknown"),
                    chunk_index=int(chunk_index or 0),
                    text=str(text or ""),
                    keyword_rank=rank,
                    bm25_score=float(score) if score is not None else None,
                    retrieval_sources=["keyword"],
                )
            )

        return items

    def _reciprocal_rank_fusion(
        self,
        dense_items: List[RetrievedAuthority],
        keyword_items: List[RetrievedAuthority],
    ) -> List[RetrievedAuthority]:
        # Merges dense and keyword results using Reciprocal Rank Fusion.
        candidates: Dict[str, RetrievedAuthority] = {}

        for rank, item in enumerate(dense_items, start=1):
            candidate = candidates.get(item.chunk_id)

            if candidate is None:
                candidate = item
                candidates[item.chunk_id] = candidate
            else:
                self._merge_candidate(candidate, item)
                if "dense" not in candidate.retrieval_sources:
                    candidate.retrieval_sources.append("dense")

            candidate.dense_rank = rank
            candidate.dense_distance = item.dense_distance
            candidate.rrf_score += 1.0 / (self.config.rrf_k + rank)

        for rank, item in enumerate(keyword_items, start=1):
            candidate = candidates.get(item.chunk_id)

            if candidate is None:
                candidate = item
                candidates[item.chunk_id] = candidate
            else:
                self._merge_candidate(candidate, item)
                if "keyword" not in candidate.retrieval_sources:
                    candidate.retrieval_sources.append("keyword")

            candidate.keyword_rank = rank
            candidate.bm25_score = item.bm25_score
            candidate.rrf_score += 1.0 / (self.config.rrf_k + rank)

        fused = sorted(
            candidates.values(),
            key=lambda chunk: chunk.rrf_score,
            reverse=True,
        )

        return fused

    @staticmethod
    def _merge_candidate(
        target: RetrievedAuthority,
        source: RetrievedAuthority,
    ) -> None:
        # Fills missing fields in the target candidate from the source candidate.
        if not target.text and source.text:
            target.text = source.text

        if target.case_name == "Unknown" and source.case_name != "Unknown":
            target.case_name = source.case_name

        if target.judgment_date == "Unknown" and source.judgment_date != "Unknown":
            target.judgment_date = source.judgment_date

        if target.source_file == "Unknown" and source.source_file != "Unknown":
            target.source_file = source.source_file

        if target.chunk_index == 0 and source.chunk_index != 0:
            target.chunk_index = source.chunk_index

    def _rerank(
        self,
        query: str,
        candidates: List[RetrievedAuthority],
        rerank_top_n: int,
    ) -> List[RetrievedAuthority]:
        # Reranks the top fused candidates using a cross-encoder.
        if self.reranker is None:
            return candidates

        if not candidates:
            return candidates

        to_rerank = candidates[:rerank_top_n]

        pairs = [(query, candidate.text) for candidate in to_rerank]

        start = time.time()
        scores = self.reranker.predict(pairs)
        elapsed = time.time() - start

        logger.info(
            "Reranked %s candidates in %.3f seconds",
            len(to_rerank),
            elapsed,
        )

        try:
            scores = list(scores)
        except TypeError:
            scores = [scores]

        for candidate, score in zip(to_rerank, scores):
            candidate.rerank_score = float(score)

        to_rerank.sort(
            key=lambda candidate: candidate.rerank_score
            if candidate.rerank_score is not None
            else -1e9,
            reverse=True,
        )

        return to_rerank + candidates[rerank_top_n:]

    def close(self) -> None:
        # Closes database connections
        if self.keyword_conn is not None:
            self.keyword_conn.close()
            self.keyword_conn = None

        self._loaded = False


def print_results(results: List[RetrievedAuthority]) -> None:
    # Readable CLI output

    print("=" * 100)
    print("NyayaSahayak Retrieval Service Results")
    print("=" * 100)

    if not results:
        print("No results found.")
        return

    for rank, result in enumerate(results, start=1):
        print(f"RANK {rank}")
        print(f"Case            : {result.case_name}")
        print(f"Judgment date   : {result.judgment_date}")
        print(f"Doc ID          : {result.doc_id}")
        print(f"Chunk ID        : {result.chunk_id}")
        print(f"Source file     : {result.source_file}")
        print(f"Chunk index     : {result.chunk_index}")
        print(f"Sources         : {', '.join(result.retrieval_sources)}")
        print(f"RRF score       : {result.rrf_score:.6f}")

        if result.rerank_score is not None:
            print(f"Rerank score    : {result.rerank_score:.4f}")

        if result.dense_rank is not None:
            distance_text = (
                f"{result.dense_distance:.4f}"
                if result.dense_distance is not None
                else "Unknown"
            )
            print(f"Dense rank      : {result.dense_rank} | Distance: {distance_text}")

        if result.keyword_rank is not None:
            bm25_text = (
                f"{result.bm25_score:.4f}"
                if result.bm25_score is not None
                else "Unknown"
            )
            print(f"Keyword rank    : {result.keyword_rank} | BM25: {bm25_text}")

        snippet = result.text.strip().replace("\n", " ")

        if len(snippet) > 800:
            snippet = snippet[:800] + "..."

        print("Snippet:")
        print(snippet)
        print("-" * 100)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test the NyayaSahayak hybrid retrieval service."
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Legal query to retrieve for.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of final results.",
    )
    parser.add_argument(
        "--dense-k",
        type=int,
        default=None,
        help="Number of dense candidates.",
    )
    parser.add_argument(
        "--keyword-k",
        type=int,
        default=None,
        help="Number of keyword candidates.",
    )
    parser.add_argument(
        "--rerank-top-n",
        type=int,
        default=None,
        help="Number of fused candidates to rerank.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for models: cuda or cpu.",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable reranking.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print results as JSON.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    config = RetrievalConfig(
        device=args.device,
        use_reranker=not args.no_rerank,
    )

    retriever = HybridRetriever(config).load()

    results = retriever.retrieve(
        query=args.query,
        final_top_k=args.top_k,
        dense_k=args.dense_k,
        keyword_k=args.keyword_k,
        rerank_top_n=args.rerank_top_n,
    )

    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2, ensure_ascii=False))
    else:
        print_results(results)

    retriever.close()


if __name__ == "__main__":
    main()
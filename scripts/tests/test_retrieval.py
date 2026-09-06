#!/usr/bin/env python3
"""
test_retrieval.py

Phase 3.1: Verify dense retrieval from the ChromaDB vault.

This script:
1. Loads the same embedding model used during ingestion.
2. Encodes a legal query.
3. Queries the ChromaDB collection.
4. Prints the top matching judgment chunks.
"""

import argparse
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test dense retrieval from NyayaSahayak ChromaDB vault."
    )
    parser.add_argument(
        "--vault-dir",
        default="vault",
        help="Path to ChromaDB vault directory.",
    )
    parser.add_argument(
        "--collection-name",
        default="nyaya_sahayak_v1",
        help="ChromaDB collection name.",
    )
    parser.add_argument(
        "--model-name",
        default="all-MiniLM-L6-v2",
        help="Exact embedding model used during ingestion.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for embedding model: cpu or cuda.",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Legal query to test.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of chunks to retrieve.",
    )
    args = parser.parse_args()

    vault_dir = Path(args.vault_dir).expanduser().resolve()

    if not vault_dir.exists():
        raise SystemExit(f"Vault directory not found: {vault_dir}")

    print("=" * 80)
    print("NyayaSahayak Retrieval Test")
    print("=" * 80)
    print(f"Vault directory : {vault_dir}")
    print(f"Collection name : {args.collection_name}")
    print(f"Embedding model : {args.model_name}")
    print(f"Device          : {args.device}")
    print(f"Query           : {args.query}")
    print("=" * 80)

    client = chromadb.PersistentClient(path=str(vault_dir))
    collection = client.get_collection(name=args.collection_name)

    print(f"Collection count: {collection.count()}")

    print("Loading embedding model...")
    model = SentenceTransformer(args.model_name, device=args.device)

    print("Encoding query...")
    start_encode = time.time()
    query_embedding = model.encode(
        [args.query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    encode_time = time.time() - start_encode
    print(f"Query embedding shape: {query_embedding.shape}")
    print(f"Embedding time: {encode_time:.3f} seconds")

    print("Querying ChromaDB...")
    start_query = time.time()
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=args.top_k,
        include=["documents", "metadatas", "distances"],
    )
    query_time = time.time() - start_query

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    print(f"ChromaDB query time: {query_time:.3f} seconds")
    print("=" * 80)

    if not ids:
        print("No results found.")
        return

    for rank, (chunk_id, document, metadata, distance) in enumerate(
        zip(ids, documents, metadatas, distances),
        start=1,
    ):
        document = document or ""
        metadata = metadata or {}

        case_name = metadata.get("case_name", "Unknown")
        judgment_date = metadata.get("judgment_date", "Unknown")
        doc_id = metadata.get("doc_id", "Unknown")
        chunk_index = metadata.get("chunk_index", "Unknown")
        source_file = metadata.get("source_file", "Unknown")

        snippet = document.strip()
        if len(snippet) > 600:
            snippet = snippet[:600] + "..."

        print(f"RANK {rank}")
        print(f"Distance        : {distance:.4f}")
        print(f"Case            : {case_name}")
        print(f"Judgment date   : {judgment_date}")
        print(f"Doc ID          : {doc_id}")
        print(f"Chunk index     : {chunk_index}")
        print(f"Source file     : {source_file}")
        print(f"Chunk ID        : {chunk_id}")
        print("Snippet:")
        print(snippet)
        print("-" * 80)


if __name__ == "__main__":
    main()
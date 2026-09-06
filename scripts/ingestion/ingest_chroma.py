#!/usr/bin/env python3
"""
ingest_chroma.py

Phase 2 ingestion pipeline for NyayaSahayak.
Reads cleaned chunks and quality flags, filters out bad documents,
and ingests the rest into a persistent ChromaDB vector store.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List

import chromadb
import torch
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger("ingest_chroma")


# ---------------------------------------------------------------------------
# Custom GPU Embedding Function
# ---------------------------------------------------------------------------
class CudaEmbeddingFunction(EmbeddingFunction):
    """
    Wraps a sentence-transformers model on CUDA so that ChromaDB
    uses the GPU for every embedding call.

    The internal batch_size controls how many chunks are encoded
    in a single GPU forward-pass.  For an RTX 4060 (8 GB VRAM),
    512 is a safe default.  Increase to 1024 if you want to push
    the GPU harder; decrease to 256 if you hit OOM errors.
    """

    def __init__(
        self,
        model_name: str = "multi-qa-mpnet-base-dot-v1",
        device: str = "cuda",
        batch_size: int = 512,
    ):
        logger.info("Loading embedding model '%s' on %s ...", model_name, device)
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        logger.info(
            "Model loaded. Parameters: %s | Device: %s",
            sum(p.numel() for p in self.model.parameters()),
            self.model.device,
        )

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self.model.encode(
            input,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,   # ChromaDB already shows its own progress
            normalize_embeddings=True, # cosine similarity works best with L2-normalised vectors
        )
        return embeddings.tolist()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU-accelerated ingestion of cleaned SC chunks into ChromaDB."
    )
    parser.add_argument(
        "--flags",
        default="data/cleaned/quality_flags.jsonl",
        help="Path to quality_flags.jsonl",
    )
    parser.add_argument(
        "--chunks",
        default="data/cleaned/judgment_chunks.jsonl",
        help="Path to judgment_chunks.jsonl",
    )
    parser.add_argument(
        "--vault-dir",
        default="vault",
        help="Directory where the ChromaDB persistent vault will be stored.",
    )
    parser.add_argument(
        "--collection-name",
        default="nyaya_sahayak_v1",
        help="Name of the ChromaDB collection.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Number of chunks to add to ChromaDB per batch.",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=512,
        help="Number of chunks the GPU encodes in a single forward-pass.",
    )
    parser.add_argument(
        "--model-name",
        default="multi-qa-mpnet-base-dot-v1",
        help="Sentence-transformers model to use for embeddings.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device: 'cuda' for GPU, 'cpu' for CPU fallback.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # ------------------------------------------------------------------
    # 0. Verify GPU availability
    # ------------------------------------------------------------------
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning(
            "CUDA is not available. Falling back to CPU. "
            "Install PyTorch with CUDA support: "
            "pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )
        args.device = "cpu"
    else:
        logger.info("Using device: %s", args.device)
        if args.device == "cuda":
            logger.info("GPU: %s", torch.cuda.get_device_name(0))
            logger.info(
                "VRAM: %.1f GB",
                torch.cuda.get_device_properties(0).total_memory / (1024 ** 3),
            )

    # ------------------------------------------------------------------
    # 1. Load Quality Flags (Whitelist)
    # ------------------------------------------------------------------
    flags_path = Path(args.flags).expanduser().resolve()
    if not flags_path.exists():
        raise SystemExit(
            f"Quality flags not found at {flags_path}. "
            "Run build_quality_flags.py first."
        )

    logger.info("Loading approved document IDs ...")
    include_ids: set = set()
    with flags_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("include_in_first_index"):
                include_ids.add(record["doc_id"])

    logger.info("Loaded %d approved doc_ids into whitelist.", len(include_ids))

    # ------------------------------------------------------------------
    # 2. Initialise GPU embedding function
    # ------------------------------------------------------------------
    embedding_fn = CudaEmbeddingFunction(
        model_name=args.model_name,
        device=args.device,
        batch_size=args.embed_batch_size,
    )

    # ------------------------------------------------------------------
    # 3. Initialise ChromaDB
    # ------------------------------------------------------------------
    vault_dir = Path(args.vault_dir).expanduser().resolve()
    vault_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Connecting to ChromaDB vault at %s ...", vault_dir)
    client = chromadb.PersistentClient(path=str(vault_dir))

    collection = client.get_or_create_collection(
        name=args.collection_name,
        embedding_function=embedding_fn,
        metadata={
            "hnsw:space": "cosine",
            "hnsw:construction_ef": 10,
            "hnsw:search_ef": 10,
            "hnsw:M": 16
        },
    )

    # ------------------------------------------------------------------
    # 4. Stream, filter, and batch-ingest
    # ------------------------------------------------------------------
    chunks_path = Path(args.chunks).expanduser().resolve()
    if not chunks_path.exists():
        raise SystemExit(f"Chunks file not found at {chunks_path}.")

    logger.info("Starting streaming ingestion ...")

    batch_ids: List[str] = []
    batch_documents: List[str] = []
    batch_metadatas: List[dict] = []

    total_ingested = 0
    total_skipped = 0
    start_time = time.time()

    # Count total lines for progress bar
    total_lines = sum(1 for _ in chunks_path.open("r", encoding="utf-8"))

    with chunks_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="Ingesting chunks"):
            chunk = json.loads(line)
            doc_id = chunk.get("doc_id")

            # Filter out bad documents
            if doc_id not in include_ids:
                total_skipped += 1
                continue

            # ChromaDB metadata values MUST be str, int, float, or bool.
            metadata = {
                "doc_id": str(doc_id),
                "case_name": str(chunk.get("case_name", "Unknown")),
                "judgment_date": str(chunk.get("judgment_date") or "Unknown"),
                "source_file": str(chunk.get("source_file", "Unknown")),
                "chunk_index": int(chunk.get("chunk_index", 0)),
            }

            batch_ids.append(chunk["chunk_id"])
            batch_documents.append(chunk["text"])
            batch_metadatas.append(metadata)

            # Push batch to ChromaDB
            if len(batch_ids) >= args.batch_size:
                collection.add(
                    ids=batch_ids,
                    documents=batch_documents,
                    metadatas=batch_metadatas,
                )
                total_ingested += len(batch_ids)
                batch_ids, batch_documents, batch_metadatas = [], [], []

    # Push any remaining chunks
    if batch_ids:
        collection.add(
            ids=batch_ids,
            documents=batch_documents,
            metadatas=batch_metadatas,
        )
        total_ingested += len(batch_ids)

    elapsed = time.time() - start_time

    logger.info("Ingestion complete!")
    logger.info("Total chunks ingested : %d", total_ingested)
    logger.info("Total chunks skipped  : %d", total_skipped)
    logger.info("Time taken            : %.2f seconds (%.2f minutes)", elapsed, elapsed / 60)
    logger.info("Final collection count: %d", collection.count())


if __name__ == "__main__":
    main()
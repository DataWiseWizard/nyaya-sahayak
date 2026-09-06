#!/usr/bin/env python3
"""
resume_ingest.py
Resumable ingestion pipeline. Picks up exactly where it left off.
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

logger = logging.getLogger("resume_ingest")

class CudaEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str = "multi-qa-mpnet-base-dot-v1", device: str = "cuda", batch_size: int = 128):
        logger.info("Loading embedding model '%s' on %s ...", model_name, device)
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self.model.encode(
            input,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable ingestion.")
    parser.add_argument("--flags", default="data/cleaned/quality_flags.jsonl")
    parser.add_argument("--chunks", default="data/cleaned/judgment_chunks.jsonl")
    parser.add_argument("--vault-dir", default="vault")
    parser.add_argument("--collection-name", default="nyaya_sahayak_v1")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--embed-batch-size", type=int, default=128)
    parser.add_argument("--model-name", default="multi-qa-mpnet-base-dot-v1")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA not available.")

    # 1. Load Whitelist
    flags_path = Path(args.flags).resolve()
    include_ids = set()
    with flags_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("include_in_first_index"):
                include_ids.add(record["doc_id"])
    logger.info("Loaded %d approved doc_ids.", len(include_ids))

    # 2. Connect to EXISTING vault
    vault_dir = Path(args.vault_dir).resolve()
    client = chromadb.PersistentClient(path=str(vault_dir))
    
    embedding_fn = CudaEmbeddingFunction(
        model_name=args.model_name,
        device=args.device,
        batch_size=args.embed_batch_size,
    )
    
    # This will load your existing 413k vectors!
    collection = client.get_or_create_collection(
        name=args.collection_name,
        embedding_function=embedding_fn,
        metadata={
            "hnsw:space": "cosine",
            "hnsw:construction_ef": 10,
            "hnsw:search_ef": 10,
            "hnsw:M": 16
        }
    )
    
    current_db_count = collection.count()
    logger.info(f"Current database count: {current_db_count}")

    # 3. Checkpoint Logic
    checkpoint_file = Path("ingest_checkpoint.txt")
    valid_chunks_to_skip = 0
    if checkpoint_file.exists():
        valid_chunks_to_skip = int(checkpoint_file.read_text().strip())
        logger.info(f"Resuming from checkpoint: skipping first {valid_chunks_to_skip} valid chunks.")
    else:
        logger.info("No checkpoint found. Syncing with current DB count...")
        valid_chunks_to_skip = current_db_count

    chunks_path = Path(args.chunks).resolve()
    total_lines = sum(1 for _ in chunks_path.open("r", encoding="utf-8"))

    valid_chunks_seen = 0
    batch_ids: List[str] = []
    batch_documents: List[str] = []
    batch_metadatas: List[dict] = []
    
    total_ingested_this_run = 0

    with chunks_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="Processing"):
            chunk = json.loads(line)
            doc_id = chunk.get("doc_id")

            if doc_id not in include_ids:
                continue

            valid_chunks_seen += 1

            # Skip already processed chunks instantly
            if valid_chunks_seen <= valid_chunks_to_skip:
                continue

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

            if len(batch_ids) >= args.batch_size:
                try:
                    collection.add(
                        ids=batch_ids,
                        documents=batch_documents,
                        metadatas=batch_metadatas,
                    )
                    valid_chunks_to_skip += len(batch_ids)
                    total_ingested_this_run += len(batch_ids)
                    
                    # SAVE CHECKPOINT IMMEDIATELY
                    checkpoint_file.write_text(str(valid_chunks_to_skip))
                    
                    # Clear batches
                    batch_ids, batch_documents, batch_metadatas = [], [], []
                    
                    # Crucial: Small sleep to let CPU catch up on HNSW graph building
                    # and prevent Windows TDR (GPU Driver Timeout)
                    time.sleep(0.05) 
                    
                except Exception as e:
                    logger.error(f"Batch failed: {e}")
                    logger.info(f"Checkpoint saved at {valid_chunks_to_skip}. You can safely restart.")
                    checkpoint_file.write_text(str(valid_chunks_to_skip))
                    return

        # Final batch
        if batch_ids:
            try:
                collection.add(
                    ids=batch_ids,
                    documents=batch_documents,
                    metadatas=batch_metadatas,
                )
                valid_chunks_to_skip += len(batch_ids)
                total_ingested_this_run += len(batch_ids)
                checkpoint_file.write_text(str(valid_chunks_to_skip))
            except Exception as e:
                logger.error(f"Final batch failed: {e}")
                checkpoint_file.write_text(str(valid_chunks_to_skip))
                return

    logger.info(f"Done! Ingested {total_ingested_this_run} new chunks this run.")
    logger.info(f"Final collection count: {collection.count()}")

if __name__ == "__main__":
    main()
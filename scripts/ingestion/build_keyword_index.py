#!/usr/bin/env python3
"""
build_keyword_index.py

Builds a SQLite FTS5 keyword index from the cleaned judgment chunks.
This enables exact keyword matching (BM25) to complement ChromaDB's vector search.
"""

import argparse
import json
import logging
import sqlite3
import time
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger("build_keyword_index")

def main() -> None:
    parser = argparse.ArgumentParser(description="Build SQLite FTS5 keyword index.")
    parser.add_argument("--flags", default="data/cleaned/quality_flags.jsonl")
    parser.add_argument("--chunks", default="data/cleaned/judgment_chunks.jsonl")
    parser.add_argument("--db-path", default="vault/nyaya_keyword.db")
    parser.add_argument("--batch-size", type=int, default=5000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # 1. Load Whitelist
    flags_path = Path(args.flags).resolve()
    include_ids = set()
    with flags_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("include_in_first_index"):
                include_ids.add(record["doc_id"])
    logger.info(f"Loaded {len(include_ids)} approved doc_ids.")

    # 2. Setup SQLite Database
    db_path = Path(args.db_path).resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Delete existing DB to start fresh
    if db_path.exists():
        db_path.unlink()

    logger.info(f"Creating SQLite database at {db_path}...")
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Create FTS5 virtual table
    # We use 'unicode61' tokenizer which handles punctuation and numbers well, 
    # which is critical for legal citations like "302 IPC" or "AIR 1950".
    cursor.execute("""
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            chunk_id,
            doc_id,
            case_name,
            judgment_date,
            text,
            tokenize='porter unicode61'
        )
    """)
    
    # Create a standard table for metadata mapping (FTS5 doesn't easily allow complex metadata filtering)
    cursor.execute("""
        CREATE TABLE chunk_metadata (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT,
            case_name TEXT,
            judgment_date TEXT,
            source_file TEXT,
            chunk_index INTEGER
        )
    """)
    conn.commit()

    # 3. Stream and Insert
    chunks_path = Path(args.chunks).resolve()
    total_lines = sum(1 for _ in chunks_path.open("r", encoding="utf-8"))
    
    batch_fts = []
    batch_meta = []
    total_ingested = 0
    start_time = time.time()

    logger.info("Starting keyword index ingestion...")
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="Indexing keywords"):
            chunk = json.loads(line)
            doc_id = chunk.get("doc_id")

            if doc_id not in include_ids:
                continue

            chunk_id = chunk["chunk_id"]
            case_name = chunk.get("case_name", "Unknown")
            judgment_date = chunk.get("judgment_date") or "Unknown"
            text = chunk.get("text", "")
            source_file = chunk.get("source_file", "Unknown")
            chunk_index = chunk.get("chunk_index", 0)

            batch_fts.append((chunk_id, doc_id, case_name, judgment_date, text))
            batch_meta.append((chunk_id, doc_id, case_name, judgment_date, source_file, chunk_index))

            if len(batch_fts) >= args.batch_size:
                cursor.executemany("INSERT INTO chunks_fts VALUES (?, ?, ?, ?, ?)", batch_fts)
                cursor.executemany("INSERT INTO chunk_metadata VALUES (?, ?, ?, ?, ?, ?)", batch_meta)
                conn.commit()
                total_ingested += len(batch_fts)
                batch_fts, batch_meta = [], []

        # Final batch
        if batch_fts:
            cursor.executemany("INSERT INTO chunks_fts VALUES (?, ?, ?, ?, ?)", batch_fts)
            cursor.executemany("INSERT INTO chunk_metadata VALUES (?, ?, ?, ?, ?, ?)", batch_meta)
            conn.commit()
            total_ingested += len(batch_fts)

    elapsed = time.time() - start_time
    logger.info(f"Keyword indexing complete!")
    logger.info(f"Total chunks indexed: {total_ingested}")
    logger.info(f"Time taken: {elapsed:.2f} seconds")
    
    conn.close()

if __name__ == "__main__":
    main()
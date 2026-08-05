"""
rag_pipeline.py
----------------
Loads the cleaned judgments parquet, chunks each judgment, embeds chunks
with a multilingual sentence-transformer (English + Hinglish support), and
writes them into a ChromaDB collection. The DB directory is only ever
touched in decrypted form inside `unlocked_vault()`, so on-disk it stays
AES-256 encrypted between sessions.

Run once to build the index:
    python backend/rag_pipeline.py ingest --passphrase "correct-horse-battery"

Then query:
    python backend/rag_pipeline.py query "What is the test laid down in
      Kesavananda Bharati for the basic structure doctrine?" \
      --passphrase "correct-horse-battery"
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import chromadb
import pandas as pd
from chromadb.utils import embedding_functions

try:
    # Newer langchain (1.x+) moved text splitters to their own package
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    # Older langchain (<1.0) keeps it under langchain.text_splitter
    from langchain_text_splitters import RecursiveCharacterTextSplitter

from encryptor import unlocked_vault

DATA_PATH = Path("data/processed/judgments_clean.parquet")
VAULT_ENCRYPTED = Path("vault/chroma_store.enc")
SALT_PATH = Path("vault/salt.bin")
TMP_DECRYPTED_DIR = Path(".vault_session")  # exists only while unlocked

# Multilingual model — handles English + Hinglish/Hindi transliteration
# reasonably well. Swap for BAAI/bge-m3 for higher recall at more compute cost.
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

COLLECTION_NAME = "sc_judgments"


def _clear_chromadb_cache() -> None:
    """
    ChromaDB caches its System instance at the class level, keyed by path,
    independent of any client/collection object we hold locally. That cache
    keeps the underlying sqlite/index file handles open even after we `del`
    our own references — which is what caused files in .vault_session to
    stay locked and unwipeable. This clears that internal cache so the
    handles actually get released.

    Wrapped defensively since this is a semi-internal API that could shift
    between chromadb versions — if it's missing, we just skip it rather
    than crash (the retry logic in secure_delete still provides a fallback).
    """
    try:
        chromadb.api.client.SharedSystemClient.clear_system_cache()
    except AttributeError:
        pass


def build_chunks(df: pd.DataFrame) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = []
    for _, row in df.iterrows():
        case_name = row.get("case_name") or row.get("title") or "Unknown Case"
        year = row.get("_year")
        text_chunks = splitter.split_text(row["clean_text"])
        for i, chunk in enumerate(text_chunks):
            chunks.append({
                "id": f"{row.name}-{i}",
                "text": chunk,
                "metadata": {
                    "case_name": str(case_name),
                    "year": int(year) if pd.notna(year) else None,
                    "chunk_index": i,
                },
            })
    return chunks


def ingest(passphrase: str):
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found — run scripts/download_dataset.py first."
        )
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded {len(df)} judgments. Chunking...")
    chunks = build_chunks(df)
    print(f"Produced {len(chunks)} chunks. Embedding + indexing "
        f"(this decrypts the vault for this session only)...")

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL_NAME
    )

    with unlocked_vault(VAULT_ENCRYPTED, SALT_PATH, passphrase, TMP_DECRYPTED_DIR) as db_dir:
        client = chromadb.PersistentClient(path=str(db_dir))
        collection = client.get_or_create_collection(
            COLLECTION_NAME, embedding_function=embed_fn
        )
        batch = 256
        for start in range(0, len(chunks), batch):
            sub = chunks[start:start + batch]
            collection.add(
                ids=[c["id"] for c in sub],
                documents=[c["text"] for c in sub],
                metadatas=[c["metadata"] for c in sub],
            )
            print(f"  indexed {min(start + batch, len(chunks))}/{len(chunks)}")

        # Windows keeps an open file handle on chroma.sqlite3 as long as
        # `client`/`collection` are alive, which blocks the secure-delete
        # step in unlocked_vault's cleanup. Explicitly drop the references
        # and force garbage collection *before* the `with` block exits, so
        # the handle is released and cleanup can delete the temp files.
        #
        # This alone isn't always enough: ChromaDB also keeps its own
        # internal class-level cache (SharedSystemClient) that holds a
        # reference to the underlying system/connection keyed by path,
        # independent of our local variables — so we clear that too.
        del collection, client
        _clear_chromadb_cache()
        gc.collect()

    print(f"Done. Encrypted vault written to {VAULT_ENCRYPTED}")


_embed_fn_cache = None


def get_embed_fn():
    """Cached so Streamlit (which reruns this module's functions repeatedly)
    doesn't reload the embedding model on every question."""
    global _embed_fn_cache
    if _embed_fn_cache is None:
        _embed_fn_cache = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL_NAME
        )
    return _embed_fn_cache


def query_open_db(db_dir: Path, question: str, top_k: int = 5,
                use_reranker: bool = True, candidate_pool: int = 20,
                verbose: bool = False):
    """
    Runs retrieval (+ optional re-ranking) against an ALREADY-DECRYPTED db
    directory. Use this when the vault is already open for a long-lived
    session (e.g. the Streamlit app via encryptor.open_vault) so you're not
    re-decrypting the whole multi-GB vault before every single question.

    For one-off CLI usage where the vault should open-and-close around a
    single question, use query() instead.
    """
    embed_fn = get_embed_fn()
    n_results = candidate_pool if use_reranker else top_k

    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_collection(COLLECTION_NAME, embedding_function=embed_fn)
    results = collection.query(query_texts=[question], n_results=n_results)
    del collection, client
    _clear_chromadb_cache()
    gc.collect()

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if use_reranker and documents:
        from reranker import rerank
        documents, metadatas, scores = rerank(question, documents, metadatas, top_k=top_k)
    else:
        documents = documents[:top_k]
        metadatas = metadatas[:top_k]
        scores = [1 - d for d in distances[:top_k]]

    if verbose:
        for doc, meta, score in zip(documents, metadatas, scores):
            print(f"\n[{meta['case_name']} ({meta.get('year')})] (score={score:.3f})")
            print(doc[:400] + ("..." if len(doc) > 400 else ""))

    # Re-shape to match the original chromadb-style structure so
    # model_loader.py doesn't need to change how it consumes this.
    return {
        "documents": [documents],
        "metadatas": [metadatas],
        "scores": [scores],
    }


def query(question: str, passphrase: str, top_k: int = 5, use_reranker: bool = True,
        candidate_pool: int = 20):
    """
    One-off CLI-style query: opens the vault, runs retrieval, re-locks the
    vault, and prints results. For a long-lived interactive session (many
    questions in a row), use encryptor.open_vault() once + query_open_db()
    per question instead — see app.py for that pattern.
    """
    with unlocked_vault(VAULT_ENCRYPTED, SALT_PATH, passphrase, TMP_DECRYPTED_DIR) as db_dir:
        return query_open_db(db_dir, question, top_k, use_reranker, candidate_pool,
                            verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("--passphrase", required=True)

    p_query = sub.add_parser("query")
    p_query.add_argument("question")
    p_query.add_argument("--passphrase", required=True)
    p_query.add_argument("--top-k", type=int, default=5)
    p_query.add_argument("--no-rerank", action="store_true",
                        help="Skip cross-encoder re-ranking (faster, less accurate)")

    args = parser.parse_args()
    if args.cmd == "ingest":
        ingest(args.passphrase)
    elif args.cmd == "query":
        query(args.question, args.passphrase, args.top_k, use_reranker=not args.no_rerank)
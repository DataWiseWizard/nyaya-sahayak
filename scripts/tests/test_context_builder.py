#!/usr/bin/env python3
"""
Quick test for context_builder.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.retrieval_service import HybridRetriever, RetrievalConfig
from backend.context_builder import build_context, build_messages, ContextConfig


def main():
    config = RetrievalConfig()
    retriever = HybridRetriever(config).load()

    query = "validity of preventive detention under Article 21 and Article 22"

    print(f"Retrieving for: {query}")
    authorities = retriever.retrieve(query, final_top_k=6)

    print(f"Retrieved {len(authorities)} authorities.")
    print("=" * 80)

    context = build_context(authorities, query)

    print("CONTEXT BLOCK:")
    print("=" * 80)
    print(context[:3000])
    print("...")
    print("=" * 80)
    print(f"Total context length: {len(context)} characters")

    messages = build_messages(authorities, query)
    print(f"\nMessages count: {len(messages)}")
    print(f"System prompt length: {len(messages[0]['content'])} characters")
    print(f"User message length: {len(messages[1]['content'])} characters")


if __name__ == "__main__":
    main()
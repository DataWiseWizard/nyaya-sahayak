#!/usr/bin/env python3
"""
build_quality_flags.py

Creates quality flags for each cleaned judgment document.

This script reads judgments_metadata.jsonl and writes quality_flags.jsonl.

The output will be used by the RAG ingestion pipeline to decide which
documents should be included in the first ChromaDB index.
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build quality flags for cleaned Supreme Court judgments."
    )
    parser.add_argument(
        "--metadata",
        default="data/cleaned/judgments_metadata.jsonl",
        help="Path to judgments_metadata.jsonl",
    )
    parser.add_argument(
        "--output",
        default="data/cleaned/quality_flags.jsonl",
        help="Path where quality_flags.jsonl will be written.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=500,
        help="Minimum cleaned_text_char_count required for first-index inclusion.",
    )
    args = parser.parse_args()

    metadata_path = Path(args.metadata).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not metadata_path.exists():
        raise SystemExit(f"Metadata file not found: {metadata_path}")

    included_docs = 0
    excluded_docs = 0

    no_marker_excluded = 0
    short_excluded = 0
    both_excluded = 0

    with metadata_path.open("r", encoding="utf-8") as metadata_file, output_path.open(
        "w", encoding="utf-8"
    ) as output_file:
        for line in metadata_file:
            line = line.strip()
            if not line:
                continue

            meta = json.loads(line)

            doc_id = meta.get("doc_id")
            chars = int(meta.get("cleaned_text_char_count") or 0)
            no_marker = bool(meta.get("warning_no_judgment_marker"))

            quality_flags = []

            if no_marker:
                quality_flags.append("no_judgment_marker")

            if chars < 100:
                quality_flags.append("extremely_short_lt_100")

            if chars < 500:
                quality_flags.append("very_short_lt_500")

            if chars < 1000:
                quality_flags.append("short_lt_1000")

            if chars < 2000:
                quality_flags.append("possibly_short_lt_2000")

            include = (not no_marker) and (chars >= args.min_chars)

            if include:
                included_docs += 1
            else:
                excluded_docs += 1

                if no_marker and chars < args.min_chars:
                    both_excluded += 1
                elif no_marker:
                    no_marker_excluded += 1
                elif chars < args.min_chars:
                    short_excluded += 1

            record = {
                "doc_id": doc_id,
                "include_in_first_index": include,
                "cleaned_text_char_count": chars,
                "warning_no_judgment_marker": no_marker,
                "quality_flags": quality_flags,
            }

            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "included_docs": included_docs,
        "excluded_docs": excluded_docs,
        "excluded_due_to_no_marker_only": no_marker_excluded,
        "excluded_due_to_short_text_only": short_excluded,
        "excluded_due_to_both_no_marker_and_short_text": both_excluded,
        "min_chars_required": args.min_chars,
        "output_file": str(output_path),
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
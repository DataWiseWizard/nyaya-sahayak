#!/usr/bin/env python3
"""
audit_cleaning.py

Audits judgments_metadata.jsonl produced by clean_dataset_v2.py.

It reports:
- total documents
- total chunks
- documents with no JUDGMENT/ORDER marker
- documents with very short cleaned text
- affected years
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def get_year(meta: Dict[str, Any]) -> str:
    judgment_date = meta.get("judgment_date") or ""

    if (
        isinstance(judgment_date, str)
        and len(judgment_date) >= 4
        and judgment_date[:4].isdigit()
    ):
        return judgment_date[:4]

    source_file = meta.get("source_file", "")
    match = re.search(r"_on_\d{1,2}_[A-Za-z]+_(\d{4})", source_file)

    if match:
        return match.group(1)

    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit cleaned Supreme Court judgment metadata."
    )
    parser.add_argument(
        "--metadata",
        default="data/cleaned/judgments_metadata.jsonl",
        help="Path to judgments_metadata.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="data/cleaned/audit",
        help="Folder where audit reports will be written.",
    )
    args = parser.parse_args()

    metadata_path = Path(args.metadata).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not metadata_path.exists():
        raise SystemExit(f"Metadata file not found: {metadata_path}")

    total_docs = 0
    total_chunks = 0
    no_marker_docs = 0
    zero_chunk_docs = 0

    short_counts = Counter()

    no_marker_by_year = Counter()
    short_by_year = Counter()
    total_by_year = Counter()

    no_marker_records = []
    short_records = []

    thresholds = [100, 500, 1000, 2000, 5000]

    for meta in iter_jsonl(metadata_path):
        total_docs += 1

        year = get_year(meta)
        total_by_year[year] += 1

        chunk_count = int(meta.get("chunk_count") or 0)
        cleaned_chars = int(meta.get("cleaned_text_char_count") or 0)

        total_chunks += chunk_count

        if chunk_count == 0:
            zero_chunk_docs += 1

        if meta.get("warning_no_judgment_marker"):
            no_marker_docs += 1
            no_marker_by_year[year] += 1
            no_marker_records.append(meta)

        for threshold in thresholds:
            if cleaned_chars < threshold:
                short_counts[threshold] += 1

        if cleaned_chars < 2000:
            short_by_year[year] += 1
            short_records.append(meta)

    summary = {
        "total_docs": total_docs,
        "total_chunks": total_chunks,
        "average_chunks_per_doc": round(total_chunks / total_docs, 2)
        if total_docs
        else 0,
        "no_judgment_marker_docs": no_marker_docs,
        "zero_chunk_docs": zero_chunk_docs,
        "short_docs_less_than": {str(t): short_counts[t] for t in thresholds},
        "no_marker_by_year_top_20": no_marker_by_year.most_common(20),
        "short_under_2000_by_year_top_20": short_by_year.most_common(20),
        "total_docs_by_year_top_20": total_by_year.most_common(20),
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    no_marker_path = output_dir / "no_marker_documents.jsonl"
    with no_marker_path.open("w", encoding="utf-8") as f:
        for record in sorted(no_marker_records, key=lambda x: x.get("source_file", "")):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    short_path = output_dir / "short_documents.jsonl"
    with short_path.open("w", encoding="utf-8") as f:
        for record in sorted(
            short_records,
            key=lambda x: int(x.get("cleaned_text_char_count") or 0),
        ):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Summary written to: {summary_path}")
    print(f"No-marker list written to: {no_marker_path}")
    print(f"Short-document list written to: {short_path}")


if __name__ == "__main__":
    main()
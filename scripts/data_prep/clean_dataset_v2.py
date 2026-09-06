#!/usr/bin/env python3
"""
clean_dataset_v2.py

Phase 1 data-cleaning pipeline for NyayaSahayak.

This script:
1. Extracts text from Supreme Court PDFs using pypdf.
2. Parses metadata such as case name, date, citations, bench, petitioner, respondent.
3. Removes pre-judgment material such as equivalent citations and publisher metadata.
4. Removes Indian Kanoon footers, page numbers, and repeated case-title headers.
5. Chunks cleaned judgment text using RecursiveCharacterTextSplitter.
6. Writes:
   - judgments_metadata.jsonl
   - judgment_chunks.jsonl
"""

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from pypdf import PdfReader
from tqdm.auto import tqdm

logger = logging.getLogger("clean_dataset_v2")

INDIAN_KANOON_LINE_RE = re.compile(
    r"^\s*Indian Kanoon\s*-\s*https?://.*$",
    re.I,
)

STANDALONE_PAGE_NUMBER_RE = re.compile(r"^\s*\d+\s*$")

JUDGMENT_MARKER_PATTERNS = [
    re.compile(r"^\s*J\s*U\s*D\s*G\s*M\s*E\s*N\s*T\b.*$", re.I | re.M),
    re.compile(r"^\s*JUDGMENT\s*:?\s*$", re.I | re.M),
    re.compile(r"^\s*ORDER\s*:?\s*$", re.I | re.M),
    re.compile(r"^\s*O\s*R\s*D\s*E\s*R\b.*$", re.I | re.M),
]


def normalize_whitespace(value: Optional[str]) -> str:
    """Collapse all whitespace into single spaces."""
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def compact_for_match(value: Optional[str]) -> str:
    """
    Used for matching repeated header lines.
    Removes punctuation, spaces, and lowercase differences.
    """
    if not value:
        return ""
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def clean_party_name(value: Optional[str]) -> str:
    """
    Clean petitioner/respondent names.

    Prevents case names like:
        A.K. GOPALAN Vs. vs THE STATE OF MADRAS

    Becomes:
        A.K. GOPALAN vs THE STATE OF MADRAS
    """
    if not value:
        return ""

    value = normalize_whitespace(value)

    # Remove trailing vs / vs. / versus
    value = re.sub(
        r"\s+(vs\.?|versus)\.?$",
        "",
        value,
        flags=re.I,
    ).strip()

    # Remove leading vs / vs. / versus, just in case
    value = re.sub(
        r"^(vs\.?|versus)\s+",
        "",
        value,
        flags=re.I,
    ).strip()

    # Remove leftover edge punctuation
    value = value.strip(" ,;:.")

    return value


def parse_filename_metadata(path: Path) -> Tuple[str, Optional[str]]:
    """
    Parse case parties and date from filenames like:

    A_N_Venkatesh_And_Anr_vs_State_Of_Karnataka_on_8_August_2005_1.PDF

    Returns:
        case_name_from_filename, judgment_date_iso_or_None
    """
    stem = path.stem

    # Remove trailing file-part/page markers like _1, _2, _001.
    # We intentionally do NOT remove 4-digit years.
    stem = re.sub(r"_\d{1,3}$", "", stem)

    match = re.match(
        r"(?P<parties>.+?)_on_(?P<day>\d{1,2})_(?P<month>[A-Za-z]+)_(?P<year>\d{4})$",
        stem,
        re.I,
    )

    if not match:
        fallback_name = normalize_whitespace(stem.replace("_", " "))
        return fallback_name, None

    parties = normalize_whitespace(match.group("parties").replace("_", " "))

    judgment_date_iso = None
    for fmt in ("%d %B %Y", "%d %b %Y"):
        try:
            judgment_date_iso = datetime.strptime(
                f"{match.group('day')} {match.group('month')} {match.group('year')}",
                fmt,
            ).date().isoformat()
            break
        except ValueError:
            continue

    return parties, judgment_date_iso


def extract_metadata(full_text: str) -> Dict[str, Any]:
    """
    Extract useful metadata from the first page / pre-judgment portion.

    This metadata is NOT embedded as judgment text. It is stored separately.
    """
    metadata: Dict[str, Any] = {}

    # Equivalent citations
    match = re.search(
        r"Equivalent citations:\s*(.*?)(?=Author:|Bench:|CASE NO\.|PETITIONER:|RESPONDENT:|DATE OF JUDGMENT:|JUDGMENT:|J\s*U\s*D\s*G\s*M\s*E\s*N\s*T|$)",
        full_text,
        re.I | re.S,
    )
    if match:
        metadata["citations_raw"] = normalize_whitespace(match.group(1))

    # Case number
    match = re.search(
        r"CASE NO\.:\s*(.+?)(?=PETITIONER:|RESPONDENT:|DATE OF JUDGMENT:|BENCH:|JUDGMENT:|$)",
        full_text,
        re.I | re.S,
    )
    if match:
        metadata["case_number"] = normalize_whitespace(match.group(1))

    # Judgment date
    match = re.search(
        r"DATE OF JUDGMENT:\s*(\d{1,2}/\d{1,2}/\d{4})",
        full_text,
        re.I,
    )
    if match:
        raw_date = match.group(1)
        try:
            metadata["judgment_date"] = datetime.strptime(
                raw_date,
                "%d/%m/%Y",
            ).date().isoformat()
        except ValueError:
            metadata["judgment_date_raw"] = raw_date

    # Bench
    match = re.search(r"^\s*BENCH:\s*(.+)$", full_text, re.I | re.M)
    if match:
        metadata["bench"] = normalize_whitespace(match.group(1))

    # Author
    match = re.search(r"Author:\s*([^\n]+)", full_text, re.I)
    if match:
        metadata["author"] = normalize_whitespace(match.group(1))

    # Petitioner
    match = re.search(
        r"PETITIONER:\s*(.+?)(?=\s*RESPONDENT:|\s*DATE OF JUDGMENT:|\s*BENCH:|\s*JUDGMENT:|\s*J\s*U\s*D\s*G\s*M\s*E\s*N\s*T|$)",
        full_text,
        re.I | re.S,
    )
    if match:
        metadata["petitioner"] = clean_party_name(match.group(1))

    # Respondent
    match = re.search(
        r"RESPONDENT:\s*(.+?)(?=\s*DATE OF JUDGMENT:|\s*BENCH:|\s*JUDGMENT:|\s*J\s*U\s*D\s*G\s*M\s*E\s*N\s*T|$)",
        full_text,
        re.I | re.S,
    )
    if match:
        metadata["respondent"] = clean_party_name(match.group(1))

    return metadata


def find_judgment_start(full_text: str) -> Optional[int]:
    """
    Find where the actual judgment/order starts.

    This is the key step for removing publisher headnotes/metadata.
    """
    best_start: Optional[int] = None

    for pattern in JUDGMENT_MARKER_PATTERNS:
        match = pattern.search(full_text)
        if match and (best_start is None or match.start() < best_start):
            best_start = match.start()

    return best_start


def clean_judgment_text(text: str, case_name: str) -> str:
    """
    Clean the judgment body by removing:
    - Indian Kanoon footer lines
    - standalone page numbers
    - repeated case-title header lines
    - excess blank lines
    """
    lines = text.splitlines()
    case_norm = compact_for_match(case_name)
    cleaned_lines: List[str] = []

    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            cleaned_lines.append("")
            continue

        if INDIAN_KANOON_LINE_RE.match(line):
            continue

        if STANDALONE_PAGE_NUMBER_RE.match(line):
            continue

        line_norm = compact_for_match(line)

        # Remove repeated header lines that are just the case title.
        # Require a fairly long case-name match to avoid accidental deletions.
        if len(case_norm) >= 25 and line_norm.startswith(case_norm) and len(line) < 250:
            continue

        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)

    # Compress horizontal whitespace while preserving paragraph breaks.
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def process_pdf(
    pdf_path: Path,
    splitter: RecursiveCharacterTextSplitter,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Process one PDF and return:
    - document metadata
    - chunk records
    """
    reader = PdfReader(str(pdf_path))

    extracted_pages = []
    for page in reader.pages:
        extracted_pages.append(page.extract_text() or "")

    full_text = "\n".join(extracted_pages)

    if not full_text.strip():
        logger.warning("No text extracted from %s", pdf_path.name)
        return None, []

    extracted_metadata = extract_metadata(full_text)
    filename_case_name, filename_date = parse_filename_metadata(pdf_path)

    petitioner = extracted_metadata.get("petitioner", "")
    respondent = extracted_metadata.get("respondent", "")

    if petitioner and respondent:
        case_name = normalize_whitespace(f"{petitioner} vs {respondent}")
    else:
        case_name = filename_case_name

    judgment_date = extracted_metadata.get("judgment_date") or filename_date

    start = find_judgment_start(full_text)
    warning_no_judgment_marker = start is None

    if start is None:
        logger.warning(
            "No JUDGMENT/ORDER marker found in %s. Using full text as fallback.",
            pdf_path.name,
        )
        judgment_text = full_text
    else:
        judgment_text = full_text[start:]

    judgment_text = clean_judgment_text(judgment_text, case_name)

    if len(judgment_text) < 100:
        logger.warning(
            "Cleaned judgment text too short for %s. Possible extraction problem.",
            pdf_path.name,
        )

    chunks = splitter.split_text(judgment_text)

    doc_id = re.sub(r"\W+", "_", pdf_path.stem).strip("_")

    doc_metadata: Dict[str, Any] = {
        "doc_id": doc_id,
        "source_file": pdf_path.name,
        "court": "Supreme Court of India",
        "case_name": case_name,
        "judgment_date": judgment_date,
        "citations_raw": extracted_metadata.get("citations_raw"),
        "case_number": extracted_metadata.get("case_number"),
        "bench": extracted_metadata.get("bench"),
        "author": extracted_metadata.get("author"),
        "petitioner": petitioner,
        "respondent": respondent,
        "page_count": len(reader.pages),
        "cleaned_text_char_count": len(judgment_text),
        "chunk_count": len(chunks),
        "extractor": "pypdf",
        "warning_no_judgment_marker": warning_no_judgment_marker,
        "cleaned_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    chunk_records: List[Dict[str, Any]] = []

    for idx, chunk in enumerate(chunks):
        chunk_text = normalize_whitespace(chunk)

        if not chunk_text:
            continue

        chunk_records.append(
            {
                "chunk_id": f"{doc_id}_chunk_{idx:04d}",
                "doc_id": doc_id,
                "source_file": pdf_path.name,
                "case_name": case_name,
                "judgment_date": judgment_date,
                "chunk_index": idx,
                "text": chunk_text,
            }
        )

    return doc_metadata, chunk_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean Indian Supreme Court PDFs for NyayaSahayak."
    )
    parser.add_argument(
        "--pdf-dir",
        default="data/raw",
        help="Folder containing PDF files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/cleaned",
        help="Folder for cleaned JSONL files.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size in characters.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=150,
        help="Chunk overlap in characters.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N PDFs. Useful for testing.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    pdf_dir = Path(args.pdf_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_dir.exists():
        raise SystemExit(f"PDF directory does not exist: {pdf_dir}")

    pdf_files = sorted(
        {
            path
            for path in pdf_dir.rglob("*")
            if path.is_file() and path.suffix.lower() == ".pdf"
        }
    )

    if args.limit:
        pdf_files = pdf_files[: args.limit]

    if not pdf_files:
        raise SystemExit(f"No PDF files found in {pdf_dir}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    metadata_path = output_dir / "judgments_metadata.jsonl"
    chunks_path = output_dir / "judgment_chunks.jsonl"

    processed_docs = 0
    written_chunks = 0
    failed_or_empty = 0

    with metadata_path.open("w", encoding="utf-8") as metadata_file, chunks_path.open(
        "w", encoding="utf-8"
    ) as chunks_file:
        for pdf_path in tqdm(pdf_files, desc="Cleaning Supreme Court PDFs"):
            try:
                doc_metadata, chunk_records = process_pdf(pdf_path, splitter)
            except Exception as exc:
                logger.warning("Failed to process %s: %s", pdf_path.name, exc)
                failed_or_empty += 1
                continue

            if not doc_metadata or not chunk_records:
                failed_or_empty += 1
                continue

            metadata_file.write(json.dumps(doc_metadata, ensure_ascii=False) + "\n")

            for chunk_record in chunk_records:
                chunks_file.write(json.dumps(chunk_record, ensure_ascii=False) + "\n")

            processed_docs += 1
            written_chunks += len(chunk_records)

    logger.info("Done.")
    logger.info("Documents written: %s", processed_docs)
    logger.info("Chunks written: %s", written_chunks)
    logger.info("Failed/empty documents: %s", failed_or_empty)
    logger.info("Metadata file: %s", metadata_path)
    logger.info("Chunks file: %s", chunks_path)


if __name__ == "__main__":
    main()
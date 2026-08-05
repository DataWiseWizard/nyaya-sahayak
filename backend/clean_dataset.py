"""
clean_dataset.py
----------------
The Kaggle "Indian Supreme Court Judgments" dataset (this variant) is NOT a
single CSV — it's a folder of PDFs organized by year:

    data/raw/supreme_court_judgments/
        1950/
            Some_Party_vs_Other_Party_on_12_March_1950_1.PDF
            ...
        2020/
            Abhilasha_vs_Parkash_on_15_September_2020_1.PDF
            ...

The case name and decision date are encoded in the filename itself
(Party1_vs_Party2_on_DD_Month_YYYY_N), so we parse that instead of relying
on CSV metadata columns.

This script walks the year folders, filters to years >= --min-year,
extracts text from each PDF, cleans it, and writes one row per judgment to
a parquet file — matching the same output schema the rest of the pipeline
(rag_pipeline.py) already expects: case_name, _year, clean_text.

Usage:
    python scripts/clean_dataset.py \
        --raw-dir data/raw/supreme_court_judgments \
        --clean-out data/processed/judgments_clean.parquet
"""

import argparse
import re
from pathlib import Path

import pandas as pd
from pypdf import PdfReader
from tqdm import tqdm

# --- Filename parsing ---------------------------------------------------------

# e.g. "Abhilasha_vs_Parkash_on_15_September_2020_1" (extension stripped)
FILENAME_RE = re.compile(
    r"^(?P<party1>.+?)_vs_(?P<party2>.+?)_on_(?P<day>\d{1,2})_(?P<month>[A-Za-z]+)_(?P<year>\d{4})(?:_\d+)?$"
)


def parse_filename(stem: str) -> dict:
    """Extracts case name + date from the filename stem. Falls back
    gracefully (year=None) if a file doesn't match the expected pattern —
    a handful of odd filenames are normal in scraped datasets."""
    m = FILENAME_RE.match(stem)
    if not m:
        return {"case_name": stem.replace("_", " "), "date": None, "year": None}
    case_name = f"{m.group('party1').replace('_', ' ')} vs {m.group('party2').replace('_', ' ')}"
    date_str = f"{m.group('day')} {m.group('month')} {m.group('year')}"
    return {"case_name": case_name, "date": date_str, "year": int(m.group("year"))}


# --- Text cleaning -----------------------------------------------------------

PAGE_NUMBER_RE = re.compile(r"\n\s*Page\s*\d+\s*(of\s*\d+)?\s*\n", re.IGNORECASE)
HEADER_FOOTER_RE = re.compile(
    r"(SUPREME COURT OF INDIA\s*\n)|(\bNon-?Reportable\b)|(\bReportable\b)",
    re.IGNORECASE,
)
MULTI_BLANK_RE = re.compile(r"\n{3,}")
CITATION_DUP_RE = re.compile(r"(\(\d{4}\)\s*\d+\s*SCC\s*\d+)(\s*\1)+")


def clean_judgment_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = PAGE_NUMBER_RE.sub("\n", text)
    text = HEADER_FOOTER_RE.sub("", text)
    text = CITATION_DUP_RE.sub(r"\1", text)
    text = MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


def extract_pdf_text(pdf_path: Path) -> str | None:
    """Returns None (rather than raising) on corrupt/unreadable PDFs so one
    bad file doesn't kill the whole ingestion run."""
    try:
        reader = PdfReader(str(pdf_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    except Exception as e:  # noqa: BLE001 - deliberately broad, see docstring
        print(f"  [WARN] Failed to read {pdf_path.name}: {e}")
        return None


# --- Main pipeline -------------------------------------------------------------

def collect_year_folders(raw_dir: Path, min_year: int) -> list[Path]:
    folders = []
    for entry in sorted(raw_dir.iterdir()):
        if entry.is_dir() and entry.name.isdigit() and int(entry.name) >= min_year:
            folders.append(entry)
    return folders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw/supreme_court_judgments",
                        help="Path to the folder containing year subfolders "
                            "(the one with 1950/, 1951/, ..., 2025/ inside)")
    parser.add_argument("--clean-out", default="data/processed/judgments_clean.parquet")
    parser.add_argument("--min-year", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on number of PDFs to process, "
                            "useful for a quick test run first")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"{raw_dir} not found. Point --raw-dir at the folder that "
            f"directly contains the year subfolders (e.g. .../supreme_court_judgments)"
        )

    year_folders = collect_year_folders(raw_dir, args.min_year)
    if not year_folders:
        raise FileNotFoundError(
            f"No year subfolders >= {args.min_year} found under {raw_dir}"
        )

    pdf_paths = []
    for yf in year_folders:
        pdf_paths.extend(sorted(yf.glob("*.PDF")) + sorted(yf.glob("*.pdf")))
    # de-dupe in case both globs matched the same files on a case-insensitive FS
    pdf_paths = sorted(set(pdf_paths))

    if args.limit:
        pdf_paths = pdf_paths[:args.limit]

    print(f"Found {len(pdf_paths)} PDFs across {len(year_folders)} year folders "
        f"(>= {args.min_year}). Extracting + cleaning...")

    rows = []
    for pdf_path in tqdm(pdf_paths):
        meta = parse_filename(pdf_path.stem)
        raw_text = extract_pdf_text(pdf_path)
        if not raw_text:
            continue
        clean_text = clean_judgment_text(raw_text)
        if len(clean_text) < 200:
            continue  # likely a scan-only/blank PDF, skip
        rows.append({
            "case_name": meta["case_name"],
            "date": meta["date"],
            "_year": meta["year"] or int(pdf_path.parent.name),  # folder name as fallback
            "source_file": str(pdf_path),
            "clean_text": clean_text,
        })

    df = pd.DataFrame(rows).drop_duplicates(subset=["clean_text"])

    out_path = Path(args.clean_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} cleaned judgments -> {out_path}")


if __name__ == "__main__":
    main()
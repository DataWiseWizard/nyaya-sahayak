"""
prepare_demo_bundle.py

Extracts only the judgment PDFs referenced in the evaluation dataset
and known edge-case tests. Creates a lightweight demo bundle (~20 MB)
so interviewers/testers can use the PDF download feature without
downloading the full 20 GB corpus.
"""

import json
import shutil
from pathlib import Path

# === CONFIGURATION ===
PROJECT_ROOT = Path(__file__).parent.parent
EVAL_DATASET = PROJECT_ROOT / "scripts" / "tests" / "eval_dataset.jsonl"
METADATA_FILE = PROJECT_ROOT / "data" / "cleaned" / "judgments_metadata.jsonl"
RAW_PDF_DIR = PROJECT_ROOT / "docs" / "judgments"
OUTPUT_DIR = PROJECT_ROOT / "demo_pdfs"

# Known edge-case source files (add any others you want in the demo)
EDGE_CASE_FILES = [
    "A_K_Gopalan_vs_The_State_Of_Madras_Union_Of_India__on_19_May_1950_1.PDF",
    "1_R_Muthammal_Died_2_Parameswari_vs_Sri_Subramaniaswami_on_14_January_1960_1.PDF",
    "A_N_Venkatesh_And_Anr_vs_State_Of_Karnataka_on_8_August_2005_1.PDF",
]


def load_eval_expected_cases() -> set:
    """Load expected case names from eval dataset."""
    cases = set()
    if not EVAL_DATASET.exists():
        print(f"WARNING: Eval dataset not found at {EVAL_DATASET}")
        return cases
    with EVAL_DATASET.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            cases.add(record.get("expected_case", ""))
    return cases


def build_doc_id_to_source_map() -> dict:
    """Build mapping from doc_id -> source_file from metadata."""
    mapping = {}
    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_FILE}")
    with METADATA_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            doc_id = record.get("doc_id", "")
            source_file = record.get("source_file", "")
            case_name = record.get("case_name", "")
            if doc_id and source_file:
                mapping[doc_id] = {"source_file": source_file, "case_name": case_name}
    return mapping


def find_source_files_for_cases(expected_cases: set, doc_map: dict) -> set:
    """Match expected case names to source files via fuzzy matching."""
    matched_files = set()
    for case in expected_cases:
        if not case:
            continue
        case_lower = case.lower().strip()
        # Normalize for matching
        case_tokens = set(case_lower.replace("vs", " ").replace("v.", " ").split())
        case_tokens = {t for t in case_tokens if len(t) > 2}

        best_match = None
        best_score = 0

        for doc_id, info in doc_map.items():
            cn = info["case_name"].lower()
            cn_tokens = set(cn.replace("vs", " ").replace("v.", " ").split())
            cn_tokens = {t for t in cn_tokens if len(t) > 2}

            if not case_tokens or not cn_tokens:
                continue

            overlap = len(case_tokens & cn_tokens)
            score = overlap / max(len(case_tokens), 1)

            if score > best_score:
                best_score = score
                best_match = info["source_file"]

        if best_match and best_score >= 0.4:
            matched_files.add(best_match)
        else:
            print(f"  WARNING: Could not match case '{case}' to any source file")

    return matched_files


def main():
    print("=" * 60)
    print("NyayaSahayak Demo Bundle Generator")
    print("=" * 60)

    # 1. Load eval dataset expected cases
    print("\n[1/4] Loading evaluation dataset...")
    expected_cases = load_eval_expected_cases()
    print(f"  Found {len(expected_cases)} expected cases in eval dataset")

    # 2. Build doc_id -> source_file mapping
    print("\n[2/4] Building metadata mapping...")
    doc_map = build_doc_id_to_source_map()
    print(f"  Mapped {len(doc_map)} documents")

    # 3. Find source files for eval cases
    print("\n[3/4] Matching eval cases to source files...")
    eval_files = find_source_files_for_cases(expected_cases, doc_map)
    print(f"  Matched {len(eval_files)} source files from eval dataset")

    # 4. Combine with edge-case files
    all_files = eval_files | set(EDGE_CASE_FILES)
    print(f"  Added {len(EDGE_CASE_FILES)} edge-case files")
    print(f"  Total unique PDFs for demo bundle: {len(all_files)}")

    # 5. Copy files
    print(f"\n[4/4] Copying PDFs to {OUTPUT_DIR}...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = 0
    total_size = 0

    for filename in sorted(all_files):
        src = RAW_PDF_DIR / filename
        dst = OUTPUT_DIR / filename

        if src.exists():
            shutil.copy2(src, dst)
            size = dst.stat().st_size
            total_size += size
            copied += 1
        else:
            # Try case-insensitive search
            found = False
            if RAW_PDF_DIR.exists():
                for f in RAW_PDF_DIR.iterdir():
                    if f.name.lower() == filename.lower():
                        shutil.copy2(f, dst)
                        size = dst.stat().st_size
                        total_size += size
                        copied += 1
                        found = True
                        break
            if not found:
                print(f"  MISSING: {filename}")
                missing += 1

    print("\n" + "=" * 60)
    print(f"DONE!")
    print(f"  Copied:   {copied} PDFs")
    print(f"  Missing:  {missing} PDFs")
    print(f"  Total size: {total_size / (1024 * 1024):.1f} MB")
    print(f"  Output:   {OUTPUT_DIR}")
    print("=" * 60)

    if missing > 0:
        print("\n⚠️  Some PDFs were not found. Check your raw PDF directory.")
    else:
        print("\n✅ All demo PDFs ready! Zip this folder for distribution:")
        print(f"   Compress-Archive -Path '{OUTPUT_DIR}\\*' -DestinationPath 'nyaya-sahayak-demo-pdfs.zip'")


if __name__ == "__main__":
    main()
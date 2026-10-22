#!/usr/bin/env python3
"""Generate project directory structure as a text file."""

import os
from pathlib import Path

IGNORE = {
    "__pycache__", ".git", "venv", "vault", "node_modules",
    ".pytest_cache", "*.pyc", ".mypy_cache", "docs", "judgments", "raw", "demo_pdfs"
}

def should_ignore(name: str) -> bool:
    return any(
        name == pattern or name.endswith(pattern.lstrip("*"))
        for pattern in IGNORE
    )

def build_tree(root: Path, prefix: str = "") -> list[str]:
    lines = []
    entries = sorted(
        [e for e in root.iterdir() if not should_ignore(e.name)],
        key=lambda e: (not e.is_dir(), e.name.lower())
    )
    for i, entry in enumerate(entries):
        connector = "└── " if i == len(entries) - 1 else "├── "
        lines.append(f"{prefix}{connector}{entry.name}")
        if entry.is_dir():
            extension = "    " if i == len(entries) - 1 else "│   "
            lines.extend(build_tree(entry, prefix + extension))
    return lines

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    output = ["📦 nyaya-sahayak/", *build_tree(project_root)]
    result = "\n".join(output)

    out_path = project_root / "project_structure.txt"
    out_path.write_text(result, encoding="utf-8")
    print(result)
    print(f"\n✅ Saved to: {out_path}")
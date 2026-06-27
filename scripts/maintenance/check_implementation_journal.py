# © Artur Czarnecki. All rights reserved.
"""Validate implementation journal INDEX ↔ entries consistency."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JOURNAL = ROOT / "docs" / "implementation-journal"
INDEX = JOURNAL / "INDEX.md"
ENTRIES = JOURNAL / "entries"

REQUIRED_SECTIONS = (
    "## Operator request",
    "## Summary",
    "## Project impact",
    "## Traceability",
    "## Changed artifacts",
    "## Verification",
    "## Risks and follow-ups",
)
ID_RE = re.compile(r"^IJ-\d{4}-\d{2}-\d{2}-\d{3}$")
DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}/")
ROW_RE = re.compile(
    r"^\|\s*(IJ-\d{4}-\d{2}-\d{2}-\d{3})\s*\|"
    r"[^|]*\|[^|]*\|[^|]*\|"
    r"\s*\[([^\]]+)\]\(entries/([^)]+)\)\s*\|",
)


def _frontmatter_id(text: str) -> str | None:
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    if end == -1:
        return None
    block = text[3:end]
    for line in block.splitlines():
        if line.startswith("id:"):
            return line.split(":", 1)[1].strip()
    return None


def main() -> int:
    errors: list[str] = []

    if not INDEX.is_file():
        errors.append("missing docs/implementation-journal/INDEX.md")
        _report(errors)
        return 1

    index_text = INDEX.read_text(encoding="utf-8")
    rows: list[tuple[str, str, str]] = []
    for line in index_text.splitlines():
        m = ROW_RE.match(line.strip())
        if m:
            rows.append((m.group(1), m.group(2), m.group(3)))

    if not rows:
        errors.append("INDEX.md has no journal rows")

    seen_ids: set[str] = set()
    for entry_id, _title, rel_path in rows:
        if entry_id in seen_ids:
            errors.append(f"duplicate INDEX id: {entry_id}")
        seen_ids.add(entry_id)
        if not ID_RE.match(entry_id):
            errors.append(f"invalid id format: {entry_id}")
        if not DATE_DIR_RE.match(rel_path):
            errors.append(
                "INDEX link must be entries/YYYY-MM-DD/<slug>.md "
                f"(not flat or date-prefixed filename): entries/{rel_path}"
            )

        entry_path = ENTRIES / rel_path
        if not entry_path.is_file():
            errors.append(f"INDEX links missing file: entries/{rel_path}")
            continue

        text = entry_path.read_text(encoding="utf-8")
        fm_id = _frontmatter_id(text)
        if fm_id != entry_id:
            errors.append(
                f"frontmatter id mismatch in entries/{rel_path}: {fm_id!r} != {entry_id!r}"
            )
        for section in REQUIRED_SECTIONS:
            if section not in text:
                errors.append(f"entries/{rel_path} missing section {section}")

    indexed_files = {rel for _, _, rel in rows}
    for path in sorted(ENTRIES.rglob("*.md")):
        if path.name.startswith("_"):
            continue
        rel = path.relative_to(ENTRIES).as_posix()
        if "/" not in rel:
            errors.append(
                "journal entry must live under entries/YYYY-MM-DD/ "
                f"(only _TEMPLATE.md may sit in entries/ root): entries/{rel}"
            )
            continue
        if not DATE_DIR_RE.match(rel):
            errors.append(
                f"journal entry path must start with YYYY-MM-DD/: entries/{rel}"
            )
        if rel not in indexed_files:
            errors.append(f"entry not in INDEX: entries/{rel}")

    _report(errors)
    return 1 if errors else 0


def _report(errors: list[str]) -> None:
    if errors:
        print("check_implementation_journal: FAIL")
        for err in errors:
            print(f"  - {err}")
    else:
        print("check_implementation_journal: OK")


if __name__ == "__main__":
    sys.exit(main())

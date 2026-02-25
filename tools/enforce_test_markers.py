# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from pathlib import Path
from typing import List, Optional


ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = ROOT / "tests"

MARKER_MAP = {
    "unit": "unit",
    "integration": "integration",
    "e2e": "e2e",
}


def detect_category(path: Path) -> Optional[str]:
    parts = path.relative_to(TESTS_DIR).parts
    if not parts:
        return None
    return MARKER_MAP.get(parts[0])


def has_pytestmark(content: str, marker: str) -> bool:
    return f"pytestmark = pytest.mark.{marker}" in content


def has_pytest_import(content: str) -> bool:
    return "import pytest" in content


def find_insert_position(lines: List[str]) -> int:
    """
    Returns index AFTER the import section.
    Safely handles:
    - copyright header
    - module docstring
    - __future__ imports
    - multiline imports with ()
    """

    index = 0
    total = len(lines)

    # Skip initial comments
    while index < total and lines[index].strip().startswith("#"):
        index += 1

    # Skip empty lines
    while index < total and lines[index].strip() == "":
        index += 1

    # Skip module docstring
    if index < total and lines[index].strip().startswith(('"""', "'''")):
        quote = lines[index].strip()[:3]
        index += 1
        while index < total and quote not in lines[index]:
            index += 1
        if index < total:
            index += 1

    # Skip __future__ imports
    while index < total and lines[index].startswith("from __future__"):
        index += 1

    # Skip remaining import section
    while index < total:
        stripped = lines[index].strip()

        if stripped.startswith("import ") or stripped.startswith("from "):
            index += 1

            # handle multiline import (...)
            if stripped.endswith("("):
                while index < total and ")" not in lines[index]:
                    index += 1
                if index < total:
                    index += 1
            continue

        break

    return index


def process_file(path: Path) -> None:
    content = path.read_text(encoding="utf-8")
    category = detect_category(path)

    if not category:
        return

    lines = content.splitlines()
    updated = False

    insert_pos = find_insert_position(lines)

    # Ensure pytest import
    if not has_pytest_import(content):
        lines.insert(insert_pos, "import pytest")
        insert_pos += 1
        updated = True

    # Ensure pytestmark
    if not has_pytestmark(content, category):
        lines.insert(insert_pos, "")
        lines.insert(insert_pos + 1, f"pytestmark = pytest.mark.{category}")
        updated = True

    if updated:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[UPDATED] {path.relative_to(ROOT)}")


def main() -> None:
    for path in TESTS_DIR.rglob("*.py"):
        process_file(path)


if __name__ == "__main__":
    main()
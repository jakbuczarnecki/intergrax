# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = ROOT / "tests"

MARKER_MAP = {
    "unit": "unit",
    "integration": "integration",
    "e2e": "e2e",
}


def detect_category(path: Path) -> str | None:
    parts = path.relative_to(TESTS_DIR).parts
    if not parts:
        return None
    return MARKER_MAP.get(parts[0])


def has_pytestmark(content: str, marker: str) -> bool:
    return f"pytestmark = pytest.mark.{marker}" in content


def has_pytest_import(content: str) -> bool:
    return "import pytest" in content


def process_file(path: Path) -> None:
    content = path.read_text(encoding="utf-8")

    category = detect_category(path)
    if not category:
        return

    updated = False
    lines = content.splitlines()

    if not has_pytest_import(content):
        insert_index = 0
        for i, line in enumerate(lines):
            if line.startswith("import ") or line.startswith("from "):
                insert_index = i
                break
        lines.insert(insert_index, "import pytest")
        updated = True

    if not has_pytestmark(content, category):
        lines.append("")
        lines.append(f"pytestmark = pytest.mark.{category}")
        updated = True

    if updated:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[UPDATED] {path.relative_to(ROOT)}")


def main() -> None:
    for path in TESTS_DIR.rglob("*.py"):
        process_file(path)


if __name__ == "__main__":
    main()
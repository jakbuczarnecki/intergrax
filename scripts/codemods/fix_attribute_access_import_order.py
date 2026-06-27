#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Move attribute_access import below __future__ and module docstring."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
IMPORT_LINE = "from intergrax.utils import attribute_access\n"


def fix_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if IMPORT_LINE.strip() not in text:
        return False
    lines = text.splitlines(keepends=True)
    import_indices = [i for i, line in enumerate(lines) if line == IMPORT_LINE]
    if not import_indices:
        return False
    for index in reversed(import_indices):
        lines.pop(index)
    insert_at = 0
    for index, line in enumerate(lines):
        if line.startswith("from __future__"):
            insert_at = index + 1
    if insert_at == 0:
        for index, line in enumerate(lines):
            if line.startswith("from ") or line.startswith("import "):
                insert_at = index
                break
    lines.insert(insert_at, IMPORT_LINE if insert_at == len(lines) or lines[insert_at - 1].endswith("\n") else IMPORT_LINE)
    new_text = "".join(lines)
    if new_text == text:
        return False
    path.write_text(new_text, encoding="utf-8")
    return True


def main() -> int:
    changed = 0
    for root_name in ("intergrax", "agents", "tests", "scripts", "testing_support", "tools", ".github"):
        root = REPO / root_name
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            if fix_file(path):
                changed += 1
    print(f"Reordered imports in {changed} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

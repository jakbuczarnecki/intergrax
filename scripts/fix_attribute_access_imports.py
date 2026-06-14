#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Ensure files using attribute_access.optional import attribute_access."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
IMPORT = "from intergrax.utils import attribute_access\n"


def main() -> int:
    changed = 0
    for root_name in ("intergrax", "agents", "tests", "scripts", "testing_support"):
        root = REPO / root_name
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            rel = path.relative_to(REPO).as_posix()
            if rel == "intergrax/utils/attribute_access.py":
                continue
            text = path.read_text(encoding="utf-8")
            if "attribute_access.optional" not in text and "attribute_access.optional_bool" not in text:
                continue
            if "import attribute_access" in text:
                continue
            lines = text.splitlines(keepends=True)
            insert_at = 0
            for index, line in enumerate(lines):
                if line.startswith("from ") or line.startswith("import "):
                    insert_at = index
                    break
            lines.insert(insert_at, IMPORT)
            path.write_text("".join(lines), encoding="utf-8")
            changed += 1
            print(rel)
    print(f"Fixed imports in {changed} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

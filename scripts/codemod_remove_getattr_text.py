#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Text-level replacement: getattr( -> attribute_access.optional(, setattr( -> object.__setattr__("""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SKIP = frozenset(
    {
        "intergrax/utils/attribute_access.py",
        "intergrax/utils/lazy_export.py",
        "scripts/codemod_remove_getattr.py",
        "scripts/codemod_remove_getattr_text.py",
        "scripts/patch_lazy_bundle_exports.py",
        "tools/ast_audit.py",
    }
)
IMPORT = "from intergrax.utils import attribute_access\n"
GETATTR = re.compile(r"\bgetattr\s*\(")
SETATTR = re.compile(r"\bsetattr\s*\(")


def patch_file(path: Path) -> bool:
    path = path.resolve()
    rel = path.relative_to(REPO).as_posix()
    if rel in SKIP:
        return False
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    changed = False
    new_lines: list[str] = []
    needs_import = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("#"):
            new_lines.append(line)
            continue
        if GETATTR.search(line):
            line = GETATTR.sub("attribute_access.optional(", line)
            needs_import = True
            changed = True
        if SETATTR.search(line) and "monkeypatch.setattr" not in line:
            line = SETATTR.sub("object.__setattr__(", line)
            changed = True
        new_lines.append(line)
    if not changed:
        return False
    text = "".join(new_lines)
    if needs_import and "attribute_access" not in text:
        insert_at = 0
        for index, line in enumerate(new_lines):
            if line.startswith("from ") or line.startswith("import "):
                insert_at = index
                break
        new_lines.insert(insert_at, IMPORT)
        text = "".join(new_lines)
    path.write_text(text, encoding="utf-8")
    return True


def main() -> int:
    roots = [Path(arg) for arg in sys.argv[1:]] if len(sys.argv) > 1 else [
        REPO / "intergrax",
        REPO / "agents",
        REPO / "tests",
        REPO / "scripts",
        REPO / "testing_support",
    ]
    changed = 0
    for root in roots:
        root = root.resolve()
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            if patch_file(path):
                changed += 1
                print(path.relative_to(REPO).as_posix())
    print(f"Patched {changed} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

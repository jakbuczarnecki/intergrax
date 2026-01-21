# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from typing import Iterable, Set, Tuple

ROOT = Path(".").resolve()

SCAN_DIRS = ["intergrax", "api", "applications", "mcp"]
PY_EXT = ".py"


def iter_py_files(base: Path) -> Iterable[Path]:
    for d in SCAN_DIRS:
        p = base / d
        if not p.exists() or not p.is_dir():
            continue
        for root, _, files in os.walk(p):
            for fn in files:
                if fn.endswith(PY_EXT):
                    yield Path(root) / fn


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_imports(src: str) -> Set[str]:
    out: Set[str] = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return out

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    out.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.add(node.module.split(".")[0])
    return out


def guess_local_top_modules(base: Path) -> Set[str]:
    local: Set[str] = set()
    for d in SCAN_DIRS:
        p = base / d
        if p.exists() and p.is_dir():
            local.add(d.replace("-", "_"))
    return local


def main() -> None:
    stdlib = getattr(sys, "stdlib_module_names", set())
    # Fallback for older versions (but you are on 3.12 so stdlib_module_names exists).
    stdlib = set(stdlib)

    local = guess_local_top_modules(ROOT)

    all_imports: Set[str] = set()
    per_file_errors: Set[Tuple[str, str]] = set()

    for path in iter_py_files(ROOT):
        src = read_text(path)
        try:
            imps = extract_imports(src)
            all_imports |= imps
        except Exception as e:
            per_file_errors.add((str(path), repr(e)))

    # Filter obvious noise
    all_imports.discard("__future__")
    all_imports.discard("typing")
    all_imports.discard("dataclasses")

    third_party = sorted(
        m for m in all_imports
        if m and (m not in stdlib) and (m not in local)
    )

    print("=== Local top-level modules detected ===")
    print(", ".join(sorted(local)) if local else "(none)")
    print()

    print("=== Third-party top-level imports (candidate dependencies) ===")
    for m in third_party:
        print(m)

    if per_file_errors:
        print()
        print("=== Parse/extract errors (non-fatal) ===")
        for p, err in sorted(per_file_errors):
            print(f"{p}: {err}")


if __name__ == "__main__":
    main()

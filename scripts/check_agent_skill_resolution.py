#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Audit Tier-2 agent contracts for skill-based tool resolution (Phase AS-3)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

SCAN_ROOT = "agents"
SKIP_PARTS = frozenset({"tests", "__pycache__"})

# Authors must not pre-populate allowed_tools; registry resolves from skills/extra_tools.
FORBIDDEN_ALLOWED_TOOLS = re.compile(
    r"allowed_tools\s*=\s*\[(?!\s*\])",
)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    root = repo_root / SCAN_ROOT
    if not root.is_dir():
        print(f"missing scan root: {SCAN_ROOT}")
        return 1

    violations: list[str] = []
    for path in root.rglob("*.py"):
        if SKIP_PARTS.intersection(path.parts):
            continue
        rel = path.relative_to(repo_root).as_posix()
        text = path.read_text(encoding="utf-8")
        if FORBIDDEN_ALLOWED_TOOLS.search(text):
            violations.append(f"{rel}: author-time allowed_tools list is forbidden")

    if violations:
        print("agent skill resolution audit failed:")
        for item in sorted(violations):
            print(f"  - {item}")
        return 1

    print("agent skill resolution audit: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

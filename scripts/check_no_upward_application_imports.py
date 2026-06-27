#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Fail when lower-layer packages import Tier-3 application composition."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SCAN_ROOTS = (
    REPO_ROOT / "agents",
    REPO_ROOT / "intergrax" / "agents",
    REPO_ROOT / "intergrax" / "runtime",
    REPO_ROOT / "intergrax" / "contracts",
    REPO_ROOT / "intergrax" / "rag",
    REPO_ROOT / "intergrax" / "tools",
    REPO_ROOT / "intergrax" / "skills",
    REPO_ROOT / "intergrax" / "llm_adapters",
)

FORBIDDEN = re.compile(
    r"^\s*(?:from|import)\s+(?:intergrax\.applications|applications)\b",
    re.MULTILINE,
)


def main() -> int:
    violations: list[str] = []
    for scan_root in SCAN_ROOTS:
        if not scan_root.is_dir():
            continue
        for path in scan_root.rglob("*.py"):
            if "tests" in path.parts:
                continue
            rel = path.relative_to(REPO_ROOT).as_posix()
            if FORBIDDEN.search(path.read_text(encoding="utf-8")):
                violations.append(rel)
    if violations:
        print("Lower layers must not import applications/ or intergrax.applications:")
        for item in sorted(set(violations)):
            print(f"  - {item}")
        return 1
    print("no upward application imports: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

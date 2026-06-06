#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Fail when Tier-0 intergrax imports Tier-3 applications (FAUDIT-TIER.2)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOT = REPO_ROOT / "intergrax"

FORBIDDEN = re.compile(
    r"^\s*(?:from|import)\s+applications\b",
    re.MULTILINE,
)

GRANDFATHER: frozenset[str] = frozenset()


def main() -> int:
    violations: list[str] = []
    if not SCAN_ROOT.is_dir():
        print("intergrax/: missing")
        return 1
    for path in SCAN_ROOT.rglob("*.py"):
        if "tests" in path.parts:
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in GRANDFATHER:
            continue
        if FORBIDDEN.search(path.read_text(encoding="utf-8")):
            violations.append(rel)
    if violations:
        print("Tier-0 intergrax must not import applications/:")
        for item in sorted(violations):
            print(f"  - {item}")
        return 1
    print("intergrax tier-3 applications import audit: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

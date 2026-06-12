#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Fail when Tier-0 ``intergrax/context`` imports Tier-2 agents or Tier-3 applications (CE-1.6)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOT = REPO_ROOT / "intergrax" / "context"

FORBIDDEN = (
    re.compile(r"^\s*(?:from|import)\s+agents\b", re.MULTILINE),
    re.compile(r"^\s*(?:from|import)\s+applications\b", re.MULTILINE),
)


def main() -> int:
    if not SCAN_ROOT.is_dir():
        print("intergrax/context/: missing")
        return 1

    violations: list[str] = []
    for path in SCAN_ROOT.rglob("*.py"):
        rel = path.relative_to(REPO_ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        for pattern in FORBIDDEN:
            if pattern.search(text):
                violations.append(rel)
                break

    if violations:
        print("intergrax/context must not import agents/ or applications/:")
        for item in sorted(set(violations)):
            print(f"  - {item}")
        return 1

    print("context tier-0 import boundary: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

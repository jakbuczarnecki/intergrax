#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Fail when new getattr/setattr appear under harness paths (Phase Q+.0.3)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

SCAN_ROOTS = (
    "intergrax/runtime/nexus",
    "intergrax/agents",
    "agents",
)

# Grandfather list empty — all harness paths must stay free of getattr/setattr.
GRANDFATHER: frozenset[str] = frozenset(
    {
        # PEP 562 lazy exports for ``AgentEngine`` / UAEP symbols.
        "intergrax/agents/__init__.py",
    }
)

PATTERN = re.compile(r"\b(getattr|setattr)\s*\(")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    violations: list[str] = []
    for root_name in SCAN_ROOTS:
        root = repo_root / root_name
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if "tests" in path.parts:
                continue
            rel = path.relative_to(repo_root).as_posix()
            if rel in GRANDFATHER:
                continue
            for line_no, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if PATTERN.search(stripped):
                    violations.append(f"{rel}:{line_no}: {stripped}")
    if violations:
        print("getattr/setattr in harness paths (not grandfathered):")
        print("\n".join(violations))
        return 1
    print("harness getattr audit: OK (zero grandfathered paths)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

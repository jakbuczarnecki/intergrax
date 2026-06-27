#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Fail when top-level agents/ packages look like pseudo-agent shared helpers."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENTS_ROOT = REPO_ROOT / "agents"

_FORBIDDEN_SUFFIXES = ("_shared", "_common", "_helpers")
_ALLOWLIST: frozenset[str] = frozenset()


def main() -> int:
    if not AGENTS_ROOT.is_dir():
        print("agents/: missing", file=sys.stderr)
        return 1

    violations: list[str] = []
    for entry in sorted(AGENTS_ROOT.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        if name in _ALLOWLIST:
            continue
        if any(name.endswith(suffix) for suffix in _FORBIDDEN_SUFFIXES):
            violations.append(name)

    if violations:
        print("agents/ must not contain pseudo-agent shared packages:")
        for name in violations:
            print(f"  - agents/{name}/")
        return 1

    print("agents pseudo-shared package audit: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

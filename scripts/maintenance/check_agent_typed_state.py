#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Fail CI when Tier-2 agents use raw dict session state access (ACP-DX-6c)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

FORBIDDEN_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bstate\.get\("), "state.get("),
    (re.compile(r"\bstate\[\s*['\"]"), "state["),
)

ALLOWLIST_RELATIVE: frozenset[str] = frozenset()


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    agents_root = repo_root / "agents"
    violations: list[str] = []

    for path in sorted(agents_root.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        rel = path.relative_to(repo_root).as_posix()
        if rel in ALLOWLIST_RELATIVE:
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            for pattern, label in FORBIDDEN_PATTERNS:
                if pattern.search(line):
                    violations.append(f"{rel}:{line_no}: forbidden {label}")

    if violations:
        print("Typed session state violations in agents/:")
        print("\n".join(violations))
        return 1

    print("agents/ typed session state audit: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

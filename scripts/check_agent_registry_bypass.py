#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Fail CI when Tier-2 agents import integrations/tools directly (Phase H-APP.0.4)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

SCAN_ROOTS = ("agents",)
ALLOWLIST = frozenset(
    {
        "agents/__init__.py",
    }
)

FORBIDDEN_IMPORTS = re.compile(
    r"^\s*(?:from|import)\s+intergrax\.(?:integrations|tools)\.",
    re.MULTILINE,
)


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
            if rel in ALLOWLIST:
                continue
            text = path.read_text(encoding="utf-8")
            if FORBIDDEN_IMPORTS.search(text):
                violations.append(rel)
    if violations:
        print("Tier-2 agents must not import intergrax.integrations/tools directly:")
        for item in sorted(violations):
            print(f"  - {item}")
        return 1
    print("check_agent_registry_bypass: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

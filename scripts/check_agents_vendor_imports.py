#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Audit Tier-2 agents for direct vendor / integration provider imports (B.24)."""

from __future__ import annotations

import sys
from pathlib import Path

BANNED_PREFIXES = (
    "intergrax.integrations.providers",
    "boto3",
    "httpx",
    "openai",
    "slack_sdk",
    "google.cloud",
    "azure.",
)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    agents_root = repo_root / "agents"
    violations: list[str] = []
    for path in agents_root.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith(("import ", "from ")):
                for banned in BANNED_PREFIXES:
                    if banned in stripped:
                        rel = path.relative_to(repo_root)
                        violations.append(f"{rel}:{line_no}: {stripped}")
    if violations:
        print("Forbidden imports in agents/:")
        print("\n".join(violations))
        return 1
    print("agents/ vendor import audit: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

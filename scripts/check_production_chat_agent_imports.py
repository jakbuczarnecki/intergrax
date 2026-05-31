#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Ensure ChatAgent is not imported from production paths (Phase K.5 / §39)."""

from __future__ import annotations

import sys
from pathlib import Path

SCAN_ROOTS = (
    "applications",
    "intergrax/runtime",
    "intergrax/fastapi_core",
    "intergrax/applications",
    "intergrax/agents",
    "intergrax/tools",
    "intergrax/scaffold",
)

FORBIDDEN = (
    "from intergrax.chat_agent import",
    "import intergrax.chat_agent",
    "from intergrax import chat_agent",
)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    violations: list[str] = []
    for root_name in SCAN_ROOTS:
        root = repo_root / root_name
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if "tests" in path.parts or path.name == "__init__.py" and "chat_agent" in str(path):
                continue
            for line_no, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                for forbidden in FORBIDDEN:
                    if forbidden in stripped:
                        rel = path.relative_to(repo_root)
                        violations.append(f"{rel}:{line_no}: {stripped}")
    if violations:
        print("ChatAgent imports in production paths:")
        print("\n".join(violations))
        return 1
    print("production ChatAgent import audit: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

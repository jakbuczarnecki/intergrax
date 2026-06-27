#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Ensure deprecated harness modules stay removed (Phase CLEAN)."""

from __future__ import annotations

import sys
from pathlib import Path

REMOVED_MODULES = (
    "intergrax/tools/tools_agent.py",
    "intergrax/legacy/chat_router.py",
    "intergrax/chains/langchain_qa_chain.py",
    "intergrax/chains/__init__.py",
)

FORBIDDEN_IMPORTS = (
    "from intergrax.tools.tools_agent import",
    "import intergrax.tools.tools_agent",
    "from intergrax.legacy.chat_router import",
    "import intergrax.legacy.chat_router",
    "from intergrax.chains",
    "import intergrax.chains",
)

SCAN_ROOTS = (
    "applications",
    "intergrax/runtime",
    "intergrax/agents",
    "intergrax/applications",
    "intergrax/scaffold",
    "agents",
)

GRANDFATHER_PREFIXES = (
    "tests/",
    "scripts/maintenance/check_legacy_modules_removed.py",
)


def _grandfathered(rel: str) -> bool:
    return any(rel.startswith(p) or p in rel for p in GRANDFATHER_PREFIXES)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    errors: list[str] = []

    for rel in REMOVED_MODULES:
        if (repo_root / rel).exists():
            errors.append(f"removed module still present: {rel}")

    for root_name in SCAN_ROOTS:
        root = repo_root / root_name
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            rel = path.relative_to(repo_root).as_posix()
            if _grandfathered(rel):
                continue
            for line_no, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                for forbidden in FORBIDDEN_IMPORTS:
                    if forbidden in stripped:
                        errors.append(f"{rel}:{line_no}: {stripped}")

    if errors:
        print("Legacy module removal audit failed:")
        print("\n".join(errors))
        return 1
    print("legacy module removal audit: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

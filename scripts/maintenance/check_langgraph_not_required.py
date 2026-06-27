#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Ensure LangGraph is not a core runtime dependency (Phase DX / harness independence)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"

# Module-level ``from langgraph`` / ``import langgraph`` allowed only in legacy adapters.
GRANDFATHER_IMPORTS: frozenset[str] = frozenset(
    {
        "intergrax/websearch/integration/langgraph_nodes.py",
        # Deprecated supervisor bridge — lazy import inside _import_langgraph().
        "intergrax/supervisor/supervisor_to_state_graph.py",
    }
)

SCAN_ROOTS = ("intergrax", "agents", "applications")

# Only module-level imports (no leading whitespace) are forbidden outside grandfather paths.
IMPORT_PATTERN = re.compile(
    r"^(?:from\s+langgraph\b|import\s+langgraph\b)",
)


def _core_dependencies_block() -> str:
    text = PYPROJECT.read_text(encoding="utf-8")
    start = text.index("dependencies = [")
    end = text.index("]", start) + 1
    return text[start:end]


def check_pyproject() -> list[str]:
    block = _core_dependencies_block()
    if re.search(r'["\']langgraph', block):
        return ["pyproject.toml: langgraph must not appear in [project].dependencies"]
    return []


def check_imports() -> list[str]:
    violations: list[str] = []
    for root_name in SCAN_ROOTS:
        root = REPO_ROOT / root_name
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if "tests" in path.parts:
                continue
            rel = path.relative_to(REPO_ROOT).as_posix()
            if rel in GRANDFATHER_IMPORTS:
                continue
            for line_no, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if IMPORT_PATTERN.match(stripped):
                    violations.append(f"{rel}:{line_no}: {stripped}")
    return violations


def main() -> int:
    errors = check_pyproject() + check_imports()
    if errors:
        print("LangGraph must not be required by the Intergrax runtime:")
        print("\n".join(errors))
        return 1
    print("langgraph independence audit: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

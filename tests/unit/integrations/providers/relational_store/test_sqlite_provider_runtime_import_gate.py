# © Artur Czarnecki. All rights reserved.

"""R6-AUDIT-MAJOR-01 — SQLite provider must not import runtime/Nexus concrete stores."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[5]
SQLITE_PROVIDER = (
    REPO / "intergrax" / "integrations" / "providers" / "relational_store" / "sqlite"
)

_ALLOWED_RUNTIME_PREFIXES = ("intergrax.runtime.integrations.",)

_FORBIDDEN_PREFIXES = (
    "intergrax.runtime.nexus.",
    "intergrax.runtime.events.",
    "intergrax.runtime.human.",
    "intergrax.runtime.long_running.",
    "intergrax.runtime.task_memory.",
    "intergrax.runtime.organization.",
    "intergrax.runtime.tools.",
    "intergrax.runtime.notifications.",
    "intergrax.experiments.",
    "intergrax.memory.stores.",
)


def _is_forbidden_runtime_import(module: str) -> bool:
    if any(module.startswith(p) for p in _ALLOWED_RUNTIME_PREFIXES):
        return False
    if module.startswith("intergrax.runtime."):
        return True
    return any(module.startswith(p) for p in _FORBIDDEN_PREFIXES)


def _forbidden_imports_in_file(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if _is_forbidden_runtime_import(name):
                    violations.append(f"{path.name}:{node.lineno} import {name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            mod = node.module
            if _is_forbidden_runtime_import(mod):
                violations.append(f"{path.name}:{node.lineno} from {mod}")
    return violations


def test_sqlite_provider_package_has_no_runtime_concrete_imports() -> None:
    violations: list[str] = []
    for path in sorted(SQLITE_PROVIDER.glob("*.py")):
        violations.extend(_forbidden_imports_in_file(path))
    assert not violations, "SQLite provider runtime import violations:\n" + "\n".join(
        violations
    )

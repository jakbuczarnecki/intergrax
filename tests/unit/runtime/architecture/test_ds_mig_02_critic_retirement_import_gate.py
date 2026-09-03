# © Artur Czarnecki. All rights reserved.

"""DS-MIG-02 — production modules must not import legacy Critic orchestration wiring."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_INTERGRAX_ROOT = _REPO_ROOT / "intergrax"

_FORBIDDEN_MODULES = frozenset(
    {
        "intergrax.runtime.critic.critic_orchestrator",
        "intergrax.runtime.critic.critic_wiring",
    },
)

_ALLOWED_PATHS = (
    "intergrax/runtime/critic/",
    "intergrax/runtime/migration/",
    "intergrax/agents/authoring/critic_gateway.py",
)


def _is_allowed_path(path: Path) -> bool:
    rel = path.relative_to(_REPO_ROOT).as_posix()
    return rel.startswith(_ALLOWED_PATHS)


def _module_from_import(node: ast.Import | ast.ImportFrom) -> str | None:
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name in _FORBIDDEN_MODULES:
                return alias.name
        return None
    if node.module in _FORBIDDEN_MODULES:
        return node.module
    return None


def _collect_forbidden_imports(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(path))
    rel = path.relative_to(_REPO_ROOT).as_posix()
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        forbidden = _module_from_import(node)
        if forbidden is not None:
            violations.append(f"{rel}:{node.lineno}: {forbidden}")
    return violations


def _production_forbidden_import_violations() -> list[str]:
    violations: list[str] = []
    for path in _INTERGRAX_ROOT.rglob("*.py"):
        if _is_allowed_path(path):
            continue
        violations.extend(_collect_forbidden_imports(path))
    return violations


def test_production_modules_do_not_import_critic_orchestrator_wiring() -> None:
    violations = _production_forbidden_import_violations()
    assert violations == [], (
        "Production intergrax modules must not import critic orchestrator wiring: "
        + ", ".join(violations)
    )

# © Artur Czarnecki. All rights reserved.

"""UE-10R1 — Nexus root lifecycle retirement architecture gate."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_NEXUS_LOOP_PATH = _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "nexus_loop.py"

_FORBIDDEN_ROOT_LIFECYCLE_CALLS = frozenset(
    {
        "mint_attempt_id",
        "mint_execution_id",
        "bind_active_execution_identity",
        "bind_root_execution_budget",
    }
)


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _collect_forbidden_calls(path: Path, forbidden: frozenset[str]) -> list[str]:
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(path))
    rel = path.relative_to(_REPO_ROOT).as_posix()
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name in forbidden:
            violations.append(f"{rel}:{node.lineno}: {name}()")
    return violations


def test_nexus_loop_has_no_root_lifecycle_mint_or_bind() -> None:
    violations = _collect_forbidden_calls(_NEXUS_LOOP_PATH, _FORBIDDEN_ROOT_LIFECYCLE_CALLS)
    assert violations == [], (
        "NexusLoop must not mint or bind root lifecycle: " + ", ".join(violations)
    )


def test_nexus_loop_has_no_root_authority_bootstrap() -> None:
    source = _NEXUS_LOOP_PATH.read_text(encoding="utf-8")
    assert "resolve_root_parent_execution_authority" not in source
    assert "bind_active_execution_authority" not in source

# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-15-R3: bounded AST guard — no private manager/memory introspection in certification."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.gate

_REPO = Path(__file__).resolve().parents[3]
_E2E_ROOT = _REPO / "tests" / "integration" / "memory" / "e2e"
_UNIT_WIRING = Path(__file__).with_name("test_mem_ent15_r2_entity_projection_wiring.py")

_FORBIDDEN_ATTRS = frozenset(
    {
        "_memory_lifecycle",
        "_store",
        "_projection",
        "_manager",
    }
)
_FORBIDDEN_REFLECTION = frozenset({"getattr", "hasattr", "setattr"})
_OWNER_NAMES = frozenset({"self", "cls"})
_DUNDER_ALLOW = frozenset({"__class__", "__dict__", "__name__", "__module__"})


def _certification_paths() -> list[Path]:
    paths = [_UNIT_WIRING]
    paths.extend(sorted(_E2E_ROOT.glob("test_mem_ent15_*.py")))
    paths.append(_E2E_ROOT / "harness.py")
    return paths


def _private_access_violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_REFLECTION:
            violations.append(f"{path.name}:{node.lineno} reflection {node.id}")
        if not isinstance(node, ast.Attribute):
            continue
        if isinstance(node.value, ast.Name) and node.value.id in _OWNER_NAMES:
            continue
        attr = node.attr
        if attr in _DUNDER_ALLOW:
            continue
        if attr in _FORBIDDEN_ATTRS or (
            attr.startswith("_") and not attr.startswith("__")
        ):
            violations.append(f"{path.name}:{node.lineno} private access {attr}")
    return violations


def test_mem_ent15_certification_has_no_external_private_access() -> None:
    all_violations: list[str] = []
    for path in _certification_paths():
        if path.name == "test_mem_ent15_r3_certification_private_access_guard.py":
            continue
        if path.name == "test_mem_ent15_guards.py":
            continue
        all_violations.extend(_private_access_violations(path))
    assert all_violations == [], "\n".join(sorted(all_violations))

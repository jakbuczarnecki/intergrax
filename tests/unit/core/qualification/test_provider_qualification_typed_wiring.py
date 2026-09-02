# © Artur Czarnecki. All rights reserved.

"""AST architecture gate for provider qualification typed wiring (PROVIDER-QUAL-2)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_FORBIDDEN_CALLS = frozenset({"getattr", "setattr", "hasattr", "vars"})
_FORBIDDEN_ATTRIBUTES = frozenset({"__dict__", "__setattr__"})

_SCOPED_PRODUCTION_FILES = (
    "intergrax/core/qualification/provider.py",
    "intergrax/core/qualification/validity.py",
    "intergrax/core/qualification/execution.py",
    "intergrax/core/qualification/suite.py",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _collect_violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS:
                violations.append(f"{path.name}:{node.lineno} calls {node.func.id}()")
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "__setattr__"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "object"
            ):
                violations.append(f"{path.name}:{node.lineno} calls object.__setattr__()")
        if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_ATTRIBUTES:
            violations.append(f"{path.name}:{node.lineno} references .{node.attr}")
    return violations


@pytest.mark.parametrize("relative_path", _SCOPED_PRODUCTION_FILES)
def test_scoped_provider_qualification_production_has_no_dynamic_attribute_access(
    relative_path: str,
) -> None:
    path = _repo_root() / relative_path
    violations = _collect_violations(path)
    assert not violations, "\n".join(violations)

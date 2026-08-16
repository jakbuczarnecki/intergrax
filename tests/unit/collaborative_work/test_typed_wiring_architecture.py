# © Artur Czarnecki. All rights reserved.

"""AST architecture gate for Collaborative Work typed wiring (COLLAB-WORK-1H-R2)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_FORBIDDEN_CALLS = frozenset({"getattr", "setattr", "hasattr", "vars"})
_FORBIDDEN_ATTRIBUTES = frozenset({"__dict__", "__setattr__"})

_SCOPED_PRODUCTION_FILES = (
    "intergrax/contracts/collaborative_work.py",
    "intergrax/collaborative_work/repository.py",
    "intergrax/collaborative_work/in_memory_repository.py",
    "intergrax/collaborative_work/sqlite_repository.py",
    "intergrax/collaborative_work/postgresql_repository.py",
    "intergrax/collaborative_work/persistence.py",
    "intergrax/collaborative_work/authority.py",
    "intergrax/collaborative_work/policy_source.py",
    "intergrax/collaborative_work/policy_composition.py",
    "intergrax/collaborative_work/enforcement_gate.py",
    "intergrax/runtime/policy/meaningful_side_effect_authorization.py",
)

_POSTGRESQL_PLATFORM_FILES = (
    "intergrax/integrations/providers/relational_store/postgresql/session.py",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


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


def _collect_postgresql_session_boundary_violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "_connection":
            if not (isinstance(node.value, ast.Name) and node.value.id == "self"):
                violations.append(
                    f"{path.name}:{node.lineno} accesses private session connection via .{node.attr}"
                )
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("psycopg"):
            if not _is_inside_function(tree, node, "import_psycopg"):
                violations.append(
                    f"{path.name}:{node.lineno} imports psycopg outside import_psycopg()"
                )
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "psycopg" and not _is_inside_function(
                    tree, node, "import_psycopg"
                ):
                    violations.append(
                        f"{path.name}:{node.lineno} imports psycopg outside import_psycopg()"
                    )
    return violations


def _is_inside_function(tree: ast.AST, target: ast.AST, function_name: str) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            for child in ast.walk(node):
                if child is target:
                    return True
    return False


@pytest.mark.parametrize("relative_path", _SCOPED_PRODUCTION_FILES)
def test_scoped_collaborative_work_production_has_no_dynamic_attribute_access(
    relative_path: str,
) -> None:
    path = _repo_root() / relative_path
    violations = _collect_violations(path)
    assert not violations, "\n".join(violations)


@pytest.mark.parametrize("relative_path", _POSTGRESQL_PLATFORM_FILES)
def test_postgresql_platform_session_boundary_has_no_private_connection_access(
    relative_path: str,
) -> None:
    path = _repo_root() / relative_path
    violations = _collect_postgresql_session_boundary_violations(path)
    assert not violations, "\n".join(violations)

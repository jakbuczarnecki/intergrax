# © Artur Czarnecki. All rights reserved.

"""Architecture gates — E2E harness vendor neutrality and typed wiring."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_E2E_HARNESS_DIR = Path(__file__).resolve().parent / "harness"
_E2E_COMPOSITION_FILES = tuple(sorted(_E2E_HARNESS_DIR.glob("*.py")))
_FORBIDDEN_IMPORT_PREFIXES = (
    "psycopg",
    "sqlite3",
    "boto3",
    "sqlalchemy",
)
_FORBIDDEN_CALLS = frozenset({"getattr", "setattr", "hasattr", "vars"})
_FORBIDDEN_ATTRIBUTES = frozenset({"__dict__", "__setattr__"})
_VENDOR_LITERALS = frozenset({"postgresql", "sqlite", "mysql", "oracle"})


def _import_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def _walk_calls(path: Path) -> list[ast.Call]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


def test_e2e_harness_core_has_no_vendor_imports() -> None:
    imported: set[str] = set()
    for path in _E2E_COMPOSITION_FILES:
        if path.name == "profile_factory.py":
            continue
        imported |= _import_names(path)
    assert not imported.intersection(_FORBIDDEN_IMPORT_PREFIXES)


def test_e2e_composition_has_no_vendor_branch_literals() -> None:
    composition = _E2E_HARNESS_DIR / "composition.py"
    source = composition.read_text(encoding="utf-8").lower()
    for literal in _VENDOR_LITERALS:
        assert literal not in source


def test_e2e_harness_has_no_untyped_wiring() -> None:
    violations: list[str] = []
    for path in _E2E_COMPOSITION_FILES:
        for call in _walk_calls(path):
            if isinstance(call.func, ast.Name) and call.func.id in _FORBIDDEN_CALLS:
                violations.append(f"{path.name}:{call.lineno} calls {call.func.id}")
            if isinstance(call.func, ast.Attribute) and call.func.attr in _FORBIDDEN_ATTRIBUTES:
                violations.append(f"{path.name}:{call.lineno} accesses {call.func.attr}")
    assert not violations

# © Artur Czarnecki. All rights reserved.

"""HARDENING-8 — diagnostic Problem persistence contract boundary."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.contracts.diagnostics.problem_persistence import ProblemPersistence
from intergrax.runtime.diagnostics.in_memory_problem_persistence import (
    InMemoryProblemPersistence,
)
from intergrax.runtime.diagnostics.persistence_conformance import sample_problem

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CONTRACTS_DIAG = _REPO_ROOT / "intergrax" / "contracts" / "diagnostics"


def _collect_runtime_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("intergrax.runtime"):
                    hits.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("intergrax.runtime"):
                hits.append(node.module)
    return hits


def test_hardening_8_diagnostic_contracts_do_not_import_runtime() -> None:
    violations: list[str] = []
    for path in sorted(_CONTRACTS_DIAG.rglob("*.py")):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for module in _collect_runtime_imports(path):
            violations.append(f"{rel}: {module}")
    assert violations == [], "diagnostics contracts → runtime:\n" + "\n".join(violations)


def test_hardening_8_runtime_problem_persistence_is_reexport_only() -> None:
    path = _REPO_ROOT / "intergrax" / "runtime" / "diagnostics" / "problem_persistence.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            forbidden.append(node.name)
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            forbidden.append(node.name)
    assert forbidden == [], f"runtime problem_persistence must not define types: {forbidden}"


def test_hardening_8_in_memory_adapter_satisfies_problem_persistence_port() -> None:
    store = InMemoryProblemPersistence()
    assert isinstance(store, ProblemPersistence)


def test_hardening_8_problem_persistence_port_is_swappable() -> None:
    primary = InMemoryProblemPersistence()
    alternate = InMemoryProblemPersistence()
    record = sample_problem(tenant_id="tenant-boundary")
    primary.create(record)
    assert alternate.get(tenant_id=record.tenant_id, problem_id=record.problem_id) is None
    loaded = primary.get(tenant_id=record.tenant_id, problem_id=record.problem_id)
    assert loaded is not None
    assert loaded.problem_id == record.problem_id


def test_hardening_8_document_store_adapter_satisfies_problem_persistence_port() -> None:
    from tests.unit.runtime.diagnostics.problem_persistence_test_support import (
        document_store_problem_persistence_for_tests,
    )

    store = document_store_problem_persistence_for_tests()
    assert isinstance(store, ProblemPersistence)

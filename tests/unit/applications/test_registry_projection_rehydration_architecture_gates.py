# © Artur Czarnecki. All rights reserved.

"""Architecture gates for durable registry projection rehydration."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
REHYDRATOR = (
    REPO_ROOT
    / "intergrax"
    / "applications"
    / "_shared"
    / "registry_projection_rehydrator.py"
)
DESCRIPTOR = (
    REPO_ROOT
    / "intergrax"
    / "applications"
    / "_shared"
    / "registry_projection_descriptor.py"
)
ARCH_DOC = REPO_ROOT / "docs" / "project" / "architecture" / "AGENT_DISTRIBUTION.md"

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_FORBIDDEN_REHYDRATOR_IMPORTS = frozenset(
    {
        "intergrax.agent_distribution.activation",
        "intergrax.agent_distribution.installation_service",
        "intergrax.agent_distribution.binding_service",
        "intergrax.agent_distribution.materialization_service",
        "intergrax.agent_distribution.admin_service",
    }
)

_FORBIDDEN_REHYDRATOR_CALLS = frozenset(
    {
        "install_agent",
        "bind_agent",
        "build_application_revision",
        "commit_activation",
        "activate_revision",
        "persist_candidate_revision",
    }
)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _call_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


def test_rehydrator_has_no_sqlite_imports() -> None:
    modules = _imported_modules(REHYDRATOR)
    assert not any(module.startswith("intergrax.agent_distribution.sqlite") for module in modules)


def test_descriptor_module_has_no_sqlite_imports() -> None:
    modules = _imported_modules(DESCRIPTOR)
    assert not any(module.startswith("sqlite3") for module in modules)
    assert not any(
        module.startswith("intergrax.agent_distribution.sqlite") for module in modules
    )


def test_rehydrator_does_not_import_lifecycle_mutation_services() -> None:
    modules = _imported_modules(REHYDRATOR)
    violations = sorted(modules & _FORBIDDEN_REHYDRATOR_IMPORTS)
    assert not violations, f"forbidden rehydrator imports: {violations}"


def test_rehydrator_does_not_call_lifecycle_mutations() -> None:
    calls = _call_names(REHYDRATOR)
    violations = sorted(calls & _FORBIDDEN_REHYDRATOR_CALLS)
    assert not violations, f"forbidden rehydrator calls: {violations}"


def test_architecture_doc_documents_rehydration_boundary() -> None:
    text = ARCH_DOC.read_text(encoding="utf-8")
    assert "rehydration" in text.lower()
    assert "startup-time projection from mutable desired state" in text.lower()

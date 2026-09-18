# © Artur Czarnecki. All rights reserved.

"""MP-5F-B3A architecture gates for canonical workspace ownership contracts."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_OWNERSHIP_PATHS = (
    _REPO_ROOT / "intergrax" / "runtime" / "context_lifecycle" / "contracts.py",
    _REPO_ROOT / "intergrax" / "runtime" / "context_lifecycle" / "repository.py",
    _REPO_ROOT / "intergrax" / "runtime" / "context_lifecycle" / "in_memory_repository.py",
    _REPO_ROOT / "intergrax" / "runtime" / "context_lifecycle" / "sqlite_repository.py",
)
_COMPOSITION_WIRING_PATHS = (
    _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "context" / "context_engine.py",
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "nexus"
    / "context"
    / "ucl_artifact_ownership_composition.py",
)
_GRAPH_ASSEMBLY_PATH = _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "context" / "graph_assembly.py"
_FORBIDDEN_IMPORT_PREFIXES = (
    "intergrax.runtime.nexus",
    "intergrax.contracts.context_view",
    "intergrax.collaborative_work",
    "applications.",
    "agents.",
)
_FORBIDDEN_AST_NAMES = frozenset({"getattr", "setattr", "hasattr"})


def _collect_imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append((node.lineno, node.module))
    return imports


def _collect_forbidden_ast_usage(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_AST_NAMES:
            violations.append(f"{path.name}:{node.lineno} references forbidden name {node.id}")
    return violations


def test_b3a_ownership_boundary_has_no_forbidden_imports() -> None:
    violations: list[str] = []
    for path in _OWNERSHIP_PATHS:
        for lineno, module in _collect_imports(path):
            if any(module == prefix or module.startswith(f"{prefix}.") for prefix in _FORBIDDEN_IMPORT_PREFIXES):
                violations.append(f"{path.name}:{lineno} imports {module}")
    assert not violations, "\n".join(violations)


def test_b3a_ownership_boundary_has_no_dynamic_bypass() -> None:
    violations: list[str] = []
    for path in _OWNERSHIP_PATHS:
        violations.extend(_collect_forbidden_ast_usage(path))
    assert not violations, "\n".join(violations)


_COMPOSITION_FORBIDDEN_IMPORT_PREFIXES = (
    "intergrax.contracts.context_view",
    "intergrax.collaborative_work",
    "applications.",
    "agents.",
    "intergrax.runtime.context_lifecycle.in_memory_repository",
    "intergrax.runtime.context_lifecycle.sqlite_repository",
)


def test_b3a_c1_composition_wiring_has_no_forbidden_imports() -> None:
    violations: list[str] = []
    for path in _COMPOSITION_WIRING_PATHS:
        for lineno, module in _collect_imports(path):
            if any(
                module == prefix or module.startswith(f"{prefix}.")
                for prefix in _COMPOSITION_FORBIDDEN_IMPORT_PREFIXES
            ):
                violations.append(f"{path.name}:{lineno} imports {module}")
    assert not violations, "\n".join(violations)


def test_b3a_c1_composition_wiring_has_no_dynamic_bypass() -> None:
    violations: list[str] = []
    for path in _COMPOSITION_WIRING_PATHS:
        violations.extend(_collect_forbidden_ast_usage(path))
    composition = _COMPOSITION_WIRING_PATHS[1].read_text(encoding="utf-8")
    if "context_scope_id" in composition:
        violations.append("ucl_artifact_ownership_composition must not derive workspace from context_scope_id")
    assert not violations, "\n".join(violations)


def test_b3a_c0_graph_assembly_does_not_read_workspace_from_metadata() -> None:
    source = _GRAPH_ASSEMBLY_PATH.read_text(encoding="utf-8")
    assert 'metadata.get("workspace_id")' not in source
    assert "context_scope_id" not in source


def test_b3a_ownership_contract_symbols_exist() -> None:
    from intergrax.runtime.context_lifecycle.contracts import (
        UclArtifactOwnership,
        UclArtifactOwnershipKind,
        UclArtifactOwnershipScope,
    )

    assert UclArtifactOwnershipKind.WORKSPACE.value == "workspace"
    scope = UclArtifactOwnershipScope(tenant_id="t1", workspace_id="w1")
    ownership = UclArtifactOwnership.for_workspace(scope)
    assert ownership.scope == scope

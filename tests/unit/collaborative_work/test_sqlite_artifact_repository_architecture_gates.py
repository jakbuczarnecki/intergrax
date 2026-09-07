# © Artur Czarnecki. All rights reserved.

"""Static architecture gates for MP-3D SQLite artifact persistence."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SQLITE_REPOSITORY = _REPO_ROOT / "intergrax" / "collaborative_work" / "sqlite_repository.py"
_PRODUCTION_PATHS = (_SQLITE_REPOSITORY,)
_FORBIDDEN_IMPORTS = (
    "intergrax.runtime.nexus",
    "intergrax.contracts.execution_evidence.receipt",
    "intergrax.runtime.artifacts",
)
_FORBIDDEN_AST_NAMES = frozenset(
    {
        "getattr",
        "setattr",
        "hasattr",
        "vars",
        "Any",
    },
)
_FORBIDDEN_GENERIC_PATTERNS = (
    re.compile(r"Repository\s*\[\s*"),
    re.compile(r"UnitOfWork"),
    re.compile(r"TransactionManager"),
    re.compile(r"ArtifactSQLiteService"),
)


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
        if isinstance(node, ast.Attribute) and node.attr == "__dict__":
            violations.append(f"{path.name}:{node.lineno} references forbidden name __dict__")
    return violations


def test_mp3d_production_modules_have_no_forbidden_imports() -> None:
    violations: list[str] = []
    for path in _PRODUCTION_PATHS:
        for lineno, module in _collect_imports(path):
            if any(module == prefix or module.startswith(f"{prefix}.") for prefix in _FORBIDDEN_IMPORTS):
                violations.append(f"{path.name}:{lineno} imports {module}")
    assert not violations, "\n".join(violations)


def test_mp3d_production_modules_have_no_forbidden_dynamic_patterns() -> None:
    violations: list[str] = []
    for path in _PRODUCTION_PATHS:
        violations.extend(_collect_forbidden_ast_usage(path))
    assert not violations, "\n".join(violations)


def test_mp3d_production_modules_have_no_generic_repository_or_uow() -> None:
    violations: list[str] = []
    for path in _PRODUCTION_PATHS:
        source = path.read_text(encoding="utf-8")
        for pattern in _FORBIDDEN_GENERIC_PATTERNS:
            if pattern.search(source):
                violations.append(f"{path.name} matches forbidden pattern {pattern.pattern}")
    assert not violations, "\n".join(violations)


def test_sqlite_schema_defines_artifact_tables() -> None:
    source = _SQLITE_REPOSITORY.read_text(encoding="utf-8")
    assert "CREATE TABLE IF NOT EXISTS work_artifacts" in source
    assert "CREATE TABLE IF NOT EXISTS work_artifact_versions" in source
    assert "idx_work_artifacts_work_item" in source
    assert "idx_work_artifact_versions_history" in source


def test_persistence_composition_defines_artifact_bundle() -> None:
    source = (_REPO_ROOT / "intergrax" / "collaborative_work" / "persistence.py").read_text(
        encoding="utf-8",
    )
    assert "class CollaborativeWorkArtifactRepositories" in source
    assert "class CollaborativeWorkRepositoriesWithArtifacts" in source
    assert "SQLiteWorkArtifactRepository" in source

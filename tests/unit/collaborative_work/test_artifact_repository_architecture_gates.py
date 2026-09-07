# © Artur Czarnecki. All rights reserved.

"""Static architecture gates for MP-3B artifact repository surface."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PRODUCTION_PATHS = (
    _REPO_ROOT / "intergrax" / "collaborative_work" / "repository.py",
    _REPO_ROOT / "intergrax" / "collaborative_work" / "in_memory_repository.py",
)
_FORBIDDEN_IMPORTS = (
    "intergrax.collaborative_work.enforcement_gate",
    "intergrax.collaborative_work.service",
    "intergrax.collaborative_work.sqlite_repository",
    "intergrax.collaborative_work.postgresql_repository",
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
_FORBIDDEN_ATTRIBUTE_NAMES = frozenset({"__dict__"})
_FORBIDDEN_GENERIC_PATTERNS = (
    re.compile(r"Repository\s*\[\s*"),
    re.compile(r"UnitOfWork"),
    re.compile(r"TransactionManager"),
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
        if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_ATTRIBUTE_NAMES:
            violations.append(f"{path.name}:{node.lineno} references forbidden name {node.attr}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "__setattr__" and isinstance(node.func.value, ast.Name):
                if node.func.value.id == "object":
                    violations.append(
                        f"{path.name}:{node.lineno} references forbidden object.__setattr__",
                    )
    return violations


def test_mp3b_production_modules_have_no_forbidden_imports() -> None:
    violations: list[str] = []
    for path in _PRODUCTION_PATHS:
        for lineno, module in _collect_imports(path):
            if any(module == prefix or module.startswith(f"{prefix}.") for prefix in _FORBIDDEN_IMPORTS):
                violations.append(f"{path.name}:{lineno} imports {module}")
    assert not violations, "\n".join(violations)


def test_mp3b_production_modules_have_no_forbidden_dynamic_patterns() -> None:
    violations: list[str] = []
    for path in _PRODUCTION_PATHS:
        violations.extend(_collect_forbidden_ast_usage(path))
    assert not violations, "\n".join(violations)


def test_mp3b_production_modules_have_no_generic_repository_or_uow() -> None:
    violations: list[str] = []
    for path in _PRODUCTION_PATHS:
        source = path.read_text(encoding="utf-8")
        for pattern in _FORBIDDEN_GENERIC_PATTERNS:
            if pattern.search(source):
                violations.append(f"{path.name} matches forbidden pattern {pattern.pattern}")
    assert not violations, "\n".join(violations)


def test_mp3b_repository_surface_defines_artifact_ports() -> None:
    source = (_REPO_ROOT / "intergrax" / "collaborative_work" / "repository.py").read_text(
        encoding="utf-8",
    )
    for symbol in (
        "class WorkArtifactRepository",
        "class WorkArtifactVersionRepository",
        "class ArtifactPublicationRepository",
        "class CreateArtifactWithInitialVersionCommand",
        "class PublishWorkArtifactVersionCommand",
        "class PublishedWorkArtifactVersion",
        "class WorkArtifactRevisionConflict",
        "class WorkArtifactTemporalConflict",
        "class ArtifactPublicationIdempotencyConflict",
    ):
        assert symbol in source

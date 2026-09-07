# © Artur Czarnecki. All rights reserved.

"""Static architecture gates for MP-3C artifact publication service."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ARTIFACT_SERVICE_PATH = _REPO_ROOT / "intergrax" / "collaborative_work" / "artifact_service.py"
_FORBIDDEN_IMPORTS = (
    "intergrax.collaborative_work.sqlite_repository",
    "intergrax.collaborative_work.postgresql_repository",
    "intergrax.runtime.nexus",
    "intergrax.contracts.execution_evidence.receipt",
    "intergrax.runtime.artifacts",
    "agents.",
    "applications.",
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
_FORBIDDEN_SOURCE_PATTERNS = (
    re.compile(r"WorkArtifactRepository\.(create|update)\s*\("),
    re.compile(r"WorkArtifactVersionRepository\.(create|update)\s*\("),
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
    return violations


def test_mp3c_artifact_service_has_no_forbidden_imports() -> None:
    violations: list[str] = []
    for lineno, module in _collect_imports(_ARTIFACT_SERVICE_PATH):
        if any(module == prefix or module.startswith(f"{prefix}") for prefix in _FORBIDDEN_IMPORTS):
            violations.append(f"{_ARTIFACT_SERVICE_PATH.name}:{lineno} imports {module}")
    assert not violations, "\n".join(violations)


def test_mp3c_artifact_service_has_no_forbidden_dynamic_patterns() -> None:
    violations = _collect_forbidden_ast_usage(_ARTIFACT_SERVICE_PATH)
    assert not violations, "\n".join(violations)


def test_mp3c_artifact_service_delegates_only_to_publication_port() -> None:
    source = _ARTIFACT_SERVICE_PATH.read_text(encoding="utf-8")
    assert "ArtifactPublicationRepository" in source
    assert "create_artifact_with_initial_version" in source
    assert "publish_version" in source
    for pattern in _FORBIDDEN_SOURCE_PATTERNS:
        assert not pattern.search(source), f"forbidden pattern {pattern.pattern}"


def test_mp3c_artifact_service_sets_execution_none() -> None:
    source = _ARTIFACT_SERVICE_PATH.read_text(encoding="utf-8")
    assert source.count("execution=None") >= 2

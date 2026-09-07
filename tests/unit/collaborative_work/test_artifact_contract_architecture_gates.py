# © Artur Czarnecki. All rights reserved.

"""Static architecture gates for COLLAB-WORK-3A contract surface."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONTRACT_PATH = _REPO_ROOT / "intergrax" / "contracts" / "collaborative_work.py"
_FORBIDDEN_IMPORT_PREFIXES = (
    "intergrax.runtime.nexus",
    "intergrax.collaborative_work.repository",
    "intergrax.collaborative_work.service",
    "intergrax.collaborative_work.in_memory_repository",
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


def test_mp3a_contract_module_has_no_forbidden_imports() -> None:
    violations: list[str] = []
    for lineno, module in _collect_imports(_CONTRACT_PATH):
        if any(module == prefix or module.startswith(f"{prefix}.") for prefix in _FORBIDDEN_IMPORT_PREFIXES):
            violations.append(f"{_CONTRACT_PATH.name}:{lineno} imports {module}")
    assert not violations, "\n".join(violations)


def test_mp3a_contract_module_has_no_forbidden_dynamic_patterns() -> None:
    violations = _collect_forbidden_ast_usage(_CONTRACT_PATH)
    assert not violations, "\n".join(violations)


def test_mp3a_contract_module_defines_artifact_surface() -> None:
    source = _CONTRACT_PATH.read_text(encoding="utf-8")
    for symbol in (
        "class ArtifactContentRef",
        "class WorkArtifact",
        "class WorkArtifactVersion",
        "class CollaborativeWorkArtifactInvariantError",
        "def validate_work_artifact_version_scope",
        "def validate_work_artifact_current_version",
        "SCHEMA_WORK_ARTIFACT_V1",
        "SCHEMA_WORK_ARTIFACT_VERSION_V1",
        "SCHEMA_ARTIFACT_CONTENT_REF_V1",
    ):
        assert symbol in source

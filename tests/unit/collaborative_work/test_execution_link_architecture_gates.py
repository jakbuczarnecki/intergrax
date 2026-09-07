# © Artur Czarnecki. All rights reserved.

"""Static architecture gate for COLLAB-WORK-2F production modules."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCOPED_2F_FILES = (
    "intergrax/contracts/execution_provenance.py",
    "intergrax/collaborative_work/execution_link_service.py",
    "intergrax/collaborative_work/repository.py",
    "intergrax/collaborative_work/in_memory_repository.py",
    "intergrax/collaborative_work/sqlite_repository.py",
    "intergrax/collaborative_work/postgresql_repository.py",
    "intergrax/collaborative_work/persistence.py",
    "intergrax/collaborative_work/serialization.py",
)
_FORBIDDEN_IMPORT_PREFIXES = (
    "intergrax.runtime.nexus",
)
_FORBIDDEN_NAMES = frozenset(
    {
        "NexusLoop",
        "GraphExecutor",
        "TaskState",
        "TaskResult",
    },
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


def _collect_forbidden_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            violations.append(f"{path.name}:{node.lineno} references forbidden name {node.id}")
        if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_NAMES:
            violations.append(f"{path.name}:{node.lineno} references forbidden name {node.attr}")
    return violations


@pytest.mark.parametrize("relative_path", _SCOPED_2F_FILES)
def test_scoped_2f_modules_have_no_nexus_imports(relative_path: str) -> None:
    path = _REPO_ROOT / relative_path
    violations: list[str] = []
    for lineno, module in _collect_imports(path):
        if any(module == prefix or module.startswith(f"{prefix}.") for prefix in _FORBIDDEN_IMPORT_PREFIXES):
            violations.append(f"{path.name}:{lineno} imports {module}")
    assert not violations, "\n".join(violations)


@pytest.mark.parametrize("relative_path", _SCOPED_2F_FILES)
def test_scoped_2f_modules_have_no_forbidden_runtime_names(relative_path: str) -> None:
    path = _REPO_ROOT / relative_path
    violations = _collect_forbidden_names(path)
    assert not violations, "\n".join(violations)

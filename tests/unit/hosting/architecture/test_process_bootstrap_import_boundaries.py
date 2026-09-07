# © Artur Czarnecki. All rights reserved.

"""Architecture gate — guarded process bootstrap must stay hosting-only (DG-001B R2)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_FORBIDDEN_IMPORT_MODULES = frozenset(
    {
        "intergrax.runtime.diagnostics",
        "intergrax.runtime.diagnostics.diagnostic_orchestrator",
        "intergrax.contracts.execution_identity",
        "kafka",
        "celery",
        "rabbitmq",
        "message_bus",
        "local_workspace_application",
        "inspect",
        "importlib",
    }
)

_FORBIDDEN_SYMBOLS = frozenset(
    {
        "DiagnosticOrchestrator",
        "TaskId",
        "RunId",
        "AttemptId",
        "ExecutionId",
    }
)

_FORBIDDEN_REFLECTION_BUILTINS = frozenset({"setattr", "getattr", "hasattr"})

_FORBIDDEN_REFLECTION_MODULES = frozenset({"inspect", "importlib"})

_FORBIDDEN_LOOSE_TYPES = frozenset({"Any", "object"})

_MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "intergrax"
    / "hosting"
    / "process_bootstrap.py"
)


def _annotation_uses_forbidden_container(node: ast.expr) -> bool:
    if isinstance(node, ast.Name) and node.id in _FORBIDDEN_LOOSE_TYPES:
        return True
    if isinstance(node, ast.Subscript):
        container = node.value
        if isinstance(container, ast.Name) and container.id in {"dict", "Mapping"}:
            slice_node = node.slice
            if isinstance(slice_node, ast.Tuple) and len(slice_node.elts) == 2:
                key_type, value_type = slice_node.elts
                if (
                    isinstance(key_type, ast.Name)
                    and key_type.id == "str"
                    and isinstance(value_type, ast.Name)
                    and value_type.id in _FORBIDDEN_LOOSE_TYPES
                ):
                    return True
    return False


def _collect_import_boundary_violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    rel = path.relative_to(Path(__file__).resolve().parents[4]).as_posix()
    violations: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module in _FORBIDDEN_IMPORT_MODULES
        ):
            violations.append(f"{rel}:{node.lineno} imports from {node.module}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in _FORBIDDEN_IMPORT_MODULES:
                    violations.append(f"{rel}:{node.lineno} imports {alias.name}")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_SYMBOLS:
            violations.append(f"{rel}:{node.lineno} references {node.id}")
        if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_SYMBOLS:
            violations.append(f"{rel}:{node.lineno} references .{node.attr}")
    return violations


def _collect_hard_contract_violations(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    rel = path.relative_to(Path(__file__).resolve().parents[4]).as_posix()
    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "__setattr__":
            violations.append(f"{rel}:{node.lineno} uses object.__setattr__")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _FORBIDDEN_REFLECTION_BUILTINS:
                violations.append(f"{rel}:{node.lineno} calls builtin {node.func.id}()")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_REFLECTION_MODULES:
            violations.append(f"{rel}:{node.lineno} references {node.id}")
        if (
            isinstance(node, ast.ImportFrom)
            and node.module in _FORBIDDEN_REFLECTION_MODULES
        ):
            violations.append(f"{rel}:{node.lineno} imports from {node.module}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in _FORBIDDEN_REFLECTION_MODULES:
                    violations.append(f"{rel}:{node.lineno} imports {alias.name}")
        if _annotation_uses_forbidden_container(node):
            violations.append(f"{rel}:{node.lineno} uses forbidden loose typing")

    for lineno, line in enumerate(source.splitlines(), start=1):
        if "type: ignore" in line:
            violations.append(f"{rel}:{lineno} uses type: ignore")

    return violations


def test_process_bootstrap_import_boundaries() -> None:
    assert _collect_import_boundary_violations(_MODULE_PATH) == []


def test_process_bootstrap_hard_contract() -> None:
    assert _collect_hard_contract_violations(_MODULE_PATH) == []

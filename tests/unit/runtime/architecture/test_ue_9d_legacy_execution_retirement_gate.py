# © Artur Czarnecki. All rights reserved.

"""UE-9D — global legacy execution retirement gate."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_EXECUTION_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "execution"
_TASK_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "task"
_INTERACTIONS_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "interactions"

_LEGACY_FILES = (
    _EXECUTION_ROOT / "task_compat.py",
    _EXECUTION_ROOT / "nexus_compat.py",
)

_LEGACY_SYMBOLS = frozenset(
    {
        "UnifiedTaskRunnerExecutionDelegate",
        "NexusTaskExecutionDelegate",
        "_LEGACY_ORCHESTRATION_CAPABILITIES",
        "legacy UnifiedTaskRunner compatibility path",
    }
)

_FORBIDDEN_IMPORTS_IN_UNIFIED_TASK_RUNNER = frozenset(
    {
        "StrategyResolver",
        "ExecutionBoundary",
        "NexusTaskExecutionDelegate",
        "UnifiedTaskRunnerExecutionDelegate",
        "mint_run_id",
        "mint_attempt_id",
        "mint_execution_id",
    }
)


def test_legacy_compat_modules_are_absent() -> None:
    for path in _LEGACY_FILES:
        assert not path.exists(), path.relative_to(_REPO_ROOT).as_posix()


def test_execution_runtime_has_zero_legacy_symbols() -> None:
    violations: list[str] = []
    for path in _EXECUTION_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for symbol in _LEGACY_SYMBOLS:
                if symbol in line:
                    violations.append(f"{rel}:{lineno}: {symbol}")
    assert violations == []


def test_unified_task_runner_is_thin_adapter_without_legacy_root_path() -> None:
    source = (_TASK_ROOT / "unified_task_runner.py").read_text(encoding="utf-8")
    for symbol in _LEGACY_SYMBOLS:
        assert symbol not in source
    for forbidden in _FORBIDDEN_IMPORTS_IN_UNIFIED_TASK_RUNNER:
        assert forbidden not in source


def test_interaction_task_executor_does_not_call_nexus_handle_task_directly() -> None:
    source = (_INTERACTIONS_ROOT / "task_executor.py").read_text(encoding="utf-8")
    assert "handle_task(" not in source


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def test_orchestration_executor_is_canonical_backend() -> None:
    path = _EXECUTION_ROOT / "orchestration.py"
    assert path.exists()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    class_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
    }
    assert "OrchestrationExecutor" in class_names
    assert "NexusTaskExecutionDelegate" not in class_names

    handle_task_calls = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_name(node.func) == "handle_task"
    ]
    assert len(handle_task_calls) == 1

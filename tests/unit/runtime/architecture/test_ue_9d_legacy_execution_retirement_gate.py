# © Artur Czarnecki. All rights reserved.

"""UE-9D / UE-9DR1 — legacy execution retirement and canonical routing gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_EXECUTION_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "execution"
_TASK_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "task"
_INTERACTIONS_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "interactions"
_PRODUCTION_ROOT = _REPO_ROOT / "intergrax"

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

_STRATEGY_RESOLVER_OWNER = _EXECUTION_ROOT / "strategy_router.py"
_STRATEGIC_BACKEND_OWNER = _STRATEGY_RESOLVER_OWNER
_STRATEGIC_BACKEND_CLASSES = frozenset({"AgentExecutor", "InferenceExecutor"})
_STRATEGIC_BACKEND_EXECUTE_ATTRS = frozenset({"_agent_executor", "_inference_executor"})

_HANDLE_TASK_ALLOWLIST = frozenset(
    {
        _EXECUTION_ROOT / "orchestration.py",
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


def _read_python_source(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _iter_production_python_files() -> list[Path]:
    paths: list[Path] = []
    for path in _PRODUCTION_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        if path.parts[-1].startswith("test_"):
            continue
        paths.append(path)
    return paths


def _collect_handle_task_call_sites() -> list[str]:
    violations: list[str] = []
    for path in _iter_production_python_files():
        if path in _HANDLE_TASK_ALLOWLIST:
            continue
        tree = ast.parse(_read_python_source(path), filename=str(path))
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _call_name(node.func) == "handle_task":
                violations.append(f"{rel}:{node.lineno}")
    return violations


def test_production_has_no_direct_nexus_handle_task_bypasses() -> None:
    violations = _collect_handle_task_call_sites()
    assert violations == [], (
        "direct handle_task() bypasses outside canonical orchestration backend: "
        + ", ".join(violations)
    )


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


def _collect_strategy_resolver_usages() -> list[str]:
    violations: list[str] = []
    for path in _iter_production_python_files():
        if path == _EXECUTION_ROOT / "strategy.py":
            continue
        if path == _STRATEGY_RESOLVER_OWNER:
            continue
        tree = ast.parse(_read_python_source(path), filename=str(path))
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _call_name(node.func) == "StrategyResolver":
                violations.append(f"{rel}:{node.lineno}: StrategyResolver()")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "resolve"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "StrategyResolver"
            ):
                violations.append(f"{rel}:{node.lineno}: StrategyResolver.resolve()")
    return violations


def test_strategy_resolver_is_owned_by_canonical_router() -> None:
    violations = _collect_strategy_resolver_usages()
    assert violations == [], (
        "StrategyResolver must be used only by strategy_router.py: "
        + ", ".join(violations)
    )


def test_strategy_router_is_canonical_strategy_owner() -> None:
    path = _STRATEGY_RESOLVER_OWNER
    assert path.exists()
    source = path.read_text(encoding="utf-8")
    assert "class StrategyExecutionRouter" in source
    assert "StrategyResolver" in source
    tree = ast.parse(source, filename=str(path))
    resolve_calls = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "resolve"
    ]
    assert len(resolve_calls) == 1

    for backend in ("inference.py", "agentic.py", "orchestration.py"):
        backend_source = (_EXECUTION_ROOT / backend).read_text(encoding="utf-8")
        assert "StrategyResolver" not in backend_source, backend


def _is_strategic_backend_instantiation(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and _call_name(node.func) in _STRATEGIC_BACKEND_CLASSES


def _collect_strategic_backend_executor_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not _is_strategic_backend_instantiation(node.value):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    return names


def _is_forbidden_strategic_backend_execute_call(
    node: ast.Call,
    executor_names: set[str],
) -> bool:
    if not isinstance(node.func, ast.Attribute) or node.func.attr != "execute":
        return False
    receiver = node.func.value
    if isinstance(receiver, ast.Name) and receiver.id in executor_names:
        return True
    return (
        isinstance(receiver, ast.Attribute)
        and receiver.attr in _STRATEGIC_BACKEND_EXECUTE_ATTRS
    )


def _collect_strategic_backend_execute_bypasses() -> list[str]:
    violations: list[str] = []
    for path in _iter_production_python_files():
        if path == _STRATEGIC_BACKEND_OWNER:
            continue
        if path.name in ("agentic.py", "inference.py"):
            continue
        tree = ast.parse(_read_python_source(path), filename=str(path))
        rel = path.relative_to(_REPO_ROOT).as_posix()
        executor_names = _collect_strategic_backend_executor_names(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not _is_forbidden_strategic_backend_execute_call(node, executor_names):
                continue
            violations.append(f"{rel}:{node.lineno}")
    return violations


def test_strategic_backends_execute_only_through_canonical_router() -> None:
    violations = _collect_strategic_backend_execute_bypasses()
    assert violations == [], (
        "AgentExecutor.execute / InferenceExecutor.execute must be invoked only by "
        "strategy_router.py: "
        + ", ".join(violations)
    )

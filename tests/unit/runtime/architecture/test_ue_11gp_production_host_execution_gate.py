# © Artur Czarnecki. All rights reserved.

"""Production application hosts must execute through canonical Unified Execution."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_APPLICATIONS_ROOT = _REPO_ROOT / "applications"
_HOST_TASK_PATH = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "host_task.py"
_SHARED_MCP_EXECUTION_PATHS = (
    _REPO_ROOT / "intergrax" / "applications" / "_shared" / "mcp_nexus_server.py",
)


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _is_unified_task_runner_instantiation(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name) and func.id == "UnifiedTaskRunner":
        return True
    return isinstance(func, ast.Attribute) and func.attr == "UnifiedTaskRunner"


def _iter_application_python_files() -> list[Path]:
    paths: list[Path] = []
    for path in _APPLICATIONS_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        if "docker" in path.parts:
            continue
        if path.name.startswith("test_"):
            continue
        if path.parts[-2:] == ("tests", path.name):
            continue
        if "tests" in path.parts:
            continue
        paths.append(path)
    return paths


def _collect_root_bypass_violations(path: Path) -> list[str]:
    rel = path.relative_to(_REPO_ROOT).as_posix()

    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(path))
    violations: list[str] = []

    unified_runner_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "intergrax.runtime.task.unified_task_runner":
            for alias in node.names:
                if alias.name == "UnifiedTaskRunner":
                    violations.append(f"{rel}:{node.lineno}: import UnifiedTaskRunner")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "intergrax.runtime.task.unified_task_runner":
                    violations.append(f"{rel}:{node.lineno}: import unified_task_runner")

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _is_unified_task_runner_instantiation(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    unified_runner_names.add(target.id)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name == "handle_task":
            violations.append(f"{rel}:{node.lineno}: handle_task()")
            continue
        if name != "run_task":
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        receiver = node.func.value
        if isinstance(receiver, ast.Name) and receiver.id in unified_runner_names:
            violations.append(f"{rel}:{node.lineno}: UnifiedTaskRunner.run_task()")
        if _is_unified_task_runner_instantiation(receiver):
            violations.append(f"{rel}:{node.lineno}: UnifiedTaskRunner().run_task()")

    return violations


def _is_execution_runtime_execute_call(node: ast.Call) -> bool:
    if not isinstance(node.func, ast.Attribute) or node.func.attr != "execute":
        return False
    receiver = node.func.value
    if isinstance(receiver, ast.Name):
        return receiver.id == "runtime"
    if isinstance(receiver, ast.Call):
        return _call_name(receiver.func) == "ExecutionRuntime"
    return False


def _collect_host_task_facade_bypass_violations() -> list[str]:
    source = _HOST_TASK_PATH.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(_HOST_TASK_PATH))
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name == "RootExecutionContext":
            violations.append(f"host_task.py:{node.lineno}: RootExecutionContext()")
        if name == "resolve_root_task_identity":
            violations.append(f"host_task.py:{node.lineno}: resolve_root_task_identity()")
        if name == "mint_root_execution_identity":
            violations.append(f"host_task.py:{node.lineno}: mint_root_execution_identity()")
        if _is_execution_runtime_execute_call(node):
            violations.append(f"host_task.py:{node.lineno}: ExecutionRuntime.execute()")
    return violations


def test_production_applications_have_no_root_nexus_or_unified_task_runner_bypasses() -> None:
    violations: list[str] = []
    for path in _iter_application_python_files():
        violations.extend(_collect_root_bypass_violations(path))
    for path in _SHARED_MCP_EXECUTION_PATHS:
        violations.extend(_collect_root_bypass_violations(path))
    assert violations == [], (
        "production application hosts must not root-execute via NexusLoop or UnifiedTaskRunner: "
        + ", ".join(violations)
    )


def test_lkw_production_host_is_not_allowlisted() -> None:
    lkw_paths = {
        path.relative_to(_REPO_ROOT).as_posix()
        for path in _iter_application_python_files()
        if path.relative_to(_APPLICATIONS_ROOT).parts[:2]
        == ("local_workspace_application", "host")
        or path.relative_to(_APPLICATIONS_ROOT).parts[:2]
        == ("local_workspace_application", "serving")
    }
    assert lkw_paths


def test_governed_contractor_serving_host_is_not_allowlisted() -> None:
    governed_paths = {
        path.relative_to(_REPO_ROOT).as_posix()
        for path in _iter_application_python_files()
        if path.relative_to(_APPLICATIONS_ROOT).parts[:2]
        == ("governed_contractor_application", "serving")
    }
    assert governed_paths


def test_governed_contractor_mcp_host_is_not_allowlisted() -> None:
    governed_mcp_paths = {
        path.relative_to(_REPO_ROOT).as_posix()
        for path in _iter_application_python_files()
        if path.relative_to(_APPLICATIONS_ROOT).parts[:3]
        == ("governed_contractor_application", "mcp", "server.py")
    }
    assert governed_mcp_paths


def test_host_task_does_not_bypass_execution_facade() -> None:
    violations = _collect_host_task_facade_bypass_violations()
    assert violations == [], (
        "host task execution must route through Execution facade, not mint or call runtime directly: "
        + ", ".join(violations)
    )

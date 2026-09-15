# © Artur Czarnecki. All rights reserved.

"""GR-2-R3 mandatory MODEL C1 architecture gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.unit.runtime.architecture.gr2_r3_model_c1_ast import (
    ArchitectureViolation,
    collect_forbidden_authority_resolution_calls,
    collect_forbidden_execution_facade_imports,
    collect_forbidden_execution_runtime_imports,
    collect_forbidden_legacy_execute_root_task_imports,
    collect_forbidden_root_construction_calls,
    collect_forbidden_root_engine_execute_calls,
    collect_forbidden_unified_task_runner_imports,
)
from tests.unit.runtime.architecture.gr2_r3_model_c1_gate_policy import (
    AUTHORITY_RESOLUTION_ALLOWLIST,
    INTERNAL_ROOT_ENGINE_ALLOWLIST,
    LEGACY_EXECUTE_ROOT_TASK_IMPORT_ALLOWLIST,
    LEGACY_UNIFIED_TASK_RUNNER_IMPORT_ALLOWLIST,
    PRODUCTION_SCAN_ROOTS,
    REPO_ROOT,
    TEST_TREE_PREFIXES,
)

pytestmark = pytest.mark.unit


def _rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _iter_production_python_files() -> list[Path]:
    paths: list[Path] = []
    for root in PRODUCTION_SCAN_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            if "docker" in path.parts or "runtime-context" in path.parts:
                continue
            rel = _rel(path)
            if rel.startswith(TEST_TREE_PREFIXES):
                continue
            if "/tests/" in f"/{rel}/":
                continue
            if path.name.startswith("test_"):
                continue
            paths.append(path)
    return paths


def _parse_fixture(source: str, name: str = "fixture.py") -> tuple[Path, ast.AST, str]:
    fixture = Path(__file__).with_name(name)
    rel = _rel(fixture)
    tree = ast.parse(source, filename=str(fixture))
    return fixture, tree, rel


def _messages(violations: list[ArchitectureViolation]) -> list[str]:
    return [v.as_message() for v in violations]


def _scan_production(collector, allowlist: frozenset[str]) -> list[ArchitectureViolation]:
    found: list[ArchitectureViolation] = []
    for path in _iter_production_python_files():
        rel = _rel(path)
        if rel in allowlist:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        found.extend(collector(tree, rel_path=rel))
    return found


def test_production_has_no_unauthorized_root_construction() -> None:
    violations = _scan_production(
        collect_forbidden_root_construction_calls,
        INTERNAL_ROOT_ENGINE_ALLOWLIST,
    )
    assert violations == []


def test_production_has_no_unauthorized_root_authority_resolution() -> None:
    violations = _scan_production(
        collect_forbidden_authority_resolution_calls,
        AUTHORITY_RESOLUTION_ALLOWLIST,
    )
    assert violations == []


def test_production_has_no_unauthorized_execution_runtime_root_execute() -> None:
    violations = _scan_production(
        collect_forbidden_root_engine_execute_calls,
        INTERNAL_ROOT_ENGINE_ALLOWLIST,
    )
    runtime = [v for v in violations if v.rule == "ROOT_RUNTIME_EXECUTE"]
    assert runtime == []


def test_production_has_no_unauthorized_execution_facade_root_execute() -> None:
    violations = _scan_production(
        collect_forbidden_root_engine_execute_calls,
        INTERNAL_ROOT_ENGINE_ALLOWLIST,
    )
    facade = [v for v in violations if v.rule == "ROOT_FACADE_EXECUTE"]
    assert facade == []


def test_production_has_no_unauthorized_execution_runtime_imports() -> None:
    violations = _scan_production(
        collect_forbidden_execution_runtime_imports,
        INTERNAL_ROOT_ENGINE_ALLOWLIST,
    )
    assert violations == []


def test_production_has_no_unauthorized_execution_facade_imports() -> None:
    violations = _scan_production(
        collect_forbidden_execution_facade_imports,
        INTERNAL_ROOT_ENGINE_ALLOWLIST,
    )
    assert violations == []


def test_production_has_no_unauthorized_legacy_execute_root_task_imports() -> None:
    violations = _scan_production(
        collect_forbidden_legacy_execute_root_task_imports,
        LEGACY_EXECUTE_ROOT_TASK_IMPORT_ALLOWLIST,
    )
    assert violations == []


def test_production_has_no_unauthorized_unified_task_runner_imports() -> None:
    violations = _scan_production(
        collect_forbidden_unified_task_runner_imports,
        LEGACY_UNIFIED_TASK_RUNNER_IMPORT_ALLOWLIST,
    )
    assert violations == []


def test_gate_detects_direct_executionruntime_instance_execute() -> None:
    source = (
        "from intergrax.runtime.execution.runtime import ExecutionRuntime\n"
        "async def bypass(delegate, request, root_context):\n"
        "    runtime = ExecutionRuntime(delegate)\n"
        "    await runtime.execute(request, root_context)\n"
    )
    _, tree, rel = _parse_fixture(source)
    violations = collect_forbidden_root_engine_execute_calls(tree, rel_path=rel)
    assert violations == [
        ArchitectureViolation(rel, 4, "ROOT_RUNTIME_EXECUTE", "ExecutionRuntime.execute()")
    ]


def test_gate_detects_inline_executionruntime_execute() -> None:
    source = (
        "from intergrax.runtime.execution.runtime import ExecutionRuntime\n"
        "async def bypass(delegate, request, root_context):\n"
        "    await ExecutionRuntime(delegate).execute(request, root_context)\n"
    )
    _, tree, rel = _parse_fixture(source)
    violations = collect_forbidden_root_engine_execute_calls(tree, rel_path=rel)
    assert violations == [
        ArchitectureViolation(rel, 3, "ROOT_RUNTIME_EXECUTE", "ExecutionRuntime.execute()")
    ]


def test_gate_detects_execution_facade_instance_execute() -> None:
    source = (
        "from intergrax.runtime.execution.facade import Execution\n"
        "async def bypass(runtime, request):\n"
        "    execution = Execution(runtime)\n"
        "    await execution.execute(request, options=object())\n"
    )
    _, tree, rel = _parse_fixture(source)
    violations = collect_forbidden_root_engine_execute_calls(tree, rel_path=rel)
    assert violations == [
        ArchitectureViolation(rel, 4, "ROOT_FACADE_EXECUTE", "Execution.execute()")
    ]


def test_gate_detects_inline_execution_facade_execute() -> None:
    source = (
        "from intergrax.runtime.execution.facade import Execution\n"
        "async def bypass(runtime, request):\n"
        "    await Execution(runtime).execute(request, options=object())\n"
    )
    _, tree, rel = _parse_fixture(source)
    violations = collect_forbidden_root_engine_execute_calls(tree, rel_path=rel)
    assert violations == [
        ArchitectureViolation(rel, 3, "ROOT_FACADE_EXECUTE", "Execution.execute()")
    ]


def test_gate_detects_executionruntime_alias_instance_execute() -> None:
    source = (
        "from intergrax.runtime.execution.runtime import ExecutionRuntime as RuntimeImpl\n"
        "async def bypass(delegate, request, root_context):\n"
        "    engine = RuntimeImpl(delegate)\n"
        "    await engine.execute(request, root_context)\n"
    )
    _, tree, rel = _parse_fixture(source)
    violations = collect_forbidden_root_engine_execute_calls(tree, rel_path=rel)
    assert violations == [
        ArchitectureViolation(rel, 4, "ROOT_RUNTIME_EXECUTE", "ExecutionRuntime.execute()")
    ]


def test_gate_detects_root_execution_options_construction() -> None:
    source = "def bypass():\n    RootExecutionOptions(authority=object())\n"
    _, tree, rel = _parse_fixture(source)
    violations = collect_forbidden_root_construction_calls(tree, rel_path=rel)
    assert violations == [
        ArchitectureViolation(rel, 2, "FORBIDDEN_ROOT_CONSTRUCTION", "RootExecutionOptions()")
    ]


def test_gate_detects_direct_intake_construction() -> None:
    source = (
        "def bypass():\n"
        "    CanonicalExecutionIntakeRequest(payload=object(), trusted_parent_execution_authority=object(), tenant_id='t')\n"
    )
    _, tree, rel = _parse_fixture(source)
    violations = collect_forbidden_root_construction_calls(tree, rel_path=rel)
    assert violations == [
        ArchitectureViolation(
            rel,
            2,
            "FORBIDDEN_ROOT_CONSTRUCTION",
            "CanonicalExecutionIntakeRequest()",
        )
    ]


def test_gate_detects_authority_resolution() -> None:
    source = (
        "from intergrax.contracts.delegation_authority import resolve_root_parent_execution_authority\n"
        "def bypass(task):\n"
        "    resolve_root_parent_execution_authority(task.execution_authority)\n"
    )
    _, tree, rel = _parse_fixture(source)
    violations = collect_forbidden_authority_resolution_calls(tree, rel_path=rel)
    assert violations == [
        ArchitectureViolation(
            rel,
            3,
            "FORBIDDEN_AUTHORITY_RESOLUTION",
            "resolve_root_parent_execution_authority()",
        )
    ]


def test_gate_negative_control_unrelated_execute() -> None:
    source = (
        "class Something:\n"
        "    async def execute(self, request):\n"
        "        return request\n"
        "async def ok(agent_executor, request):\n"
        "    await agent_executor.execute(request)\n"
    )
    _, tree, rel = _parse_fixture(source)
    violations = collect_forbidden_root_engine_execute_calls(tree, rel_path=rel)
    assert violations == []


def test_gate_detects_forbidden_legacy_production_caller() -> None:
    source = (
        "from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner\n"
        "from intergrax.runtime.execution.orchestration import execute_root_task\n"
    )
    _, tree, rel = _parse_fixture(source, "synthetic_production_legacy_caller.py")
    runner_violations = collect_forbidden_unified_task_runner_imports(tree, rel_path=rel)
    root_violations = collect_forbidden_legacy_execute_root_task_imports(tree, rel_path=rel)
    assert runner_violations == [
        ArchitectureViolation(
            rel,
            1,
            "FORBIDDEN_LEGACY_UNIFIED_TASK_RUNNER_IMPORT",
            "UnifiedTaskRunner",
        )
    ]
    assert root_violations == [
        ArchitectureViolation(
            rel,
            2,
            "FORBIDDEN_LEGACY_EXECUTE_ROOT_TASK_IMPORT",
            "execute_root_task",
        )
    ]


def test_gate_allows_certified_harness_unified_task_runner_import() -> None:
    rel = "intergrax/runtime/long_running/wiring.py"
    path = REPO_ROOT / rel
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    assert rel in LEGACY_UNIFIED_TASK_RUNNER_IMPORT_ALLOWLIST
    raw = collect_forbidden_unified_task_runner_imports(tree, rel_path=rel)
    assert raw == [
        ArchitectureViolation(
            rel,
            24,
            "FORBIDDEN_LEGACY_UNIFIED_TASK_RUNNER_IMPORT",
            "UnifiedTaskRunner",
        )
    ]
    assert (
        _scan_production(
            collect_forbidden_unified_task_runner_imports,
            LEGACY_UNIFIED_TASK_RUNNER_IMPORT_ALLOWLIST,
        )
        == []
    )

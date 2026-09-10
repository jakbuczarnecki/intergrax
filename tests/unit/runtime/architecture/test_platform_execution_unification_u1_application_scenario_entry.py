# © Artur Czarnecki. All rights reserved.

"""U1 — supported application and scenario production entries resolve through canonical host execution."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_INTERGRAX_ROOT = _REPO_ROOT / "intergrax"
_APPLICATIONS_ROOT = _REPO_ROOT / "applications"
_SCENARIO_BASELINE = _INTERGRAX_ROOT / "applications" / "_shared" / "scenario_runtime_baseline.py"
_TASK_CONTROL_WIRING = _INTERGRAX_ROOT / "applications" / "_shared" / "task_control_wiring.py"
_HARNESS_HOST_RUNTIME = _INTERGRAX_ROOT / "applications" / "_shared" / "harness_host_runtime.py"
_SHARED_HOST_WIRING = (
    _INTERGRAX_ROOT / "applications" / "_shared" / "host_task_execution_wiring.py"
)

_CANONICAL_FACTORY_RESOLVERS = frozenset(
    {
        "build_environment_host_task_execution",
        "build_harness_host_runtime",
        "build_lkw_host_task_execution",
        "build_governed_contractor_host_task_execution",
    },
)

_UNIFIED_TASK_RUNNER_MODULE_SUFFIXES = frozenset(
    {
        "intergrax.runtime.task.unified_task_runner",
        "runtime.task.unified_task_runner",
    },
)


def _rel_posix(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


def _tier3_factory_paths() -> list[Path]:
    paths: list[Path] = []
    for path in _APPLICATIONS_ROOT.rglob("host/factory.py"):
        if "__pycache__" in path.parts:
            continue
        if "docker" in path.parts and "runtime-context" in path.parts:
            continue
        paths.append(path)
    return sorted(paths)


def _application_execution_wiring_paths() -> list[Path]:
    return sorted(_APPLICATIONS_ROOT.glob("*/host/execution_wiring.py"))


def _platform_proof_scenario_modules() -> list[Path]:
    return sorted(_REPO_ROOT.glob("platform_proofs/scenarios/*/application/scenario.py"))


def _production_application_py_modules() -> list[Path]:
    modules: list[Path] = []
    for path in _APPLICATIONS_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts or "tests" in path.parts:
            continue
        if "docker" in path.parts and "runtime-context" in path.parts:
            continue
        modules.append(path)
    return sorted(modules)


def _imports_unified_task_runner(path: Path) -> bool:
    text = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(text, filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module or ""
        if module in _UNIFIED_TASK_RUNNER_MODULE_SUFFIXES:
            return True
        for alias in node.names:
            if alias.name == "UnifiedTaskRunner":
                return True
    return False


def test_u1_ep02_scenario_task_uses_environment_host_execution() -> None:
    """EP-02: execute_scenario_task → build_environment_host_task_execution → host_execution.execute."""
    source = _SCENARIO_BASELINE.read_text(encoding="utf-8")
    assert "async def execute_scenario_task" in source
    assert "build_environment_host_task_execution" in source
    assert "host_execution.execute" in source


def test_u1_ep03_platform_proof_scenarios_delegate_to_execute_scenario_task() -> None:
    """EP-03: platform proof scenarios must not introduce a local scenario runner."""
    modules = _platform_proof_scenario_modules()
    assert modules, "expected platform proof scenario modules"
    for path in modules:
        source = path.read_text(encoding="utf-8")
        assert "execute_scenario_task" in source, (
            f"{_rel_posix(path)} must delegate through execute_scenario_task"
        )


def test_u1_ep04_harness_http_tasks_wire_canonical_host_executor() -> None:
    """EP-04: harness HTTP task control mounts canonical routes with HostTaskExecutionExecutor."""
    source = _TASK_CONTROL_WIRING.read_text(encoding="utf-8")
    assert "mount_canonical_harness_task_routes" in source
    assert "HostTaskExecutionExecutor(host_execution" in source
    assert "host_execution: HostTaskExecutionPort" in source


def test_u1_ep05_tier3_factories_resolve_canonical_host_execution_composition() -> None:
    """EP-05: supported application hosts build execution via shared host-task composition only."""
    factories = _tier3_factory_paths()
    assert factories, "expected Tier-3 host/factory.py modules"
    missing: list[str] = []
    for path in factories:
        source = path.read_text(encoding="utf-8")
        if not any(marker in source for marker in _CANONICAL_FACTORY_RESOLVERS):
            missing.append(_rel_posix(path))
    assert missing == [], (
        "Tier-3 factory missing canonical host execution resolver:\n" + "\n".join(missing)
    )


def test_u1_application_execution_wiring_delegates_to_shared_host_task_execution() -> None:
    """App-local execution_wiring modules must delegate to shared build_host_task_execution."""
    for path in _application_execution_wiring_paths():
        source = path.read_text(encoding="utf-8")
        assert "build_host_task_execution" in source
        assert "intergrax.applications._shared.host_task_execution_wiring" in source


def test_u1_harness_host_runtime_uses_environment_host_task_execution() -> None:
    """Harness host runtime (lab and harness apps) must not construct a parallel execution engine."""
    source = _HARNESS_HOST_RUNTIME.read_text(encoding="utf-8")
    assert "build_environment_host_task_execution" in source
    shared = _SHARED_HOST_WIRING.read_text(encoding="utf-8")
    assert "def build_environment_host_task_execution" in shared
    assert "build_host_task_execution" in shared
    assert "intergrax.runtime.execution.nexus_host_execution" in shared


def test_u1_production_application_tree_does_not_import_unified_task_runner() -> None:
    """Production application modules must not root execution on UnifiedTaskRunner."""
    violations = [
        _rel_posix(path)
        for path in _production_application_py_modules()
        if _imports_unified_task_runner(path)
    ]
    assert violations == [], (
        "UnifiedTaskRunner import in production application code:\n" + "\n".join(violations)
    )

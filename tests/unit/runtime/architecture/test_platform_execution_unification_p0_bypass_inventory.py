# © Artur Czarnecki. All rights reserved.

"""P0 — platform-wide execution bypass inventory static gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_INTERGRAX_ROOT = _REPO_ROOT / "intergrax"
_ARCH_DOC = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "architecture"
    / "PLATFORM_EXECUTION_UNIFICATION_ARCHITECTURE.md"
)
_INVENTORY_DOC = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "PLATFORM_EXECUTION_UNIFICATION_P0_BYPASS_INVENTORY.md"
)

_FROZEN_CHILD_RUNNER_IMPORTS = frozenset(
    {
        "intergrax/runtime/execution/delegated_subtask_child_port.py",
        "intergrax/runtime/execution/execution_work_port.py",
        "intergrax/runtime/nexus/execution/graph_executor.py",
        "intergrax/applications/_shared/production_agent_capability_runtime.py",
    },
)

_PRODUCTION_AGENT_CAPABILITY_RUNTIME = (
    _INTERGRAX_ROOT / "applications" / "_shared" / "production_agent_capability_runtime.py"
)
_COMPENSATION_WORKER = _INTERGRAX_ROOT / "agents" / "persistence" / "compensation_queue_worker.py"
_SCENARIO_BASELINE = _INTERGRAX_ROOT / "applications" / "_shared" / "scenario_runtime_baseline.py"
_HOST_TASK = _INTERGRAX_ROOT / "runtime" / "execution" / "host_task.py"


def _rel_posix(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


def _modules_importing_child_execution_runner() -> set[str]:
    found: set[str] = set()
    for path in _INTERGRAX_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8-sig")
        if "ChildExecutionRunner" not in text:
            continue
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "intergrax.runtime.execution.child":
                for alias in node.names:
                    if alias.name == "ChildExecutionRunner":
                        found.add(_rel_posix(path))
                        break
    return found


def test_p0_qualification_documents_present() -> None:
    assert _ARCH_DOC.is_file(), "architecture qualification doc missing"
    assert _INVENTORY_DOC.is_file(), "P0 inventory qualification doc missing"
    inventory = _INVENTORY_DOC.read_text(encoding="utf-8")
    assert "## Central inventory" in inventory
    assert "BY-01" in inventory
    assert "BY-02" in inventory


def test_p0_frozen_child_execution_runner_import_surface() -> None:
    """Only proven canonical adapters + one documented composition bypass may import ChildExecutionRunner."""
    found = _modules_importing_child_execution_runner()
    assert found == _FROZEN_CHILD_RUNNER_IMPORTS, (
        "ChildExecutionRunner import surface changed — update P0 inventory and this gate together. "
        f"found={sorted(found)} expected={sorted(_FROZEN_CHILD_RUNNER_IMPORTS)}"
    )


def test_p0_documented_direct_child_bypass_still_at_composition_default() -> None:
    """BY-01 evidence anchor — default factory must remain visible until U4 closes it."""
    source = _PRODUCTION_AGENT_CAPABILITY_RUNTIME.read_text(encoding="utf-8")
    assert "DelegatedSubtaskServiceFactory" in source
    assert "as_child_execution_port(ChildExecutionRunner" in source


def test_p0_compensation_worker_not_execution_runtime_entry() -> None:
    """BY-02 evidence anchor — worker drains tools without ExecutionRuntime admission."""
    source = _COMPENSATION_WORKER.read_text(encoding="utf-8")
    assert "drain_pending_compensation_jobs" in source
    assert "DeclarativeToolInvoker" in source
    assert "ExecutionRuntime" not in source
    assert "HostTaskExecutionPort" not in source


def test_p0_scenario_entry_uses_host_task_execution() -> None:
    source = _SCENARIO_BASELINE.read_text(encoding="utf-8")
    assert "async def execute_scenario_task" in source
    assert "build_environment_host_task_execution" in source
    assert "host_execution.execute" in source


def test_p0_host_task_routes_through_execution_facade() -> None:
    source = _HOST_TASK.read_text(encoding="utf-8")
    assert "ExecutionRuntime" in source
    assert "Execution(" in source
    assert "UnifiedTaskRunner" not in source

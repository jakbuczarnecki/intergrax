# © Artur Czarnecki. All rights reserved.

"""EE-FINAL-ARCH — single orchestration / scheduler ownership (Nexus)."""

from __future__ import annotations

import pytest

from tests.unit.runtime.architecture._ee_final_arch_facts import ARCH_MODEL, _REPO_ROOT

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_GRAPH_EXECUTOR = (
    _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "execution" / "graph_executor.py"
)
_NEXUS_LOOP = _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "nexus_loop.py"
_FROZEN_GATES = (
    "tests/unit/runtime/architecture/test_npsc4_agent_runtime_governance_gate.py",
    "tests/unit/runtime/architecture/test_npsc4_2_residual_compatibility_gate.py",
)


def test_ee_final_arch_scheduler_documentation_names_nexus_owner() -> None:
    text = ARCH_MODEL.read_text(encoding="utf-8")
    assert "Nexus" in text
    assert "GraphExecutor" in text
    assert "duplicate" in text.lower() or "single" in text.lower()


def test_ee_final_arch_graph_executor_routes_child_through_runner() -> None:
    source = _GRAPH_EXECUTOR.read_text(encoding="utf-8")
    assert "ChildExecutionRunner" in source
    assert "UnifiedTaskRunner" not in source


def test_ee_final_arch_nexus_loop_does_not_construct_second_execution_runtime() -> None:
    source = _NEXUS_LOOP.read_text(encoding="utf-8")
    assert "ExecutionRuntime(" not in source


def test_ee_final_arch_frozen_nexus_governance_gates_present() -> None:
    missing = [rel for rel in _FROZEN_GATES if not (_REPO_ROOT / rel).is_file()]
    assert missing == []

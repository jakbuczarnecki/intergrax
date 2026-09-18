# © Artur Czarnecki. All rights reserved.

"""EE-FINAL-ARCH — composition roots converge on canonical execution abstractions."""

from __future__ import annotations

import pytest

from tests.unit.runtime.architecture._ee_final_arch_facts import ARCH_MODEL, _REPO_ROOT

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_HOST_TASK = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "host_task.py"
_SCENARIO = (
    _REPO_ROOT
    / "intergrax"
    / "applications"
    / "_shared"
    / "scenario_runtime_baseline.py"
)
_CONVERGENCE_GATES = (
    "tests/unit/runtime/architecture/test_npsc3c_d_canonical_execution_engine_conformance_gate.py",
    "tests/unit/runtime/architecture/test_platform_execution_unification_u4_child_execution_closure.py",
    "tests/unit/applications/architecture/test_binding_contract_identity_authority.py",
)


def test_ee_final_arch_canonical_flow_documented() -> None:
    text = ARCH_MODEL.read_text(encoding="utf-8")
    assert "ExecutionBoundary" in text
    assert "StrategyExecutionRouter" in text


def test_ee_final_arch_host_task_and_scenario_reference_execution_runtime() -> None:
    host = _HOST_TASK.read_text(encoding="utf-8")
    scenario = _SCENARIO.read_text(encoding="utf-8")
    assert "ExecutionRuntime" in host
    assert (
        "HostTaskExecutionPort" in scenario
        or "build_harness_environment_host_task_execution" in scenario
    )


def test_ee_final_arch_convergence_gates_present() -> None:
    missing = [rel for rel in _CONVERGENCE_GATES if not (_REPO_ROOT / rel).is_file()]
    assert missing == []

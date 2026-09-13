# © Artur Czarnecki. All rights reserved.

"""DS-E2E-15J-DOCKER-E2E-SYSTEM-QUALIFICATION."""

from __future__ import annotations

import pytest

from testing_support.decision_e2e.docker_system_qualification import (
    run_docker_system_scenario,
)
from testing_support.decision_e2e.docker_system_scenarios import (
    run_docker_system_scenario as run_local_scenario,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.e2e,
    pytest.mark.docker,
    pytest.mark.qualification,
    pytest.mark.decision_e2e,
    pytest.mark.no_ci,
    pytest.mark.slow,
]

_QUALIFICATION_TASK = "DS-E2E-15J-DOCKER-E2E-SYSTEM-QUALIFICATION"


def _assert_docker_pass(scenario_id: str) -> dict:
    run = run_docker_system_scenario(scenario_id)
    if run.block_reason:
        pytest.skip(run.block_reason)
    assert run.result is not None, f"missing result for {scenario_id}"
    assert run.result.get("passed") is True, run.result
    assert run.result.get("qualification_task_id") == _QUALIFICATION_TASK
    return run.result


@pytest.mark.parametrize(
    "scenario_id",
    (
        "startup-health",
        "flow-success",
        "governance-deny",
        "governance-approval",
        "evidence-chain",
        "execution-failure",
        "missing-governance",
        "invalid-config-startup",
        "plugin-compatibility",
    ),
)
def test_docker_system_scenario_runs_in_container(scenario_id: str) -> None:
    _assert_docker_pass(scenario_id)


def test_docker_startup_qualification_gate() -> None:
    result = _assert_docker_pass("startup-health")
    assert result.get("integration_status") == "success"


def test_docker_full_success_flow() -> None:
    result = _assert_docker_pass("flow-success")
    assert result.get("governance_disposition") == "allow"
    assert result.get("execution_status") == "executed"
    assert result.get("audit_entries") == 1


def test_docker_governance_block_flow() -> None:
    result = _assert_docker_pass("governance-deny")
    assert result.get("governance_disposition") == "block"


def test_docker_approval_flow() -> None:
    result = _assert_docker_pass("governance-approval")
    assert result.get("governance_disposition") == "require_approval"
    assert result.get("stopped") is True


def test_docker_evidence_validation() -> None:
    result = _assert_docker_pass("evidence-chain")
    for key in (
        "decision_id",
        "governance_decision_id",
        "execution_result_id",
        "audit_provider_id",
        "mapping_version",
    ):
        assert result.get(key), f"missing evidence field {key}"


def test_docker_failure_scenario() -> None:
    result = _assert_docker_pass("execution-failure")
    assert result.get("failure_class") == "RuntimeError"


def test_local_scenario_parity_before_docker() -> None:
    """Fast parity check (no Docker) for qualification logic."""
    for scenario_id in (
        "startup-health",
        "flow-success",
        "governance-deny",
        "governance-approval",
        "evidence-chain",
    ):
        local = run_local_scenario(scenario_id)
        assert local.passed, local.detail

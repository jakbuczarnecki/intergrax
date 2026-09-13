# © Artur Czarnecki. All rights reserved.

"""Local parity for canonical Docker execution qualification scenarios."""

from __future__ import annotations

import pytest

from testing_support.decision_e2e.docker_system_scenarios import (
    run_docker_system_scenario,
)

pytestmark = pytest.mark.unit

_CANONICAL_SCENARIOS = (
    "canonical-execution-success",
    "canonical-governance-deny",
    "canonical-governance-approval",
    "canonical-evidence-chain",
    "canonical-execution-failure",
)


@pytest.mark.parametrize("scenario_id", _CANONICAL_SCENARIOS)
def test_canonical_docker_scenario_local_passes(scenario_id: str) -> None:
    result = run_docker_system_scenario(scenario_id)
    assert result.passed, result.detail


def test_canonical_success_does_not_use_recording_provider() -> None:
    result = run_docker_system_scenario("canonical-execution-success")
    assert result.passed, result.detail
    assert result.payload.get("recording_execution_provider_used") is False
    assert result.payload.get("execution_path") == "canonical-execution-runtime"
    assert result.payload.get("authorization_id")
    assert int(result.payload.get("execution_engine_invocations", 0)) >= 1

# © Artur Czarnecki. All rights reserved.

"""Local parity for DS-E2E-15J Docker system scenarios (no container)."""

from __future__ import annotations

import pytest

from testing_support.decision_e2e.docker_system_scenarios import (
    run_docker_system_scenario,
)

pytestmark = pytest.mark.unit


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
def test_docker_system_scenario_local_passes(scenario_id: str) -> None:
    result = run_docker_system_scenario(scenario_id)
    assert result.passed, result.detail

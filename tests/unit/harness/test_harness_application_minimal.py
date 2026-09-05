# © Artur Czarnecki. All rights reserved.

"""HarnessApplication minimal offline run (Phase DX-2.5)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from echo.echo_agent import EchoAgent
from intergrax.applications.contracts.graph_builder import AgentGraph
from intergrax.harness import HarnessApplication
from intergrax.integrations.registry.profile import IntegrationProfile

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


def test_harness_application_echo_run() -> None:
    app = (
        HarnessApplication("harness_test", route_prefix="/v1/harness_test")
        .agents(EchoAgent, contract_id="echo")
        .integrations(IntegrationProfile.lab_stack())
        .graph(AgentGraph().default(EchoAgent))
        .build_fastapi()
    )
    client = TestClient(app)
    agents = client.get("/v1/harness_test/agents")
    assert agents.status_code == 200
    run = client.post(
        "/v1/harness_test/run",
        json={"message": "hello", "capability": "echo.basic"},
    )
    assert run.status_code == 200
    assert run.json().get("state") == "completed"

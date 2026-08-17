# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from dispute_sim_application.host.factory import create_dispute_sim_backend_app
from dispute_sim_application.tests.dispute_sim_ac3_projection import (
    build_dispute_sim_test_registry_projection,
)
from testing_support.builder import MeteringFakeLLMAdapter

pytestmark = [pytest.mark.unit]

_PREFIX = "/v1/dispute_sim"


@pytest.fixture
def _stub_host_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Product host resolves env LLM (Ollama); unit smoke must stay offline."""
    adapter = MeteringFakeLLMAdapter()

    def _resolve(
        env: object,
        agent_override: object | None = None,
        **_: object,
    ) -> object:
        del env
        if agent_override is not None:
            return agent_override
        return adapter

    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.resolve_llm_adapter",
        _resolve,
    )


def test_dispute_sim_backend_health():
    client = TestClient(
        create_dispute_sim_backend_app(
            registry_projection=build_dispute_sim_test_registry_projection(),
        )
    )
    response = client.get("/health")
    assert response.status_code == 200


def test_dispute_sim_backend_lists_agents():
    client = TestClient(
        create_dispute_sim_backend_app(
            registry_projection=build_dispute_sim_test_registry_projection(),
        )
    )
    response = client.get(f"{_PREFIX}/agents")
    assert response.status_code == 200
    assert "agents" in response.json()


def test_dispute_sim_backend_run(_stub_host_llm: None):
    client = TestClient(
        create_dispute_sim_backend_app(
            registry_projection=build_dispute_sim_test_registry_projection(),
        )
    )
    response = client.post(
        f"{_PREFIX}/run",
        json={"message": "hello", "capability": "dispute.intake"},
    )
    assert response.status_code == 200
    assert response.json().get("state") == "completed"

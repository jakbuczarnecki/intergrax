# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from governed_contractor_application.host.factory import create_governed_contractor_backend_app
from testing_support.builder import MeteringFakeLLMAdapter

pytestmark = [pytest.mark.unit]

_PREFIX = "/v1/governed_contractor"


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


def test_governed_contractor_backend_health():
    client = TestClient(create_governed_contractor_backend_app())
    response = client.get("/health")
    assert response.status_code == 200


def test_governed_contractor_backend_lists_agents():
    client = TestClient(create_governed_contractor_backend_app())
    response = client.get(f"{_PREFIX}/agents")
    assert response.status_code == 200
    assert "agents" in response.json()


def test_governed_contractor_backend_run(_stub_host_llm: None):
    client = TestClient(create_governed_contractor_backend_app())
    response = client.post(
        f"{_PREFIX}/run",
        json={"message": "hello", "capability": "external_contractor.adapt"},
    )
    assert response.status_code == 200
    assert response.json().get("state") == "completed"

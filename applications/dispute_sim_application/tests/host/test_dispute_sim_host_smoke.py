# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from dispute_sim_application.host.factory import create_dispute_sim_backend_app
from dispute_sim_application.tests.dispute_sim_ac3_projection import (
    build_dispute_sim_host_test_manifest,
    build_dispute_sim_test_registry_projection,
)
from testing_support.host_fixture_wiring import (
    install_diagnostic_cursor_secret,
    install_host_llm_stub,
    reference_host_platform_persistence_kwargs,
)

pytestmark = [pytest.mark.unit]

_PREFIX = "/v1/dispute_sim"


@pytest.fixture
def _stub_host_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    install_host_llm_stub(monkeypatch)


@pytest.fixture
def _diagnostic_cursor_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    install_diagnostic_cursor_secret(monkeypatch)


@pytest.fixture
def dispute_sim_host_test_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "dispute_sim_application.host.factory.build_dispute_sim_manifest",
        lambda: build_dispute_sim_host_test_manifest(strict_platform_backing=True),
    )


def test_dispute_sim_backend_health(
    _stub_host_llm: None,
    _diagnostic_cursor_secret: None,
    dispute_sim_host_test_manifest: None,
):
    client = TestClient(
        create_dispute_sim_backend_app(
            registry_projection=build_dispute_sim_test_registry_projection(),
            **reference_host_platform_persistence_kwargs(),
        )
    )
    response = client.get("/health")
    assert response.status_code == 200


def test_dispute_sim_backend_lists_agents(
    _stub_host_llm: None,
    _diagnostic_cursor_secret: None,
    dispute_sim_host_test_manifest: None,
):
    client = TestClient(
        create_dispute_sim_backend_app(
            registry_projection=build_dispute_sim_test_registry_projection(),
            **reference_host_platform_persistence_kwargs(),
        )
    )
    response = client.get(f"{_PREFIX}/agents")
    assert response.status_code == 200
    assert "agents" in response.json()


def test_dispute_sim_backend_run(
    _stub_host_llm: None,
    _diagnostic_cursor_secret: None,
    dispute_sim_host_test_manifest: None,
):
    client = TestClient(
        create_dispute_sim_backend_app(
            registry_projection=build_dispute_sim_test_registry_projection(),
            **reference_host_platform_persistence_kwargs(),
        )
    )
    response = client.post(
        f"{_PREFIX}/run",
        json={"message": "hello", "capability": "dispute.intake"},
    )
    assert response.status_code == 200
    assert response.json().get("state") == "completed"

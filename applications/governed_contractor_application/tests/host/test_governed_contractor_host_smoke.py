# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from governed_contractor_application.host.factory import create_governed_contractor_backend_app
from governed_contractor_application.tests.governed_contractor_ac3_projection import (
    build_governed_contractor_test_registry_projection,
)
from testing_support.host_fixture_wiring import (
    install_diagnostic_cursor_secret,
    install_host_llm_stub,
    reference_host_document_store,
)

pytestmark = [pytest.mark.unit]

_PREFIX = "/v1/governed_contractor"


@pytest.fixture
def _stub_host_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    install_host_llm_stub(monkeypatch)


@pytest.fixture
def _diagnostic_cursor_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    install_diagnostic_cursor_secret(monkeypatch)


def test_governed_contractor_backend_health(
    _stub_host_llm: None,
    _diagnostic_cursor_secret: None,
):
    client = TestClient(
        create_governed_contractor_backend_app(
            registry_projection=build_governed_contractor_test_registry_projection(),
            document_store=reference_host_document_store(),
        )
    )
    response = client.get("/health")
    assert response.status_code == 200


def test_governed_contractor_backend_lists_agents(
    _stub_host_llm: None,
    _diagnostic_cursor_secret: None,
):
    client = TestClient(
        create_governed_contractor_backend_app(
            registry_projection=build_governed_contractor_test_registry_projection(),
            document_store=reference_host_document_store(),
        )
    )
    response = client.get(f"{_PREFIX}/agents")
    assert response.status_code == 200
    assert "agents" in response.json()


def test_governed_contractor_backend_run(
    _stub_host_llm: None,
    _diagnostic_cursor_secret: None,
):
    client = TestClient(
        create_governed_contractor_backend_app(
            registry_projection=build_governed_contractor_test_registry_projection(),
            document_store=reference_host_document_store(),
        )
    )
    response = client.post(
        f"{_PREFIX}/run",
        json={"message": "hello", "capability": "external_contractor.adapt"},
    )
    assert response.status_code == 200
    assert response.json().get("state") == "completed"

# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from proof_infrastructure.controlled_project_status_service.app import create_app
from proof_infrastructure.controlled_project_status_service.models import (
    ProjectBlockerStatusV1,
    ProjectBlockerV1,
    ProjectStatusControlUpdateV1,
)
from proof_infrastructure.controlled_project_status_service.seed import (
    ORION_FIXTURE_BLOCKER_ID,
    ORION_FIXTURE_PROJECT_ID,
    ORION_FIXTURE_READINESS_SCORE,
    seed_orion_fixture,
)
from proof_infrastructure.controlled_project_status_service.state import ProjectStatusStore

pytestmark = pytest.mark.unit


@pytest.fixture
def service_client() -> TestClient:
    store = ProjectStatusStore()
    seed_orion_fixture(store)
    return TestClient(create_app(store=store))


def test_seeded_status_can_be_read(service_client: TestClient) -> None:
    response = service_client.get(f"/projects/{ORION_FIXTURE_PROJECT_ID}/status")
    assert response.status_code == 200
    body = response.json()
    assert body["project_id"] == ORION_FIXTURE_PROJECT_ID
    assert body["readiness_score"] == ORION_FIXTURE_READINESS_SCORE
    assert body["blockers"][0]["id"] == ORION_FIXTURE_BLOCKER_ID
    assert body["blockers"][0]["status"] == ProjectBlockerStatusV1.OPEN.value


def test_control_endpoint_changes_blocker_open_to_closed(service_client: TestClient) -> None:
    response = service_client.put(
        f"/control/projects/{ORION_FIXTURE_PROJECT_ID}/status",
        json={
            "blockers": [
                {
                    "id": ORION_FIXTURE_BLOCKER_ID,
                    "status": ProjectBlockerStatusV1.CLOSED.value,
                }
            ]
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["blockers"][0]["status"] == ProjectBlockerStatusV1.CLOSED.value


def test_read_endpoint_reflects_new_state(service_client: TestClient) -> None:
    service_client.put(
        f"/control/projects/{ORION_FIXTURE_PROJECT_ID}/status",
        json={
            "blockers": [
                {
                    "id": ORION_FIXTURE_BLOCKER_ID,
                    "status": ProjectBlockerStatusV1.CLOSED.value,
                }
            ]
        },
    )
    response = service_client.get(f"/projects/{ORION_FIXTURE_PROJECT_ID}/status")
    assert response.status_code == 200
    assert response.json()["blockers"][0]["status"] == ProjectBlockerStatusV1.CLOSED.value


def test_request_counter_increments_only_on_read_provider_endpoint(
    service_client: TestClient,
) -> None:
    assert service_client.get("/control/request-count").json()["read_request_count"] == 0
    service_client.get(f"/projects/{ORION_FIXTURE_PROJECT_ID}/status")
    assert service_client.get("/control/request-count").json()["read_request_count"] == 1
    service_client.put(
        f"/control/projects/{ORION_FIXTURE_PROJECT_ID}/status",
        json={"status": "active"},
    )
    assert service_client.get("/control/request-count").json()["read_request_count"] == 1
    service_client.get(f"/projects/{ORION_FIXTURE_PROJECT_ID}/status")
    assert service_client.get("/control/request-count").json()["read_request_count"] == 2


def test_counter_reset_works(service_client: TestClient) -> None:
    service_client.get(f"/projects/{ORION_FIXTURE_PROJECT_ID}/status")
    assert service_client.get("/control/request-count").json()["read_request_count"] == 1
    reset = service_client.post("/control/request-count/reset")
    assert reset.status_code == 204
    assert service_client.get("/control/request-count").json()["read_request_count"] == 0


def test_invalid_project_status_payload_fails_safely(service_client: TestClient) -> None:
    invalid = service_client.put(
        f"/control/projects/{ORION_FIXTURE_PROJECT_ID}/status",
        json={
            "blockers": [
                {"id": ORION_FIXTURE_BLOCKER_ID, "status": "OPEN"},
                {"id": ORION_FIXTURE_BLOCKER_ID, "status": "CLOSED"},
            ]
        },
    )
    assert invalid.status_code == 422

    missing = service_client.get("/projects/UNKNOWN/status")
    assert missing.status_code == 404


def test_real_http_server_contract() -> None:
    from proof_infrastructure.controlled_project_status_service.lifecycle import (
        ControlledProjectStatusServer,
    )

    server = ControlledProjectStatusServer.start()
    try:
        response = httpx.get(
            f"{server.base_url}/projects/{ORION_FIXTURE_PROJECT_ID}/status",
            timeout=2.0,
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["blockers"][0]["status"] == ProjectBlockerStatusV1.OPEN.value
        assert httpx.get(
            f"{server.base_url}/control/request-count",
            timeout=2.0,
        ).json()["read_request_count"] == 1
    finally:
        server.stop()

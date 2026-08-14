# © Artur Czarnecki. All rights reserved.

"""AP-11 Agent Platform admin HTTP route tests."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from intergrax.applications._shared.agent_platform_admin_routes import (
    mount_agent_platform_admin_routes,
)
from tests.unit.agent_distribution.test_agent_platform_admin_service import (
    _DIGEST,
    _META_REF,
    _PACKAGE_ID,
    build_admin_stack,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_APP = "app-a"
_ENV = "env-prod"
_PREFIX = f"/v1/agent-platform/applications/{_APP}/environments/{_ENV}"


def _client(*, with_catalog: bool = True) -> tuple[TestClient, object]:
    stack = build_admin_stack(with_catalog=with_catalog)
    app = FastAPI()
    mount_agent_platform_admin_routes(app, admin_service=stack.service)
    return TestClient(app), stack


def _trust_payload() -> dict[str, object]:
    return {
        "qualification_status": "production_qualified",
        "package_digest": _DIGEST,
        "publisher_identity_ref": "publisher:acme",
        "source_provider_id": "builtin",
        "trust_evidence_refs": [
            {
                "evidence_id": "evidence:service:0",
                "kind": "signature_verification",
            }
        ],
    }


def _install_payload() -> dict[str, object]:
    return {
        "installation_id": "inst-1",
        "installation_slot_id": "slot-search",
        "package_identity": {
            "distribution_package_id": _PACKAGE_ID,
            "package_version": "1.0.0",
            "package_digest": _DIGEST,
        },
        "artifact_store_ref": "store://artifacts/inst-1",
        "trust_record": _trust_payload(),
        "agent_project_metadata_ref": _META_REF,
    }


def _bind_payload() -> dict[str, object]:
    return {
        "application_binding_id": "bind-search",
        "logical_agent_id": "researcher",
        "installation_slot_id": "slot-search",
        "config": {"mode": "fast"},
    }


def _build_payload(revision_id: str) -> dict[str, object]:
    return {
        "runtime_revision_id": revision_id,
        "application_release_id": "rel-1",
        "platform_version": "0.1.0",
        "python_version": "3.12",
        "source_context_root": "/tmp/src",
        "output_root": "/tmp/out",
        "application_source_root": "applications/app-a",
        "materialization_topology": "oci_image",
        "repository_declaration": {
            "application_release_id": "rel-1",
            "direct_dependencies": [],
        },
        "resolver_algorithm_id": "intergrax.test-resolver",
        "resolver_algorithm_version": "1.0.0",
    }


def _seed_enabled(client: TestClient) -> None:
    assert client.post(f"{_PREFIX}/installations", json=_install_payload()).status_code == 200
    assert client.post(f"{_PREFIX}/bindings", json=_bind_payload()).status_code == 200
    enable = client.post(
        f"{_PREFIX}/bindings/bind-search/enable",
        json={"expected_revision": 0},
    )
    assert enable.status_code == 200


def test_read_endpoint_response_contracts() -> None:
    client, _stack = _client()
    _seed_enabled(client)
    listed = client.get(f"{_PREFIX}/installations")
    assert listed.status_code == 200
    body = listed.json()
    assert body["installations"][0]["installation_id"] == "inst-1"
    bindings = client.get(f"{_PREFIX}/bindings")
    assert bindings.status_code == 200
    assert bindings.json()["bindings"][0]["logical_agent_id"] == "researcher"
    roster = client.get(f"{_PREFIX}/roster")
    assert roster.status_code == 200
    assert roster.json()["entries"][0]["effective_enablement"] is True
    serving = client.get(f"{_PREFIX}/serving")
    assert serving.status_code == 200
    assert serving.json()["traffic_serving_revision_id"] is None
    status = client.get(f"{_PREFIX}/agents/researcher/status")
    assert status.status_code == 200
    payload = status.json()
    assert payload["enabled_in_desired_state"] is True
    assert payload["included_in_active_revision"] is False
    catalog = client.get("/v1/agent-platform/catalog/agents")
    assert catalog.status_code == 200
    assert catalog.json()["entries"][0]["display_name"] == "Researcher"


def test_install_request_validation() -> None:
    client, _stack = _client()
    invalid = _install_payload()
    invalid["installation_id"] = ""
    response = client.post(f"{_PREFIX}/installations", json=invalid)
    assert response.status_code == 422


def test_bind_request_validation() -> None:
    client, _stack = _client()
    client.post(f"{_PREFIX}/installations", json=_install_payload())
    invalid = _bind_payload()
    invalid["logical_agent_id"] = ""
    response = client.post(f"{_PREFIX}/bindings", json=invalid)
    assert response.status_code == 422


def test_config_raw_secret_rejection_preserved() -> None:
    client, _stack = _client()
    client.post(f"{_PREFIX}/installations", json=_install_payload())
    payload = _bind_payload()
    payload["config"] = {"api_key": "sk-live-secret"}
    response = client.post(f"{_PREFIX}/bindings", json=payload)
    assert response.status_code == 422
    client.post(f"{_PREFIX}/bindings", json=_bind_payload())
    update = client.patch(
        f"{_PREFIX}/bindings/bind-search/config",
        json={"expected_revision": 0, "config": {"password": "hunter2"}},
    )
    assert update.status_code == 422


def test_enable_disable_routes() -> None:
    client, _stack = _client()
    client.post(f"{_PREFIX}/installations", json=_install_payload())
    client.post(f"{_PREFIX}/bindings", json=_bind_payload())
    enabled = client.post(
        f"{_PREFIX}/bindings/bind-search/enable",
        json={"expected_revision": 0},
    )
    assert enabled.status_code == 200
    assert enabled.json()["binding"]["enablement"] is True
    serving = client.get(f"{_PREFIX}/serving").json()
    assert serving["traffic_serving_revision_id"] is None
    disabled = client.post(
        f"{_PREFIX}/bindings/bind-search/disable",
        json={"expected_revision": 1},
    )
    assert disabled.status_code == 200
    assert disabled.json()["binding"]["enablement"] is False
    assert client.get(f"{_PREFIX}/serving").json()["traffic_serving_revision_id"] is None


def test_build_activate_rollback_routes() -> None:
    client, _stack = _client()
    _seed_enabled(client)
    built = client.post(f"{_PREFIX}/revisions/build", json=_build_payload("rev-17"))
    assert built.status_code == 200
    body = built.json()
    assert body["runtime_revision_id"] == "rev-17"
    assert body["revision_state"] == "validated"
    assert client.get(f"{_PREFIX}/serving").json()["traffic_serving_revision_id"] is None
    activated = client.post(
        f"{_PREFIX}/revisions/activate",
        json={
            "runtime_revision_id": "rev-17",
            "artifact_locator": body["artifact_locator"],
            "expected_artifact_digest": body["materialization_artifact_digest"],
            "expected_serving_pointer_revision": 0,
        },
    )
    assert activated.status_code == 200
    assert activated.json()["traffic_serving_revision_id"] == "rev-17"
    second = client.post(f"{_PREFIX}/revisions/build", json=_build_payload("rev-18"))
    assert second.status_code == 200
    activated_two = client.post(
        f"{_PREFIX}/revisions/activate",
        json={
            "runtime_revision_id": "rev-18",
            "artifact_locator": second.json()["artifact_locator"],
            "expected_artifact_digest": second.json()["materialization_artifact_digest"],
            "expected_serving_pointer_revision": 1,
            "expected_prior_traffic_revision_id": "rev-17",
        },
    )
    assert activated_two.status_code == 200
    rolled = client.post(
        f"{_PREFIX}/revisions/rollback",
        json={
            "expected_current_traffic_revision_id": "rev-18",
            "expected_serving_pointer_revision": 2,
            "target_runtime_revision_id": "rev-17",
        },
    )
    assert rolled.status_code == 200
    assert rolled.json()["restored_revision_id"] == "rev-17"


def test_404_missing_resource() -> None:
    client, _stack = _client()
    response = client.get(f"{_PREFIX}/installations/missing")
    assert response.status_code == 404
    missing_rev = client.get(f"{_PREFIX}/revisions/rev-missing")
    assert missing_rev.status_code == 404


def test_409_concurrency_conflict() -> None:
    client, _stack = _client()
    client.post(f"{_PREFIX}/installations", json=_install_payload())
    client.post(f"{_PREFIX}/bindings", json=_bind_payload())
    first = client.post(
        f"{_PREFIX}/bindings/bind-search/enable",
        json={"expected_revision": 0},
    )
    assert first.status_code == 200
    stale = client.post(
        f"{_PREFIX}/bindings/bind-search/enable",
        json={"expected_revision": 0},
    )
    assert stale.status_code == 409


def test_invalid_request_422() -> None:
    client, _stack = _client()
    response = client.post(
        f"{_PREFIX}/installations",
        json={"installation_id": "inst-1", "unknown_field": True},
    )
    assert response.status_code == 422


def test_admin_authorization_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERGRAX_HARNESS_API_KEY", "lab-key")
    client, _stack = _client()
    denied = client.get(f"{_PREFIX}/bindings")
    assert denied.status_code == 401
    allowed = client.get(f"{_PREFIX}/bindings", headers={"X-Api-Key": "lab-key"})
    assert allowed.status_code == 200


def test_shared_harness_auth_still_applies_to_mutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INTERGRAX_HARNESS_API_KEY", "lab-key")
    client, _stack = _client()
    denied = client.post(f"{_PREFIX}/installations", json=_install_payload())
    assert denied.status_code == 401
    allowed = client.post(
        f"{_PREFIX}/installations",
        json=_install_payload(),
        headers={"X-Api-Key": "lab-key"},
    )
    assert allowed.status_code == 200


def test_ap10_registry_projection_surface_still_importable() -> None:
    from intergrax.applications._shared.registry_projection import (
        ApplicationRegistryProjectionCoordinator,
        build_registry_projection,
    )

    assert ApplicationRegistryProjectionCoordinator is not None
    assert callable(build_registry_projection)

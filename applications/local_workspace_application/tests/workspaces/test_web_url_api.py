# © Artur Czarnecki. All rights reserved.

"""HTTP API tests for WEB_URL Knowledge Intake."""

from __future__ import annotations

import json
import uuid
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.websearch.capture.contracts import WebContentCaptureError, WebContentCaptureErrorCode
from intergrax.websearch.capture.url_policy import WebUrlAccessPolicy
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.serving.workspace_routes import mount_managed_workspace_routes
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.sync_runtime import build_managed_workspace_sync_runtime
from local_workspace_application.workspaces.sync_service import ManagedWorkspaceSyncService

pytestmark = pytest.mark.unit

_PREFIX = "/v1/local_workspace"
PUBLIC_IP = "93.184.216.34"


async def _resolve_public(_host: str) -> tuple[str, ...]:
    return (PUBLIC_IP,)


class _FakeExecutor:
    async def execute(self, task: object) -> object:
        _ = task
        return type("R", (), {"metadata": {"ingest_summary": {"used": True, "reason": "ingest_complete"}}})()


def _headers(tenant_id: str, *, idempotency: str = "url-1") -> dict[str, str]:
    return {"X-Tenant-Id": tenant_id, "Idempotency-Key": idempotency}


@pytest.fixture
def api_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    data_home = tmp_path / "data"
    data_home.mkdir()
    monkeypatch.setenv("DATA_HOME", str(data_home))
    monkeypatch.setenv("INTERGRAX_ALLOWED_READ_ROOTS", str(tmp_path / "docs"))
    (tmp_path / "docs").mkdir()
    settings = replace(LocalWorkspaceBackendSettings.from_env(), data_home=str(data_home))
    executor = _FakeExecutor()
    sync = ManagedWorkspaceSyncService(repo, executor)  # type: ignore[arg-type]
    runtime = build_managed_workspace_sync_runtime(
        document_store=store,
        sync_service=sync,
        repository=repo,
    )
    policy = WebUrlAccessPolicy(dns_resolver=_resolve_public)
    app = FastAPI()
    mount_managed_workspace_routes(
        app,
        task_executor=executor,  # type: ignore[arg-type]
        settings=settings,
        repository=repo,
        sync_runtime=runtime,
        web_url_access_policy=policy,
        web_content_capture=None,
        object_storage=None,
    )
    with TestClient(app) as client:
        yield client, repo


def _create_workspace(client: TestClient, tenant: str) -> str:
    response = client.post(
        f"{_PREFIX}/workspaces",
        headers={"X-Tenant-Id": tenant},
        json={"name": "Web"},
    )
    assert response.status_code == 201, response.text
    return response.json()["workspace_id"]


def test_web_url_api_accept_and_safe_response(api_bundle) -> None:
    client, _ = api_bundle
    tenant = f"tenant-{uuid.uuid4().hex[:8]}"
    workspace_id = _create_workspace(client, tenant)
    response = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/knowledge/web-urls",
        headers=_headers(tenant),
        json={"url": "https://example.com/docs?language=pl"},
    )
    assert response.status_code == 202, response.text
    payload = response.json()
    assert payload["workspace_id"] == workspace_id
    assert payload["safe_display_url"] == "https://example.com/docs"
    assert payload["status"] in {"accepted", "queued", "processing", "completed", "failed"}
    serialized = json.dumps(payload)
    for forbidden in ("language=pl", "canonical_private_url", "web_url_locator", PUBLIC_IP):
        assert forbidden not in serialized


def test_web_url_api_missing_idempotency_key(api_bundle) -> None:
    client, _ = api_bundle
    tenant = f"tenant-{uuid.uuid4().hex[:8]}"
    workspace_id = _create_workspace(client, tenant)
    response = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/knowledge/web-urls",
        headers={"X-Tenant-Id": tenant},
        json={"url": "https://example.com/docs"},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "idempotency_key_required"


def test_web_url_api_conflict_and_not_found(api_bundle) -> None:
    client, _ = api_bundle
    tenant = f"tenant-{uuid.uuid4().hex[:8]}"
    workspace_id = _create_workspace(client, tenant)
    first = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/knowledge/web-urls",
        headers=_headers(tenant, idempotency="k1"),
        json={"url": "https://example.com/a"},
    )
    assert first.status_code == 202
    conflict = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/knowledge/web-urls",
        headers=_headers(tenant, idempotency="k2"),
        json={"url": "https://example.com/a"},
    )
    assert conflict.status_code == 409
    assert conflict.json()["detail"] == "web_url_already_registered"

    foreign = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/knowledge/web-urls",
        headers=_headers("other-tenant", idempotency="k3"),
        json={"url": "https://example.com/a"},
    )
    assert foreign.status_code == 404


async def _resolve_private(_host: str) -> tuple[str, ...]:
    return ("10.0.0.5",)


def test_web_url_api_private_dns_blocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    data_home = tmp_path / "data"
    data_home.mkdir()
    monkeypatch.setenv("DATA_HOME", str(data_home))
    settings = replace(LocalWorkspaceBackendSettings.from_env(), data_home=str(data_home))
    executor = _FakeExecutor()
    sync = ManagedWorkspaceSyncService(repo, executor)  # type: ignore[arg-type]
    runtime = build_managed_workspace_sync_runtime(
        document_store=store,
        sync_service=sync,
        repository=repo,
    )
    policy = WebUrlAccessPolicy(dns_resolver=_resolve_private)
    app = FastAPI()
    mount_managed_workspace_routes(
        app,
        task_executor=executor,  # type: ignore[arg-type]
        settings=settings,
        repository=repo,
        sync_runtime=runtime,
        web_url_access_policy=policy,
    )
    tenant = f"tenant-{uuid.uuid4().hex[:8]}"
    with TestClient(app) as client:
        workspace_id = _create_workspace(client, tenant)
        response = client.post(
            f"{_PREFIX}/workspaces/{workspace_id}/knowledge/web-urls",
            headers=_headers(tenant),
            json={"url": "https://example.com/docs"},
        )
    assert response.status_code == 400
    assert response.json()["detail"] == "web_url_non_global_address_blocked"

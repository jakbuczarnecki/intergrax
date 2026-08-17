# © Artur Czarnecki. All rights reserved.

"""GET /sources safe list projection (LKW-WORKSPACE-CONTENTS-1A)."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from local_workspace_application.host.factory import create_local_workspace_backend_app
from local_workspace_application.tests.lkw_ac3_projection import build_lkw_test_registry_projection
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings

from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)

pytestmark = pytest.mark.unit

_PREFIX = "/v1/local_workspace"


def _unique_tenant(prefix: str = "tenant") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


@pytest.fixture
def workspace_root(tmp_path: Path) -> Path:
    root = tmp_path / "Contracts"
    root.mkdir()
    return root


@pytest.fixture
def api_client(
    workspace_root: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    store = InMemoryDocumentStore()
    data_home = workspace_root.parent / "lkw-data"
    sqlite_dir = workspace_root.parent / "sqlite"
    shadow_dir = workspace_root.parent / "shadow"
    for path in (data_home, sqlite_dir, shadow_dir):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCAL_WORKSPACE_VECTOR_STORE", "inmemory")
    monkeypatch.setenv("INTERGRAX_ALLOWED_READ_ROOTS", str(workspace_root.resolve()))
    monkeypatch.setenv("LOCAL_WORKSPACE_ENABLE_RAG", "true")
    monkeypatch.setenv("LOCAL_WORKSPACE_ENABLE_RAG_INGEST", "true")
    monkeypatch.setenv("LOCAL_WORKSPACE_INCLUDE_MCP", "false")
    monkeypatch.setenv("DATA_HOME", str(data_home))
    monkeypatch.setenv("LKW_DATA_HOME", str(data_home))
    monkeypatch.setenv("INTERGRAX_SQLITE_DATA_DIR", str(sqlite_dir))
    monkeypatch.setenv("INTERGRAX_SHADOW_ROOT", str(shadow_dir))
    monkeypatch.delenv("INTERGRAX_MONGODB_URI", raising=False)
    monkeypatch.setattr(
        "local_workspace_application.serving.workspace_routes.resolve_managed_workspace_document_store",
        lambda document_store=None: store,
    )
    settings = LocalWorkspaceBackendSettings.from_env()
    app = create_local_workspace_backend_app(registry_projection=build_lkw_test_registry_projection(settings), settings=settings)
    with TestClient(app) as client:
        yield client, store, workspace_root


def _headers(tenant_id: str) -> dict[str, str]:
    return {"X-Tenant-Id": tenant_id}


def _create_workspace(client: TestClient, tenant: str, name: str = "Case") -> str:
    created = client.post(
        f"{_PREFIX}/workspaces",
        headers=_headers(tenant),
        json={"name": name},
    )
    assert created.status_code == 201, created.text
    return created.json()["workspace_id"]


def test_list_sources_empty_workspace(api_client) -> None:
    client, _, _ = api_client
    tenant = _unique_tenant()
    workspace_id = _create_workspace(client, tenant)
    response = client.get(
        f"{_PREFIX}/workspaces/{workspace_id}/sources",
        headers=_headers(tenant),
    )
    assert response.status_code == 200
    assert response.json() == {"sources": []}


def test_list_sources_safe_projection_no_path(api_client) -> None:
    client, _, workspace_root = api_client
    tenant = _unique_tenant()
    workspace_id = _create_workspace(client, tenant)
    registered = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/sources",
        headers=_headers(tenant),
        json={
            "source_type": "local_folder",
            "path": str(workspace_root),
            "recursive": True,
        },
    )
    assert registered.status_code == 201, registered.text
    assert "path" in registered.json()

    listed = client.get(
        f"{_PREFIX}/workspaces/{workspace_id}/sources",
        headers=_headers(tenant),
    )
    assert listed.status_code == 200, listed.text
    body = listed.json()
    assert len(body["sources"]) == 1
    item = body["sources"][0]
    assert "path" not in item
    assert "locator" not in item
    assert item["label"] == "Contracts"
    assert item["source_type"] == "local_folder"
    assert item["status"] == "registered"
    assert item["workspace_id"] == workspace_id
    assert "source_id" in item
    full_path = str(workspace_root.resolve())
    dumped = listed.text
    assert full_path not in dumped
    # Parent fragments from a typical private path must not appear as the label.
    assert item["label"] != full_path


def test_list_sources_unknown_and_cross_tenant_404(api_client) -> None:
    client, _, workspace_root = api_client
    tenant_a = _unique_tenant("tenant-a")
    tenant_b = _unique_tenant("tenant-b")
    workspace_id = _create_workspace(client, tenant_a)
    client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/sources",
        headers=_headers(tenant_a),
        json={"source_type": "local_folder", "path": str(workspace_root)},
    )

    unknown = client.get(
        f"{_PREFIX}/workspaces/does-not-exist/sources",
        headers=_headers(tenant_a),
    )
    cross = client.get(
        f"{_PREFIX}/workspaces/{workspace_id}/sources",
        headers=_headers(tenant_b),
    )
    assert unknown.status_code == 404
    assert cross.status_code == 404
    assert unknown.json() == cross.json()


def test_register_source_still_returns_detailed_path(api_client) -> None:
    client, _, workspace_root = api_client
    tenant = _unique_tenant()
    workspace_id = _create_workspace(client, tenant)
    registered = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/sources",
        headers=_headers(tenant),
        json={"source_type": "local_folder", "path": str(workspace_root)},
    )
    assert registered.status_code == 201
    body = registered.json()
    assert body["path"] == str(workspace_root.resolve())
    assert body["status"] == "registered"

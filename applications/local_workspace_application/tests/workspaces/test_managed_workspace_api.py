# © Artur Czarnecki. All rights reserved.

"""API tests for managed workspace product endpoints (LKW-PRODUCT-1)."""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from local_workspace_application.host.factory import create_local_workspace_backend_app
from local_workspace_application.tests.lkw_ac3_projection import build_lkw_test_registry_projection
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository

pytestmark = pytest.mark.unit

_PREFIX = "/v1/local_workspace"


def _unique_tenant(prefix: str = "tenant") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


@pytest.fixture
def workspace_root(tmp_path: Path) -> Path:
    root = tmp_path / "user_docs"
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


def _wait_operation(
    client: TestClient,
    operation_id: str,
    *,
    tenant_id: str,
    timeout_seconds: float = 60.0,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, object] = {}
    while time.monotonic() < deadline:
        response = client.get(
            f"{_PREFIX}/operations/{operation_id}",
            headers=_headers(tenant_id),
        )
        assert response.status_code == 200, response.text
        last = response.json()
        if last.get("status") in {"completed", "failed"}:
            return last
        time.sleep(0.25)
    raise AssertionError(f"operation did not finish: {last}")


def test_create_list_get_workspace(api_client) -> None:
    client, _, _ = api_client
    tenant = _unique_tenant("tenant-a")
    created = client.post(
        f"{_PREFIX}/workspaces",
        headers=_headers(tenant),
        json={"name": "Buildlogic Legal Case", "description": "Documents"},
    )
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["name"] == "Buildlogic Legal Case"
    assert body["status"] == "active"
    assert body["tenant_id"] == tenant
    workspace_id = body["workspace_id"]

    listed = client.get(f"{_PREFIX}/workspaces", headers=_headers(tenant))
    assert listed.status_code == 200
    assert any(item["workspace_id"] == workspace_id for item in listed.json()["workspaces"])

    fetched = client.get(
        f"{_PREFIX}/workspaces/{workspace_id}",
        headers=_headers(tenant),
    )
    assert fetched.status_code == 200
    assert fetched.json()["workspace_id"] == workspace_id


def test_tenant_isolation_workspace_404(api_client) -> None:
    client, _, _ = api_client
    tenant_a = _unique_tenant("tenant-a")
    tenant_b = _unique_tenant("tenant-b")
    created = client.post(
        f"{_PREFIX}/workspaces",
        headers=_headers(tenant_a),
        json={"name": "Private"},
    )
    workspace_id = created.json()["workspace_id"]
    response = client.get(
        f"{_PREFIX}/workspaces/{workspace_id}",
        headers=_headers(tenant_b),
    )
    assert response.status_code == 404


def test_register_source_and_reject_invalid_path(api_client) -> None:
    client, _, workspace_root = api_client
    tenant = _unique_tenant("tenant-a")
    created = client.post(
        f"{_PREFIX}/workspaces",
        headers=_headers(tenant),
        json={"name": "Case"},
    )
    workspace_id = created.json()["workspace_id"]

    ok = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/sources",
        headers=_headers(tenant),
        json={"source_type": "local_folder", "path": str(workspace_root), "recursive": True},
    )
    assert ok.status_code == 201, ok.text
    assert ok.json()["status"] == "registered"

    bad = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/sources",
        headers=_headers(tenant),
        json={
            "source_type": "local_folder",
            "path": str(workspace_root.parent / "outside"),
            "recursive": True,
        },
    )
    assert bad.status_code == 400


def test_nonexistent_workspace_and_source_404(api_client) -> None:
    client, _, workspace_root = api_client
    tenant = _unique_tenant("tenant-a")
    missing_ws = client.post(
        f"{_PREFIX}/workspaces/does-not-exist/sources",
        headers=_headers(tenant),
        json={"source_type": "local_folder", "path": str(workspace_root)},
    )
    assert missing_ws.status_code == 404

    created = client.post(
        f"{_PREFIX}/workspaces",
        headers=_headers(tenant),
        json={"name": "Case"},
    )
    workspace_id = created.json()["workspace_id"]
    missing_source = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/sources/missing/sync",
        headers=_headers(tenant),
    )
    assert missing_source.status_code == 404


def test_sync_search_idempotency_and_workspace_isolation(api_client) -> None:
    client, _, workspace_root = api_client
    tenant = _unique_tenant("tenant-a")
    tenant_b = _unique_tenant("tenant-b")
    marker_a = f"LKW_MANAGED_MARKER_ALPHA_{uuid.uuid4().hex[:8]}"
    marker_b = f"LKW_MANAGED_MARKER_BETA_{uuid.uuid4().hex[:8]}"
    file_a = workspace_root / "invoice-alpha.txt"
    file_b = workspace_root / "payment-beta.txt"
    file_a.write_text(f"Payment obligations include {marker_a}", encoding="utf-8")
    file_b.write_text(f"Outstanding invoices mention {marker_b}", encoding="utf-8")
    original_mtime_a = file_a.stat().st_mtime_ns
    original_mtime_b = file_b.stat().st_mtime_ns

    ws1 = client.post(
        f"{_PREFIX}/workspaces",
        headers=_headers(tenant),
        json={"name": "Workspace One"},
    ).json()
    source = client.post(
        f"{_PREFIX}/workspaces/{ws1['workspace_id']}/sources",
        headers=_headers(tenant),
        json={"source_type": "local_folder", "path": str(workspace_root), "recursive": True},
    ).json()

    sync = client.post(
        f"{_PREFIX}/workspaces/{ws1['workspace_id']}/sources/{source['source_id']}/sync",
        headers=_headers(tenant),
    )
    assert sync.status_code == 202, sync.text
    assert sync.json()["status"] == "queued"
    operation = _wait_operation(client, sync.json()["operation_id"], tenant_id=tenant)
    assert operation["status"] == "completed", operation
    assert operation["files_discovered"] >= 2
    assert operation["documents_indexed"] >= 2

    search = client.post(
        f"{_PREFIX}/workspaces/{ws1['workspace_id']}/search",
        headers=_headers(tenant),
        json={"query": marker_a, "limit": 10},
    )
    assert search.status_code == 200, search.text
    results = search.json()["results"]
    assert results, search.json()
    hit = results[0]
    assert hit["workspace_id"] == ws1["workspace_id"]
    assert hit["source_id"] == source["source_id"]
    assert hit["source_path"]
    assert hit["file_name"]

    search_b = client.post(
        f"{_PREFIX}/workspaces/{ws1['workspace_id']}/search",
        headers=_headers(tenant),
        json={"query": marker_b, "limit": 10},
    )
    assert search_b.status_code == 200, search_b.text
    results_b = search_b.json()["results"]
    assert results_b, search_b.json()
    hit_b = next(
        (item for item in results_b if marker_b in item.get("snippet", "")),
        None,
    )
    assert hit_b is not None, search_b.json()
    assert hit_b["source_id"] == source["source_id"]
    assert hit_b["workspace_id"] == ws1["workspace_id"]

    semantic_a = client.post(
        f"{_PREFIX}/workspaces/{ws1['workspace_id']}/search",
        headers=_headers(tenant),
        json={"query": "payment obligations", "limit": 10},
    )
    assert semantic_a.status_code == 200, semantic_a.text
    assert any(
        item.get("file_name") == "invoice-alpha.txt"
        for item in semantic_a.json()["results"]
    )
    semantic_b = client.post(
        f"{_PREFIX}/workspaces/{ws1['workspace_id']}/search",
        headers=_headers(tenant),
        json={"query": "outstanding invoices", "limit": 10},
    )
    assert semantic_b.status_code == 200, semantic_b.text
    assert any(
        item.get("file_name") == "payment-beta.txt"
        for item in semantic_b.json()["results"]
    )

    second = client.post(
        f"{_PREFIX}/workspaces/{ws1['workspace_id']}/sources/{source['source_id']}/sync",
        headers=_headers(tenant),
    )
    second_op = _wait_operation(client, second.json()["operation_id"], tenant_id=tenant)
    assert second_op["status"] == "completed", second_op
    assert second_op["documents_indexed"] == 0
    assert second_op["documents_unchanged"] >= 2

    ws2 = client.post(
        f"{_PREFIX}/workspaces",
        headers=_headers(tenant),
        json={"name": "Workspace Two"},
    ).json()
    isolated = client.post(
        f"{_PREFIX}/workspaces/{ws2['workspace_id']}/search",
        headers=_headers(tenant),
        json={"query": marker_a, "limit": 10},
    )
    assert isolated.status_code == 200
    assert isolated.json()["results"] == []

    foreign_op = client.get(
        f"{_PREFIX}/operations/{operation['operation_id']}",
        headers=_headers(tenant_b),
    )
    assert foreign_op.status_code == 404

    assert file_a.stat().st_mtime_ns == original_mtime_a
    assert file_b.stat().st_mtime_ns == original_mtime_b
    assert file_a.read_text(encoding="utf-8").endswith(marker_a)
    assert file_b.read_text(encoding="utf-8").endswith(marker_b)


def test_local_folder_search_returns_verified_document_and_isolates_workspace(
    api_client,
) -> None:
    client, store, workspace_root = api_client
    tenant = _unique_tenant("tenant")
    marker = f"LKW_VERIFIED_LOCAL_MARKER_{uuid.uuid4().hex[:8]}"
    path = workspace_root / "verified-local.txt"
    path.write_text(f"Verified local evidence: {marker}", encoding="utf-8")

    workspace_a = client.post(
        f"{_PREFIX}/workspaces",
        headers=_headers(tenant),
        json={"name": "Workspace A"},
    ).json()
    source = client.post(
        f"{_PREFIX}/workspaces/{workspace_a['workspace_id']}/sources",
        headers=_headers(tenant),
        json={"source_type": "local_folder", "path": str(workspace_root), "recursive": True},
    ).json()
    accepted = client.post(
        f"{_PREFIX}/workspaces/{workspace_a['workspace_id']}/sources/{source['source_id']}/sync",
        headers=_headers(tenant),
    )
    operation = _wait_operation(
        client,
        accepted.json()["operation_id"],
        tenant_id=tenant,
    )
    assert operation["status"] == "completed", operation

    refs = ManagedWorkspaceRepository(store).list_document_refs(
        tenant_id=tenant,
        workspace_id=workspace_a["workspace_id"],
    )
    expected_ref = next(ref for ref in refs if ref.source_path == str(path.resolve()).replace("\\", "/"))

    visible = client.post(
        f"{_PREFIX}/workspaces/{workspace_a['workspace_id']}/search",
        headers=_headers(tenant),
        json={"query": marker, "limit": 10},
    )
    assert visible.status_code == 200, visible.text
    matching = [
        hit for hit in visible.json()["results"] if marker in hit["snippet"]
    ]
    assert matching, visible.json()
    hit = matching[0]
    assert hit["document_id"] == expected_ref.document_id
    assert hit["source_id"] == source["source_id"]
    assert hit["source_path"] == expected_ref.source_path
    assert hit["workspace_id"] == workspace_a["workspace_id"]

    foreign_tenant = client.post(
        f"{_PREFIX}/workspaces/{workspace_a['workspace_id']}/search",
        headers=_headers(_unique_tenant("other")),
        json={"query": marker, "limit": 10},
    )
    assert foreign_tenant.status_code == 404

    workspace_b = client.post(
        f"{_PREFIX}/workspaces",
        headers=_headers(tenant),
        json={"name": "Workspace B"},
    ).json()
    isolated = client.post(
        f"{_PREFIX}/workspaces/{workspace_b['workspace_id']}/search",
        headers=_headers(tenant),
        json={"query": marker, "limit": 10},
    )
    assert isolated.status_code == 200, isolated.text
    assert isolated.json()["results"] == []

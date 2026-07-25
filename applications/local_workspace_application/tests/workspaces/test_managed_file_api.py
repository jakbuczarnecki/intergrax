# © Artur Czarnecki. All rights reserved.

"""HTTP API tests for managed-file Knowledge Intake."""

from __future__ import annotations

import uuid
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.object_storage import StoredObject
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.serving.workspace_routes import mount_managed_workspace_routes
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.sync_runtime import build_managed_workspace_sync_runtime
from local_workspace_application.workspaces.sync_service import ManagedWorkspaceSyncService

pytestmark = pytest.mark.unit

_PREFIX = "/v1/local_workspace"


class FakeObjectStorage:
    def __init__(self) -> None:
        self.objects: dict[str, StoredObject] = {}

    def put(
        self,
        key: str,
        body: bytes,
        *,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> None:
        self.objects[key] = StoredObject(
            key=key,
            body=body,
            content_type=content_type,
            metadata=dict(metadata or {}),
            size_bytes=len(body),
        )

    def get(self, key: str) -> StoredObject | None:
        return self.objects.get(key)

    def delete(self, key: str) -> None:
        self.objects.pop(key, None)

    def presigned_url(self, key: str, *, expires_in_seconds: int = 3600, method: str = "GET") -> str:
        _ = expires_in_seconds, method
        return f"memory://{key}"

    def close(self) -> None:
        return None


class _FakeExecutor:
    async def execute(self, task: object) -> object:
        _ = task
        return type(
            "R",
            (),
            {
                "metadata": {
                    "ingest_summary": {
                        "used": True,
                        "reason": "ingest_complete",
                        "num_chunks": 1,
                    }
                }
            },
        )()


def _headers(tenant_id: str, *, idempotency: str | None = "batch-1") -> dict[str, str]:
    headers = {"X-Tenant-Id": tenant_id}
    if idempotency is not None:
        headers["Idempotency-Key"] = idempotency
    return headers


def _assert_safe_response(payload: dict[str, object]) -> None:
    text = str(payload)
    for forbidden in (
        "storage_key",
        "object_id",
        "s3://",
        "presigned",
        "sha256:",
        "managed_files",
        "managed_upload_staging",
    ):
        assert forbidden not in text


@pytest.fixture
def api_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    storage = FakeObjectStorage()
    data_home = tmp_path / "data"
    data_home.mkdir()
    monkeypatch.setenv("DATA_HOME", str(data_home))
    monkeypatch.setenv("INTERGRAX_ALLOWED_READ_ROOTS", str(tmp_path / "docs"))
    (tmp_path / "docs").mkdir()
    settings = replace(
        LocalWorkspaceBackendSettings.from_env(),
        managed_file_max_bytes=1024,
        managed_file_max_batch_files=3,
        data_home=str(data_home),
    )
    executor = _FakeExecutor()
    sync = ManagedWorkspaceSyncService(repo, executor)  # type: ignore[arg-type]
    runtime = build_managed_workspace_sync_runtime(
        document_store=store,
        sync_service=sync,
        repository=repo,
    )
    app = FastAPI()
    mount_managed_workspace_routes(
        app,
        task_executor=executor,  # type: ignore[arg-type]
        settings=settings,
        repository=repo,
        sync_runtime=runtime,
        object_storage=storage,
    )
    with TestClient(app) as client:
        yield client, storage, settings


def _create_workspace(client: TestClient, tenant: str) -> str:
    response = client.post(
        f"{_PREFIX}/workspaces",
        headers=_headers(tenant, idempotency=None),
        json={"name": "Docs"},
    )
    assert response.status_code == 201, response.text
    return response.json()["workspace_id"]


def test_single_and_multiple_multipart(api_bundle) -> None:
    client, _, _ = api_bundle
    tenant = f"t-{uuid.uuid4().hex[:8]}"
    workspace_id = _create_workspace(client, tenant)

    single = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/knowledge/files",
        headers=_headers(tenant, idempotency="single-1"),
        files=[("files", ("a.pdf", b"%PDF-a", "application/pdf"))],
    )
    assert single.status_code == 202, single.text
    body = single.json()
    assert body["status"] == "accepted"
    assert body["accepted_count"] == 1
    _assert_safe_response(body)

    multi = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/knowledge/files",
        headers=_headers(tenant, idempotency="multi-1"),
        files=[
            ("files", ("a.pdf", b"%PDF-a", "application/pdf")),
            ("files", ("b.pdf", b"%PDF-b", "application/pdf")),
        ],
    )
    assert multi.status_code == 202, multi.text
    assert multi.json()["accepted_count"] == 2
    _assert_safe_response(multi.json())


def test_idempotency_retry_and_conflict(api_bundle) -> None:
    client, _, _ = api_bundle
    tenant = f"t-{uuid.uuid4().hex[:8]}"
    workspace_id = _create_workspace(client, tenant)
    files = [("files", ("a.pdf", b"%PDF-a", "application/pdf"))]
    first = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/knowledge/files",
        headers=_headers(tenant, idempotency="same"),
        files=files,
    )
    second = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/knowledge/files",
        headers=_headers(tenant, idempotency="same"),
        files=files,
    )
    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["batch_id"] == second.json()["batch_id"]
    assert first.json()["items"][0]["input_id"] == second.json()["items"][0]["input_id"]

    conflict = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/knowledge/files",
        headers=_headers(tenant, idempotency="same"),
        files=[("files", ("a.pdf", b"%PDF-CHANGED", "application/pdf"))],
    )
    assert conflict.status_code == 409


def test_missing_key_unknown_workspace_cross_tenant(api_bundle) -> None:
    client, _, _ = api_bundle
    tenant = f"t-{uuid.uuid4().hex[:8]}"
    workspace_id = _create_workspace(client, tenant)
    missing = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/knowledge/files",
        headers=_headers(tenant, idempotency=None),
        files=[("files", ("a.pdf", b"x", "application/pdf"))],
    )
    assert missing.status_code == 400

    unknown = client.post(
        f"{_PREFIX}/workspaces/does-not-exist/knowledge/files",
        headers=_headers(tenant, idempotency="u1"),
        files=[("files", ("a.pdf", b"x", "application/pdf"))],
    )
    assert unknown.status_code == 404

    other = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/knowledge/files",
        headers=_headers("other-tenant", idempotency="x1"),
        files=[("files", ("a.pdf", b"x", "application/pdf"))],
    )
    assert other.status_code == 404


def test_too_many_files_and_storage_unavailable(api_bundle, tmp_path: Path, monkeypatch) -> None:
    client, _, settings = api_bundle
    tenant = f"t-{uuid.uuid4().hex[:8]}"
    workspace_id = _create_workspace(client, tenant)
    files = [
        ("files", (f"{i}.pdf", b"x", "application/pdf"))
        for i in range(settings.managed_file_max_batch_files + 1)
    ]
    too_many = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/knowledge/files",
        headers=_headers(tenant, idempotency="many"),
        files=files,
    )
    assert too_many.status_code == 413

    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    data_home = tmp_path / "data2"
    data_home.mkdir()
    monkeypatch.setenv("DATA_HOME", str(data_home))
    settings2 = LocalWorkspaceBackendSettings.from_env()
    app = FastAPI()
    executor = _FakeExecutor()
    sync = ManagedWorkspaceSyncService(repo, executor)  # type: ignore[arg-type]
    runtime = build_managed_workspace_sync_runtime(
        document_store=store,
        sync_service=sync,
        repository=repo,
    )
    mount_managed_workspace_routes(
        app,
        task_executor=executor,  # type: ignore[arg-type]
        settings=settings2,
        repository=repo,
        sync_runtime=runtime,
        object_storage=None,
    )
    with TestClient(app) as bare:
        created = bare.post(
            f"{_PREFIX}/workspaces",
            headers=_headers(tenant, idempotency=None),
            json={"name": "Docs"},
        )
        assert created.status_code == 201
        ws = created.json()["workspace_id"]
        unavailable = bare.post(
            f"{_PREFIX}/workspaces/{ws}/knowledge/files",
            headers=_headers(tenant, idempotency="no-store"),
            files=[("files", ("a.pdf", b"x", "application/pdf"))],
        )
        assert unavailable.status_code == 503
        assert unavailable.json()["detail"] == "managed_file_storage_unavailable"


def test_partial_and_all_failed(api_bundle) -> None:
    client, _, _ = api_bundle
    tenant = f"t-{uuid.uuid4().hex[:8]}"
    workspace_id = _create_workspace(client, tenant)
    partial = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/knowledge/files",
        headers=_headers(tenant, idempotency="partial"),
        files=[
            ("files", ("ok.pdf", b"ok", "application/pdf")),
            ("files", ("bad.pdf", b"", "application/pdf")),
        ],
    )
    assert partial.status_code == 202
    body = partial.json()
    assert body["status"] == "partial"
    assert body["accepted_count"] == 1
    assert body["failed_count"] == 1
    _assert_safe_response(body)

    failed = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/knowledge/files",
        headers=_headers(tenant, idempotency="all-fail"),
        files=[
            ("files", ("bad1.pdf", b"", "application/pdf")),
            ("files", ("bad2.pdf", b"", "application/pdf")),
        ],
    )
    assert failed.status_code == 202
    assert failed.json()["status"] == "failed"
    _assert_safe_response(failed.json())

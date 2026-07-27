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
from local_workspace_application.serving.workspace_routes import (
    _prepare_managed_file_batch_candidate,
    mount_managed_workspace_routes,
)
from local_workspace_application.workspaces.managed_files import (
    IntakeBatchIdempotencyConflict,
    ManagedFileIntakeService,
    managed_file_request_fingerprint,
)
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


def test_missing_and_unsafe_filename_api(api_bundle) -> None:
    client, _, _ = api_bundle
    tenant = f"t-{uuid.uuid4().hex[:8]}"
    workspace_id = _create_workspace(client, tenant)

    missing = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/knowledge/files",
        headers=_headers(tenant, idempotency="missing-name"),
        files=[("files", ("   ", b"x", "application/pdf"))],
    )
    assert missing.status_code == 202
    body = missing.json()
    assert body["status"] == "failed"
    assert body["items"][0]["error_code"] == "managed_file_name_required"
    assert body["items"][0]["file_name"] == "rejected-item-0.bin"
    assert "unnamed.bin" not in str(body)
    _assert_safe_response(body)

    unsafe_name = "evil/../secret.pdf"
    unsafe = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/knowledge/files",
        headers=_headers(tenant, idempotency="unsafe-name"),
        files=[("files", (unsafe_name, b"x", "application/pdf"))],
    )
    assert unsafe.status_code == 202
    unsafe_body = unsafe.json()
    assert unsafe_body["items"][0]["file_name"] == "rejected-item-0.bin"
    assert unsafe_body["items"][0]["error_code"] == "managed_file_name_unsafe"
    assert unsafe_name not in str(unsafe_body)
    assert "evil" not in str(unsafe_body)
    assert "secret.pdf" not in str(unsafe_body)
    _assert_safe_response(unsafe_body)


def test_empty_failed_retry_and_conflict_api(api_bundle) -> None:
    client, _, _ = api_bundle
    tenant = f"t-{uuid.uuid4().hex[:8]}"
    workspace_id = _create_workspace(client, tenant)
    empty_files = [("files", ("a.pdf", b"", "application/pdf"))]
    first = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/knowledge/files",
        headers=_headers(tenant, idempotency="empty-retry"),
        files=empty_files,
    )
    second = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/knowledge/files",
        headers=_headers(tenant, idempotency="empty-retry"),
        files=empty_files,
    )
    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["batch_id"] == second.json()["batch_id"]
    assert first.json()["items"][0]["error_code"] == "managed_file_empty"

    changed = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/knowledge/files",
        headers=_headers(tenant, idempotency="empty-retry"),
        files=[("files", ("a.pdf", b"%PDF-nonempty", "application/pdf"))],
    )
    assert changed.status_code == 409
    assert changed.json()["detail"] == "intake_batch_idempotency_conflict"

    ctype = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/knowledge/files",
        headers=_headers(tenant, idempotency="ctype-api"),
        files=[("files", ("a.pdf", b"abc", "application/pdf"))],
    )
    assert ctype.status_code == 202
    ctype_conflict = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/knowledge/files",
        headers=_headers(tenant, idempotency="ctype-api"),
        files=[("files", ("a.pdf", b"abc", "text/plain"))],
    )
    assert ctype_conflict.status_code == 409


def test_unexpected_internal_error_safe_api(api_bundle, monkeypatch) -> None:
    client, _, _ = api_bundle
    tenant = f"t-{uuid.uuid4().hex[:8]}"
    workspace_id = _create_workspace(client, tenant)

    from local_workspace_application.workspaces import managed_files as mf_mod

    original = mf_mod.ManagedFileIntakeService.accept_one

    def boom(self, **kwargs):  # type: ignore[no-untyped-def]
        _ = self, kwargs
        raise RuntimeError("s3://private-bucket/key credential=secret")

    monkeypatch.setattr(mf_mod.ManagedFileIntakeService, "accept_one", boom)
    response = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/knowledge/files",
        headers=_headers(tenant, idempotency="boom-api"),
        files=[("files", ("a.pdf", b"%PDF", "application/pdf"))],
    )
    assert response.status_code == 202
    body = response.json()
    assert body["items"][0]["error_code"] == "managed_file_accept_failed"
    text = str(body)
    for forbidden in ("s3://", "private-bucket", "credential", "secret", "RuntimeError"):
        assert forbidden not in text
    _assert_safe_response(body)
    _ = original


class _PartialFailUpload:
    def __init__(self) -> None:
        self.filename = "a.pdf"
        self.content_type = "application/pdf"
        self._reads = 0

    async def read(self, size: int = -1) -> bytes:
        _ = size
        self._reads += 1
        if self._reads == 1:
            return b"abc"
        raise RuntimeError("stream broken token=secret-path=/tmp/x")

    async def close(self) -> None:
        return None


class _CloseFailUpload:
    def __init__(self, body: bytes) -> None:
        self.filename = "a.pdf"
        self.content_type = "application/pdf"
        self._body = body
        self._done = False

    async def read(self, size: int = -1) -> bytes:
        _ = size
        if self._done:
            return b""
        self._done = True
        return self._body

    async def close(self) -> None:
        raise RuntimeError("close failed token=secret")


class _CompleteUpload:
    def __init__(
        self,
        body: bytes,
        *,
        filename: str = "a.pdf",
        content_type: str = "application/pdf",
    ) -> None:
        self.filename = filename
        self.content_type = content_type
        self._body = body
        self._done = False

    async def read(self, size: int = -1) -> bytes:
        _ = size
        if self._done:
            return b""
        self._done = True
        return self._body

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_upload_read_failure_bound_into_fingerprint() -> None:
    from datetime import UTC, datetime

    from intergrax.queueing.providers.document_store import DocumentStoreTaskQueue
    from intergrax.tools.registry.wiring import ToolWiringContext
    from local_workspace_application.workspaces.knowledge_intake import KnowledgeIntakeService
    from local_workspace_application.workspaces.managed_files import ManagedFileSourceResolver
    from local_workspace_application.workspaces.models import Workspace, WorkspaceStatus

    failed = await _prepare_managed_file_batch_candidate(
        _PartialFailUpload(),  # type: ignore[arg-type]
        max_bytes=1024,
    )
    assert failed.preflight_error_code == "managed_file_upload_read_failed"
    assert failed.body is None
    assert "secret" not in failed.request_fingerprint
    assert "token" not in str(failed)
    expected_failed = managed_file_request_fingerprint(
        raw_file_name="a.pdf",
        raw_content_type="application/pdf",
        size_bytes=3,
        body_hash=failed.body_hash,
        request_state="read_failed",
    )
    assert failed.request_fingerprint == expected_failed

    complete = await _prepare_managed_file_batch_candidate(
        _CompleteUpload(b"abc"),  # type: ignore[arg-type]
        max_bytes=1024,
    )
    assert failed.request_fingerprint != complete.request_fingerprint

    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    repo.put_workspace(
        Workspace(
            workspace_id="ws",
            tenant_id="t",
            name="Demo",
            status=WorkspaceStatus.ACTIVE,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
    )
    queue = DocumentStoreTaskQueue(store)
    intake = KnowledgeIntakeService(
        repo,
        ManagedFileSourceResolver(repo),
        ToolWiringContext(message_bus=queue),
    )
    managed = ManagedFileIntakeService(
        repo,
        FakeObjectStorage(),
        intake,
        max_bytes=1024,
        max_batch_files=20,
    )
    managed.accept_prepared_many(
        tenant_id="t",
        workspace_id="ws",
        idempotency_key="read-retry",
        candidates=[failed],
    )
    with pytest.raises(IntakeBatchIdempotencyConflict):
        managed.accept_prepared_many(
            tenant_id="t",
            workspace_id="ws",
            idempotency_key="read-retry",
            candidates=[complete],
        )


@pytest.mark.asyncio
async def test_close_failure_becomes_safe_rejected_candidate() -> None:
    failed = await _prepare_managed_file_batch_candidate(
        _CloseFailUpload(b"abc"),  # type: ignore[arg-type]
        max_bytes=1024,
    )
    assert failed.preflight_error_code == "managed_file_upload_read_failed"
    assert failed.body is None
    complete = await _prepare_managed_file_batch_candidate(
        _CompleteUpload(b"abc"),  # type: ignore[arg-type]
        max_bytes=1024,
    )
    assert failed.request_fingerprint != complete.request_fingerprint
    assert "secret" not in str(failed)

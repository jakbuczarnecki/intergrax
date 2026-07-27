# © Artur Czarnecki. All rights reserved.

"""HTTP API tests for preconfigured Source Candidates."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.queueing.contracts.task_queue import TaskHandle, TaskQueue, TaskRequest, TaskResult, TaskStatus
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.serving.workspace_routes import mount_managed_workspace_routes
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.sync_runtime import build_managed_workspace_sync_runtime
from local_workspace_application.workspaces.sync_service import ManagedWorkspaceSyncService

pytestmark = pytest.mark.unit

_PREFIX = "/v1/local_workspace"


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


class _RaisingBus(TaskQueue):
    def enqueue(self, request: TaskRequest) -> TaskHandle:
        _ = request
        raise RuntimeError("bus down")

    def get_status(self, handle: TaskHandle) -> TaskStatus:
        _ = handle
        return TaskStatus.PENDING

    def get_result(self, handle: TaskHandle) -> TaskResult | None:
        _ = handle
        return None


def _headers(tenant_id: str, *, idempotency: str | None = None) -> dict[str, str]:
    headers = {"X-Tenant-Id": tenant_id}
    if idempotency is not None:
        headers["Idempotency-Key"] = idempotency
    return headers


def _write_candidates(path: Path, candidates: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "lkw.source_candidates.v1",
                "candidates": candidates,
            }
        ),
        encoding="utf-8",
    )


def _assert_safe(payload: object, *, forbidden: tuple[str, ...]) -> None:
    text = str(payload)
    for item in forbidden:
        assert item not in text


@pytest.fixture
def api_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("hello", encoding="utf-8")
    data_home = tmp_path / "data"
    data_home.mkdir()
    config_file = data_home / "config" / "source_candidates.json"
    _write_candidates(
        config_file,
        [
            {
                "candidate_id": "contracts",
                "tenant_id": "tenant-a",
                "label": "Contracts",
                "description": "Safe label",
                "source_type": "local_folder",
                "path": str(docs.resolve()),
                "recursive": True,
                "enabled": True,
            },
            {
                "candidate_id": "policies",
                "tenant_id": "tenant-a",
                "label": "Apple",
                "description": "",
                "source_type": "local_folder",
                "path": str(tmp_path / "missing"),
                "recursive": False,
                "enabled": True,
            },
            {
                "candidate_id": "other",
                "tenant_id": "tenant-b",
                "label": "Other",
                "description": "other tenant",
                "source_type": "local_folder",
                "path": str(docs.resolve()),
                "recursive": True,
                "enabled": True,
            },
        ],
    )
    monkeypatch.setenv("DATA_HOME", str(data_home))
    monkeypatch.setenv("INTERGRAX_ALLOWED_READ_ROOTS", str(docs.resolve()))
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    settings = replace(
        LocalWorkspaceBackendSettings.from_env(),
        data_home=str(data_home),
        allowed_read_roots=frozenset({str(docs.resolve())}),
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
        object_storage=None,
    )
    with TestClient(app) as client:
        yield client, repo, docs, data_home, settings, runtime


def _create_workspace(client: TestClient, tenant: str) -> str:
    response = client.post(
        f"{_PREFIX}/workspaces",
        headers=_headers(tenant),
        json={"name": "Docs"},
    )
    assert response.status_code == 201
    return response.json()["workspace_id"]


def test_list_candidates_safe_sorted_and_unavailable(api_bundle) -> None:
    client, _repo, docs, _data_home, _settings, _runtime = api_bundle
    workspace_id = _create_workspace(client, "tenant-a")
    response = client.get(
        f"{_PREFIX}/workspaces/{workspace_id}/source-candidates",
        headers=_headers("tenant-a"),
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["workspace_id"] == workspace_id
    ids = [item["candidate_id"] for item in payload["candidates"]]
    assert ids == ["policies", "contracts"]
    by_id = {item["candidate_id"]: item for item in payload["candidates"]}
    assert by_id["contracts"]["available"] is True
    assert by_id["policies"]["available"] is False
    _assert_safe(
        payload,
        forbidden=(
            str(docs),
            "sha256:",
            "source_candidates.json",
            "allowlist",
            str(_data_home),
        ),
    )


def test_list_empty_and_missing_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_home = tmp_path / "data"
    data_home.mkdir()
    monkeypatch.setenv("DATA_HOME", str(data_home))
    monkeypatch.setenv("INTERGRAX_ALLOWED_READ_ROOTS", str(tmp_path))
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    settings = replace(
        LocalWorkspaceBackendSettings.from_env(),
        data_home=str(data_home),
        allowed_read_roots=frozenset({str(tmp_path)}),
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
        object_storage=None,
    )
    with TestClient(app) as client:
        workspace_id = _create_workspace(client, "tenant-a")
        response = client.get(
            f"{_PREFIX}/workspaces/{workspace_id}/source-candidates",
            headers=_headers("tenant-a"),
        )
        assert response.status_code == 200
        assert response.json()["candidates"] == []
        missing = client.get(
            f"{_PREFIX}/workspaces/missing/source-candidates",
            headers=_headers("tenant-a"),
        )
        assert missing.status_code == 404


def test_list_cross_tenant_and_invalid_registry(api_bundle, tmp_path: Path, monkeypatch) -> None:
    client, _repo, docs, data_home, settings, runtime = api_bundle
    workspace_id = _create_workspace(client, "tenant-a")
    cross = client.get(
        f"{_PREFIX}/workspaces/{workspace_id}/source-candidates",
        headers=_headers("tenant-b"),
    )
    assert cross.status_code == 404

    bad_home = tmp_path / "bad-home"
    bad_home.mkdir()
    config = bad_home / "config" / "source_candidates.json"
    config.parent.mkdir(parents=True)
    config.write_text("{bad", encoding="utf-8")
    monkeypatch.setenv("DATA_HOME", str(bad_home))
    monkeypatch.setenv("INTERGRAX_ALLOWED_READ_ROOTS", str(docs.resolve()))
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    bad_settings = replace(
        LocalWorkspaceBackendSettings.from_env(),
        data_home=str(bad_home),
        allowed_read_roots=frozenset({str(docs.resolve())}),
    )
    executor = _FakeExecutor()
    sync = ManagedWorkspaceSyncService(repo, executor)  # type: ignore[arg-type]
    bad_runtime = build_managed_workspace_sync_runtime(
        document_store=store,
        sync_service=sync,
        repository=repo,
    )
    app = FastAPI()
    mount_managed_workspace_routes(
        app,
        task_executor=executor,  # type: ignore[arg-type]
        settings=bad_settings,
        repository=repo,
        sync_runtime=bad_runtime,
        object_storage=None,
    )
    with TestClient(app) as bad_client:
        wid = _create_workspace(bad_client, "tenant-a")
        response = bad_client.get(
            f"{_PREFIX}/workspaces/{wid}/source-candidates",
            headers=_headers("tenant-a"),
        )
        assert response.status_code == 503
        assert response.json()["detail"] == "source_candidate_registry_unavailable"
        assert "JSON" not in str(response.json())
        assert str(config) not in str(response.json())
    _ = data_home, settings, runtime


def test_accept_success_and_operation_get(api_bundle) -> None:
    client, _repo, docs, _data_home, _settings, runtime = api_bundle
    workspace_id = _create_workspace(client, "tenant-a")
    response = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/knowledge/source-candidates/contracts",
        headers=_headers("tenant-a", idempotency="idem-1"),
    )
    assert response.status_code == 202
    payload = response.json()
    assert set(payload.keys()) == {
        "candidate_id",
        "label",
        "workspace_id",
        "source_id",
        "operation_id",
        "status",
    }
    assert payload["candidate_id"] == "contracts"
    assert payload["label"] == "Contracts"
    assert payload["workspace_id"] == workspace_id
    assert payload["status"] in {"accepted", "queued", "processing", "completed", "failed"}
    _assert_safe(payload, forbidden=(str(docs), "sha256:", "path"))

    runtime.worker.drain_once()
    op = client.get(
        f"{_PREFIX}/operations/{payload['operation_id']}",
        headers=_headers("tenant-a"),
    )
    assert op.status_code == 200
    body = op.json()
    assert body["operation_id"] == payload["operation_id"]
    assert body["files_discovered"] == 1
    assert body["files_processed"] == 1
    assert body["files_failed"] == 0
    assert body["documents_indexed"] == 1
    _assert_safe(body, forbidden=(str(docs), "sha256:"))


def test_embedded_path_in_public_fields_returns_503_without_path_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("hello", encoding="utf-8")
    private = "C:\\Private\\Acquisition"
    data_home = tmp_path / "data"
    data_home.mkdir()
    _write_candidates(
        data_home / "config" / "source_candidates.json",
        [
            {
                "candidate_id": "contracts",
                "tenant_id": "tenant-a",
                "label": f"Contracts from {private}",
                "description": "Documents in /srv/company/confidential",
                "source_type": "local_folder",
                "path": str(docs.resolve()),
                "recursive": True,
                "enabled": True,
            }
        ],
    )
    monkeypatch.setenv("DATA_HOME", str(data_home))
    monkeypatch.setenv("INTERGRAX_ALLOWED_READ_ROOTS", str(docs.resolve()))
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    settings = replace(
        LocalWorkspaceBackendSettings.from_env(),
        data_home=str(data_home),
        allowed_read_roots=frozenset({str(docs.resolve())}),
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
        object_storage=None,
    )
    with TestClient(app) as client:
        workspace_id = _create_workspace(client, "tenant-a")
        response = client.get(
            f"{_PREFIX}/workspaces/{workspace_id}/source-candidates",
            headers=_headers("tenant-a"),
        )
        assert response.status_code == 503
        assert response.json()["detail"] == "source_candidate_registry_unavailable"
        text = str(response.json())
        assert private not in text
        assert "/srv/company/confidential" not in text
        assert str(docs) not in text


def test_accept_error_matrix(api_bundle) -> None:
    client, _repo, docs, _data_home, _settings, _runtime = api_bundle
    workspace_id = _create_workspace(client, "tenant-a")

    missing_key = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/knowledge/source-candidates/contracts",
        headers=_headers("tenant-a"),
    )
    assert missing_key.status_code == 400
    assert missing_key.json()["detail"] == "idempotency_key_required"

    missing_candidate = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/knowledge/source-candidates/nope",
        headers=_headers("tenant-a", idempotency="k1"),
    )
    assert missing_candidate.status_code == 404

    cross_candidate = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/knowledge/source-candidates/other",
        headers=_headers("tenant-a", idempotency="k2"),
    )
    assert cross_candidate.status_code == 404

    missing_ws = client.post(
        f"{_PREFIX}/workspaces/missing/knowledge/source-candidates/contracts",
        headers=_headers("tenant-a", idempotency="k3"),
    )
    assert missing_ws.status_code == 404

    unavailable = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/knowledge/source-candidates/policies",
        headers=_headers("tenant-a", idempotency="k4"),
    )
    assert unavailable.status_code == 409
    assert unavailable.json()["detail"] == "source_candidate_unavailable"
    assert str(docs) not in str(unavailable.json())

    first = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/knowledge/source-candidates/contracts",
        headers=_headers("tenant-a", idempotency="shared"),
    )
    assert first.status_code == 202
    conflict = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/knowledge/source-candidates/policies",
        headers=_headers("tenant-a", idempotency="shared"),
    )
    assert conflict.status_code == 409
    assert conflict.json()["detail"] == "source_candidate_idempotency_conflict"

    again = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/knowledge/source-candidates/contracts",
        headers=_headers("tenant-a", idempotency="other-key"),
    )
    assert again.status_code == 409
    assert again.json()["detail"] == "source_candidate_already_registered"


def test_accept_dispatch_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("hello", encoding="utf-8")
    data_home = tmp_path / "data"
    data_home.mkdir()
    _write_candidates(
        data_home / "config" / "source_candidates.json",
        [
            {
                "candidate_id": "contracts",
                "tenant_id": "tenant-a",
                "label": "Contracts",
                "description": "",
                "source_type": "local_folder",
                "path": str(docs.resolve()),
                "recursive": True,
                "enabled": True,
            }
        ],
    )
    monkeypatch.setenv("DATA_HOME", str(data_home))
    monkeypatch.setenv("INTERGRAX_ALLOWED_READ_ROOTS", str(docs.resolve()))
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    settings = replace(
        LocalWorkspaceBackendSettings.from_env(),
        data_home=str(data_home),
        allowed_read_roots=frozenset({str(docs.resolve())}),
    )
    executor = _FakeExecutor()
    sync = ManagedWorkspaceSyncService(repo, executor)  # type: ignore[arg-type]
    runtime = build_managed_workspace_sync_runtime(
        document_store=store,
        sync_service=sync,
        repository=repo,
        existing_message_bus=_RaisingBus(),
    )
    app = FastAPI()
    mount_managed_workspace_routes(
        app,
        task_executor=executor,  # type: ignore[arg-type]
        settings=settings,
        repository=repo,
        sync_runtime=runtime,
        object_storage=None,
    )
    with TestClient(app) as client:
        workspace_id = _create_workspace(client, "tenant-a")
        response = client.post(
            f"{_PREFIX}/workspaces/{workspace_id}/knowledge/source-candidates/contracts",
            headers=_headers("tenant-a", idempotency="dispatch-fail"),
        )
        assert response.status_code == 502
        assert response.json()["detail"] == "source_candidate_dispatch_failed"


def test_managed_file_still_unavailable_without_object_storage(api_bundle) -> None:
    client, *_rest = api_bundle
    workspace_id = _create_workspace(client, "tenant-a")
    response = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/knowledge/files",
        headers=_headers("tenant-a", idempotency="mf-1"),
        files=[("files", ("a.txt", b"hello", "text/plain"))],
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "managed_file_storage_unavailable"

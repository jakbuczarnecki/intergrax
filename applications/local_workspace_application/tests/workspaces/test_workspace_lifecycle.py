# © Artur Czarnecki. All rights reserved.

"""LKW-WORKSPACE-MANAGEMENT-1 — backend create/delete lifecycle cleanup."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.providers.vector_store.inmemory.rag_store import InMemoryVectorStore
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager
from local_workspace_application.host.factory import create_local_workspace_backend_app
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.workspaces.ask_models import AskRunStatus, WorkspaceAskRun
from local_workspace_application.workspaces.ask_repository import WorkspaceAskRepository
from local_workspace_application.workspaces.models import (
    WorkspaceDocumentReference,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.service import ManagedWorkspaceService
from local_workspace_application.workspaces.vector_cleanup import (
    VectorstoreManagerWorkspaceCleanup,
)

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
def api_client(workspace_root: Path, monkeypatch: pytest.MonkeyPatch):
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
    app = create_local_workspace_backend_app(settings=settings)
    # Avoid live Qdrant from the application manifest in unit API tests.
    # Vector-scoped cleanup is covered by the service-level lifecycle test.
    app.state.lkw_managed_workspace_service._vector_cleanup = None  # noqa: SLF001
    with TestClient(app) as client:
        yield client, store, workspace_root


def _headers(tenant_id: str) -> dict[str, str]:
    return {"X-Tenant-Id": tenant_id}


def test_same_tenant_create(api_client) -> None:
    client, _, _ = api_client
    tenant = _unique_tenant()
    created = client.post(
        f"{_PREFIX}/workspaces",
        headers=_headers(tenant),
        json={"name": "Alpha Case"},
    )
    assert created.status_code == 201
    assert created.json()["tenant_id"] == tenant
    assert created.json()["name"] == "Alpha Case"


def test_cross_tenant_isolation(api_client) -> None:
    client, _, _ = api_client
    tenant_a = _unique_tenant("a")
    tenant_b = _unique_tenant("b")
    created = client.post(
        f"{_PREFIX}/workspaces",
        headers=_headers(tenant_a),
        json={"name": "Private"},
    )
    workspace_id = created.json()["workspace_id"]
    listed_b = client.get(f"{_PREFIX}/workspaces", headers=_headers(tenant_b))
    assert listed_b.status_code == 200
    assert listed_b.json()["workspaces"] == []
    fetched = client.get(
        f"{_PREFIX}/workspaces/{workspace_id}",
        headers=_headers(tenant_b),
    )
    assert fetched.status_code == 404


def test_same_tenant_delete_and_list_gone(api_client) -> None:
    client, _, _ = api_client
    tenant = _unique_tenant()
    created = client.post(
        f"{_PREFIX}/workspaces",
        headers=_headers(tenant),
        json={"name": "To Delete"},
    )
    workspace_id = created.json()["workspace_id"]
    deleted = client.delete(
        f"{_PREFIX}/workspaces/{workspace_id}",
        headers=_headers(tenant),
    )
    assert deleted.status_code == 204, deleted.text
    assert deleted.content in (b"", b"null")
    listed = client.get(f"{_PREFIX}/workspaces", headers=_headers(tenant))
    assert listed.status_code == 200
    assert listed.json()["workspaces"] == []


def test_unknown_delete_404(api_client) -> None:
    client, _, _ = api_client
    tenant = _unique_tenant()
    response = client.delete(
        f"{_PREFIX}/workspaces/{uuid.uuid4()}",
        headers=_headers(tenant),
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "not_found"


def test_cross_tenant_delete_404(api_client) -> None:
    client, _, _ = api_client
    tenant_a = _unique_tenant("a")
    tenant_b = _unique_tenant("b")
    created = client.post(
        f"{_PREFIX}/workspaces",
        headers=_headers(tenant_a),
        json={"name": "Owned"},
    )
    workspace_id = created.json()["workspace_id"]
    response = client.delete(
        f"{_PREFIX}/workspaces/{workspace_id}",
        headers=_headers(tenant_b),
    )
    assert response.status_code == 404
    still = client.get(
        f"{_PREFIX}/workspaces/{workspace_id}",
        headers=_headers(tenant_a),
    )
    assert still.status_code == 200


def test_delete_cleans_sources_docs_ops_ask_vectors_and_preserves_files(
    workspace_root: Path,
) -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    ask_repo = WorkspaceAskRepository(store)
    vector_store = InMemoryVectorStore(tenant_id="t-life")
    manager = VectorstoreManager(vector_store)
    service = ManagedWorkspaceService(
        repo,
        allowlist_roots=frozenset({str(workspace_root.resolve())}),
        ask_repository=ask_repo,
        vector_cleanup=VectorstoreManagerWorkspaceCleanup(manager),
    )

    keep = service.create_workspace(tenant_id="t-life", name="Keep")
    target = service.create_workspace(tenant_id="t-life", name="Target")
    source_file = workspace_root / "contract.txt"
    source_file.write_text("hello evidence", encoding="utf-8")
    source = service.register_local_folder_source(
        tenant_id="t-life",
        workspace_id=target.workspace_id,
        path=str(workspace_root),
    )
    keep_source = service.register_local_folder_source(
        tenant_id="t-life",
        workspace_id=keep.workspace_id,
        path=str(workspace_root),
    )
    op = service.create_sync_operation(
        tenant_id="t-life",
        workspace_id=target.workspace_id,
        source_id=source.source_id,
    )
    keep_op = service.create_sync_operation(
        tenant_id="t-life",
        workspace_id=keep.workspace_id,
        source_id=keep_source.source_id,
    )
    now = datetime.now(UTC)
    repo.put_document_ref(
        WorkspaceDocumentReference(
            document_id="doc-target-1",
            tenant_id="t-life",
            workspace_id=target.workspace_id,
            source_id=source.source_id,
            source_path="contract.txt",
            file_name="contract.txt",
            content_hash="abc",
            indexed_at=now,
        )
    )
    repo.put_document_ref(
        WorkspaceDocumentReference(
            document_id="doc-keep-1",
            tenant_id="t-life",
            workspace_id=keep.workspace_id,
            source_id=keep_source.source_id,
            source_path="contract.txt",
            file_name="contract.txt",
            content_hash="abc",
            indexed_at=now,
        )
    )
    ask_repo.put_run(
        WorkspaceAskRun(
            run_id="ask-target",
            tenant_id="t-life",
            workspace_id=target.workspace_id,
            question="Q?",
            status=AskRunStatus.COMPLETED,
            answer="A",
            created_at=now,
            completed_at=now,
        )
    )
    ask_repo.put_run(
        WorkspaceAskRun(
            run_id="ask-keep",
            tenant_id="t-life",
            workspace_id=keep.workspace_id,
            question="Q2?",
            status=AskRunStatus.COMPLETED,
            answer="A2",
            created_at=now,
            completed_at=now,
        )
    )
    manager.add_documents(
        [Document(page_content="target chunk", metadata={"workspace_id": target.workspace_id})],
        [[0.1, 0.2]],
        ids=["vec-target"],
        base_metadata={
            "tenant_id": "t-life",
            "workspace_id": target.workspace_id,
            "collection_id": target.workspace_id,
        },
    )
    manager.add_documents(
        [Document(page_content="keep chunk", metadata={"workspace_id": keep.workspace_id})],
        [[0.3, 0.4]],
        ids=["vec-keep"],
        base_metadata={
            "tenant_id": "t-life",
            "workspace_id": keep.workspace_id,
            "collection_id": keep.workspace_id,
        },
    )

    assert service.delete_workspace(
        tenant_id="t-life",
        workspace_id=target.workspace_id,
    )
    assert service.get_workspace(tenant_id="t-life", workspace_id=target.workspace_id) is None
    assert repo.list_sources(tenant_id="t-life", workspace_id=target.workspace_id) == []
    assert repo.list_document_refs(tenant_id="t-life", workspace_id=target.workspace_id) == []
    assert repo.get_operation(tenant_id="t-life", operation_id=op.operation_id) is None
    assert ask_repo.get_run(tenant_id="t-life", run_id="ask-target") is None

    # Other workspace untouched.
    assert service.get_workspace(tenant_id="t-life", workspace_id=keep.workspace_id) is not None
    assert len(repo.list_sources(tenant_id="t-life", workspace_id=keep.workspace_id)) == 1
    assert len(repo.list_document_refs(tenant_id="t-life", workspace_id=keep.workspace_id)) == 1
    assert repo.get_operation(tenant_id="t-life", operation_id=keep_op.operation_id) is not None
    assert ask_repo.get_run(tenant_id="t-life", run_id="ask-keep") is not None

    remaining = manager.search_by_metadata(
        conditions={"tenant_id": "t-life"},
        limit=50,
    )
    remaining_ids = {item["id"] for item in remaining}
    assert "vec-target" not in remaining_ids
    assert "vec-keep" in remaining_ids

    # Local files never touched.
    assert source_file.exists()
    assert source_file.read_text(encoding="utf-8") == "hello evidence"


def test_vector_cleanup_uses_store_tenant_when_product_tenant_differs() -> None:
    """Slack product tenant must not fail Qdrant/InMemory fixed-tenant search."""
    vector_store = InMemoryVectorStore(tenant_id="default")
    manager = VectorstoreManager(vector_store)
    cleanup = VectorstoreManagerWorkspaceCleanup(manager)
    workspace_id = "ws-product-tenant-mismatch"
    manager.add_documents(
        [Document(page_content="chunk", metadata={"workspace_id": workspace_id})],
        [[0.1, 0.2]],
        ids=["vec-mismatch"],
        base_metadata={
            "tenant_id": "default",
            "workspace_id": workspace_id,
            "collection_id": workspace_id,
        },
    )

    deleted = cleanup.delete_workspace_vectors(
        tenant_id="lkw-ask-qdrant-durability",
        workspace_id=workspace_id,
    )
    assert deleted >= 1
    remaining = manager.search_by_metadata(
        conditions={"tenant_id": "default"},
        limit=50,
    )
    assert "vec-mismatch" not in {item["id"] for item in remaining}


def test_vector_cleanup_absent_qdrant_collection_is_empty_not_error() -> None:
    """Never-indexed workspace: missing Qdrant collection must delete 0, not raise."""

    class _UnexpectedResponse(Exception):
        pass

    class _AbsentCollectionStore:
        tenant_id = "default"

        def search_by_metadata(self, *, conditions: dict, limit: int = 50) -> list[dict]:
            raise _UnexpectedResponse(
                'Unexpected Response: 404 (Not Found)\n'
                'Raw response content:\n'
                'b\'{"status":{"error":"Not found: Collection '
                "`local_workspace__tenant__default` doesn't exist!"
                '},"time":2.496e-6}\''
            )

        def delete(self, ids: list[str]) -> None:
            raise AssertionError("delete must not be called when collection is absent")

    cleanup = VectorstoreManagerWorkspaceCleanup(_AbsentCollectionStore())
    assert (
        cleanup.delete_workspace_vectors(
            tenant_id="lkw-ask-qdrant-durability",
            workspace_id="adfcd45f-45e1-4d5f-a8f8-01e13a4ebcb8",
        )
        == 0
    )


def test_delete_continues_when_vector_cleanup_unsupported(workspace_root: Path) -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)

    class _BrokenCleanup:
        def delete_workspace_vectors(self, *, tenant_id: str, workspace_id: str) -> int:
            raise RuntimeError("vectorstore_workspace_cleanup_not_supported")

    service = ManagedWorkspaceService(
        repo,
        allowlist_roots=frozenset({str(workspace_root.resolve())}),
        vector_cleanup=_BrokenCleanup(),
    )
    created = service.create_workspace(tenant_id="t-soft", name="Soft Fail")
    assert service.delete_workspace(
        tenant_id="t-soft",
        workspace_id=created.workspace_id,
    )
    assert service.get_workspace(tenant_id="t-soft", workspace_id=created.workspace_id) is None

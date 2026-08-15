# © Artur Czarnecki. All rights reserved.

"""HTTP tests for knowledge inventory, operations, and workspace operation list."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import DocumentRecord
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.serving.workspace_routes import mount_managed_workspace_routes
from local_workspace_application.workspaces.knowledge_indexed_source_lifecycle_service import (
    IndexedSourceLifecycleStateV1,
    IndexedSourceSyncStateV1,
)
from local_workspace_application.workspaces.knowledge_inspection_operations_service import (
    KnowledgeAccessModeV1,
    KnowledgeInventoryError,
    KnowledgeInventoryItemV1,
    KnowledgeInventorySummaryV1,
    KnowledgeInventoryV1,
    KnowledgeOperationCommandV1,
    KnowledgeOperationError,
    KnowledgeOperationResultV1,
    KnowledgeOperationV1,
    KnowledgeRevisionKindV1,
    indexed_knowledge_item_id,
    live_knowledge_item_id,
)
from local_workspace_application.workspaces.knowledge_live_access_service import (
    LiveAccessLifecycleStateV1,
)
from local_workspace_application.workspaces.models import (
    WorkspaceOperation,
    WorkspaceOperationIndexEntryV1,
    WorkspaceOperationStatus,
    WorkspaceOperationType,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.sync_runtime import build_managed_workspace_sync_runtime
from local_workspace_application.workspaces.sync_service import ManagedWorkspaceSyncService

pytestmark = pytest.mark.unit

_PREFIX = "/v1/local_workspace"
_NOW = datetime(2026, 8, 8, 10, 0, tzinfo=UTC)
_TENANT = "tenant-a"
_TENANT_B = "tenant-b"
_HASH = "a" * 64
_CONFIRM_SECRET = "confirm-secret"


class _FakeExecutor:
    async def execute(self, task: object) -> object:
        _ = task
        return type("R", (), {"metadata": {}})()


def _headers(
    tenant: str = _TENANT,
    *,
    idempotency: str | None = "idem-1",
) -> dict[str, str]:
    payload = {"X-Tenant-Id": tenant}
    if idempotency is not None:
        payload["Idempotency-Key"] = idempotency
    return payload


def _indexed_item(*, error: bool = False) -> KnowledgeInventoryItemV1:
    return KnowledgeInventoryItemV1(
        tenant_id=_TENANT,
        workspace_id="ws-1",
        knowledge_item_id=indexed_knowledge_item_id("idx-1"),
        mode=KnowledgeAccessModeV1.INDEXED,
        source_id="source-1",
        indexed_source_binding_id="idx-1",
        display_label="Indexed",
        lifecycle_state="error" if error else "active",
        enabled=True,
        detached=False,
        sync_state="failed" if error else "succeeded",
        last_error_code="sync_failed" if error else None,
        revision=3,
        revision_kind=KnowledgeRevisionKindV1.LIFECYCLE,
        available_actions=(
            KnowledgeOperationV1.SYNC,
            KnowledgeOperationV1.RETRY_SYNC,
            KnowledgeOperationV1.DISABLE,
            KnowledgeOperationV1.DETACH,
        )
        if error
        else (
            KnowledgeOperationV1.SYNC,
            KnowledgeOperationV1.DISABLE,
            KnowledgeOperationV1.DETACH,
        ),
        updated_at=_NOW,
    )


def _live_item(*, unavailable: bool = False) -> KnowledgeInventoryItemV1:
    return KnowledgeInventoryItemV1(
        tenant_id=_TENANT,
        workspace_id="ws-1",
        knowledge_item_id=live_knowledge_item_id("live-1"),
        mode=KnowledgeAccessModeV1.LIVE,
        live_access_binding_id="live-1",
        connection_ref="conn-1",
        display_label="Live",
        provider_id="provider-1",
        source_kind="wiki",
        capability_ids=("cap.read",),
        lifecycle_state="active",
        enabled=True,
        detached=False,
        runtime_available=not unavailable,
        last_error_code="connection_unavailable" if unavailable else None,
        revision=4,
        revision_kind=KnowledgeRevisionKindV1.CONFIGURATION,
        available_actions=(KnowledgeOperationV1.DISABLE, KnowledgeOperationV1.DETACH),
        updated_at=_NOW,
    )


def _inventory(*, items: tuple[KnowledgeInventoryItemV1, ...]) -> KnowledgeInventoryV1:
    return KnowledgeInventoryV1(
        tenant_id=_TENANT,
        workspace_id="ws-1",
        items=items,
        summary=KnowledgeInventorySummaryV1(
            total=len(items),
            indexed=sum(item.mode is KnowledgeAccessModeV1.INDEXED for item in items),
            live=sum(item.mode is KnowledgeAccessModeV1.LIVE for item in items),
            active=sum(item.lifecycle_state == "active" for item in items),
            disabled=sum(item.lifecycle_state == "disabled" for item in items),
            attention_required=sum(
                item.lifecycle_state in {"error", "detach_blocked"}
                or (
                    item.runtime_available is False
                    and item.enabled
                    and not item.detached
                )
                for item in items
            ),
        ),
        updated_at=_NOW,
    )


@pytest.fixture
def api_bundle(tmp_path, monkeypatch: pytest.MonkeyPatch):
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    data_home = tmp_path / "data"
    data_home.mkdir()
    monkeypatch.setenv("DATA_HOME", str(data_home))
    monkeypatch.setenv("INTERGRAX_ALLOWED_READ_ROOTS", str(tmp_path / "docs"))
    (tmp_path / "docs").mkdir()
    settings = replace(
        LocalWorkspaceBackendSettings.from_env(),
        data_home=str(data_home),
        knowledge_admin_confirmation_secret=_CONFIRM_SECRET,
    )
    executor = _FakeExecutor()
    sync = ManagedWorkspaceSyncService(repo, executor)  # type: ignore[arg-type]
    runtime = build_managed_workspace_sync_runtime(
        document_store=store,
        sync_service=sync,
        repository=repo,
    )
    app = FastAPI()
    service = mount_managed_workspace_routes(
        app,
        task_executor=executor,  # type: ignore[arg-type]
        settings=settings,
        repository=repo,
        sync_runtime=runtime,
    )
    workspace = service.create_workspace(tenant_id=_TENANT, name="Docs")
    with TestClient(app) as client:
        yield client, repo, service, workspace.workspace_id, app


def _inventory_path(workspace_id: str) -> str:
    return f"{_PREFIX}/workspaces/{workspace_id}/knowledge/inventory"


def _operation_path(workspace_id: str, item_id: str) -> str:
    return f"{_PREFIX}/workspaces/{workspace_id}/knowledge/items/{item_id}/operations"


def _operations_list_path(workspace_id: str) -> str:
    return f"{_PREFIX}/workspaces/{workspace_id}/operations"


def test_empty_workspace_inventory(api_bundle) -> None:
    client, _, _, workspace_id, app = api_bundle
    inspection = MagicMock()
    inspection.list_items.return_value = _inventory(items=())
    app.state.lkw_knowledge_inspection_service = inspection

    response = client.get(_inventory_path(workspace_id), headers=_headers())

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["items"] == []
    assert body["summary"]["total"] == 0


def test_indexed_item_projection(api_bundle) -> None:
    client, _, _, workspace_id, app = api_bundle
    item = _indexed_item()
    inspection = MagicMock()
    inspection.list_items.return_value = _inventory(items=(item,))
    app.state.lkw_knowledge_inspection_service = inspection

    response = client.get(_inventory_path(workspace_id), headers=_headers())

    assert response.status_code == 200
    projected = response.json()["items"][0]
    assert projected["mode"] == "indexed"
    assert projected["indexed_source_binding_id"] == "idx-1"
    assert projected["sync_state"] == "succeeded"
    assert "token" not in str(projected).casefold()


def test_live_item_projection(api_bundle) -> None:
    client, _, _, workspace_id, app = api_bundle
    item = _live_item()
    inspection = MagicMock()
    inspection.list_items.return_value = _inventory(items=(item,))
    app.state.lkw_knowledge_inspection_service = inspection

    response = client.get(_inventory_path(workspace_id), headers=_headers())

    assert response.status_code == 200
    projected = response.json()["items"][0]
    assert projected["mode"] == "live"
    assert projected["runtime_available"] is True
    assert projected["provider_id"] == "provider-1"


def test_attention_required_projection(api_bundle) -> None:
    client, _, _, workspace_id, app = api_bundle
    items = (_indexed_item(error=True), _live_item(unavailable=True))
    inspection = MagicMock()
    inspection.list_items.return_value = _inventory(items=items)
    app.state.lkw_knowledge_inspection_service = inspection

    response = client.get(_inventory_path(workspace_id), headers=_headers())

    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["attention_required"] == 2
    assert body["items"][0]["last_error_code"] == "sync_failed"
    assert body["items"][1]["last_error_code"] == "connection_unavailable"


def test_available_actions_projection(api_bundle) -> None:
    client, _, _, workspace_id, app = api_bundle
    item = _indexed_item(error=True)
    inspection = MagicMock()
    inspection.list_items.return_value = _inventory(items=(item,))
    app.state.lkw_knowledge_inspection_service = inspection

    response = client.get(_inventory_path(workspace_id), headers=_headers())

    assert response.status_code == 200
    actions = response.json()["items"][0]["available_actions"]
    assert "retry_sync" in actions
    assert "sync" in actions


def test_inventory_tenant_workspace_isolation(api_bundle) -> None:
    client, _, _, workspace_id, app = api_bundle
    inspection = MagicMock()
    inspection.list_items.side_effect = KnowledgeInventoryError("knowledge_item_not_found")
    app.state.lkw_knowledge_inspection_service = inspection

    foreign = client.get(_inventory_path(workspace_id), headers=_headers(_TENANT_B))
    assert foreign.status_code == 404

    missing = client.get(f"{_PREFIX}/workspaces/missing/knowledge/inventory", headers=_headers())
    assert missing.status_code == 404


def test_allowed_operation_delegates_to_service(api_bundle) -> None:
    client, _, _, workspace_id, app = api_bundle
    item = _indexed_item()
    result = KnowledgeOperationResultV1(
        item=item,
        operation=KnowledgeOperationV1.SYNC,
        operation_id="op-sync",
        mutation_id="mut-sync",
    )
    operations = MagicMock()
    operations.execute = AsyncMock(return_value=result)
    app.state.lkw_knowledge_operations_service = operations

    response = client.post(
        _operation_path(workspace_id, item.knowledge_item_id),
        headers=_headers(),
        json={"operation": "sync", "expected_revision": 3},
    )

    assert response.status_code == 200, response.text
    operations.execute.assert_awaited_once()
    command = operations.execute.await_args.args[0]
    assert isinstance(command, KnowledgeOperationCommandV1)
    assert command.operation is KnowledgeOperationV1.SYNC


def test_unsupported_operation_rejected(api_bundle) -> None:
    client, _, _, workspace_id, app = api_bundle
    operations = MagicMock()
    operations.execute = AsyncMock(
        side_effect=KnowledgeOperationError("knowledge_operation_not_supported")
    )
    app.state.lkw_knowledge_operations_service = operations

    response = client.post(
        _operation_path(workspace_id, indexed_knowledge_item_id("idx-1")),
        headers=_headers(),
        json={"operation": "resume_detach", "expected_revision": 1},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "knowledge_operation_not_supported"


def test_destructive_operation_requires_confirmation(api_bundle) -> None:
    client, _, _, workspace_id, app = api_bundle
    item = _indexed_item()
    inspection = MagicMock()
    inspection.get_item.return_value = item
    operations = MagicMock()
    operations.execute = AsyncMock()
    app.state.lkw_knowledge_inspection_service = inspection
    app.state.lkw_knowledge_operations_service = operations

    first = client.post(
        _operation_path(workspace_id, item.knowledge_item_id),
        headers=_headers(),
        json={"operation": "detach", "expected_revision": 3},
    )

    assert first.status_code == 409
    detail = first.json()["detail"]
    assert detail["error_code"] == "knowledge_admin_confirmation_required"
    assert detail["confirmation_token"]
    operations.execute.assert_not_called()


def test_revision_precondition_failure_preserved(api_bundle) -> None:
    client, _, _, workspace_id, app = api_bundle
    operations = MagicMock()
    operations.execute = AsyncMock(
        side_effect=KnowledgeOperationError("knowledge_operation_conflict")
    )
    app.state.lkw_knowledge_operations_service = operations

    response = client.post(
        _operation_path(workspace_id, indexed_knowledge_item_id("idx-1")),
        headers=_headers(),
        json={"operation": "sync", "expected_revision": 99},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "knowledge_operation_conflict"


def test_route_does_not_mutate_lifecycle_directly(api_bundle) -> None:
    client, _, _, workspace_id, app = api_bundle
    indexed_lifecycle = MagicMock()
    app.state.lkw_connected_source_wiring = SimpleNamespace(
        indexed_source_lifecycle_service=indexed_lifecycle
    )
    item = _indexed_item()
    result = KnowledgeOperationResultV1(
        item=item,
        operation=KnowledgeOperationV1.SYNC,
        operation_id="op-sync",
        mutation_id=None,
    )
    operations = MagicMock()
    operations.execute = AsyncMock(return_value=result)
    app.state.lkw_knowledge_operations_service = operations

    response = client.post(
        _operation_path(workspace_id, item.knowledge_item_id),
        headers=_headers(),
        json={"operation": "sync", "expected_revision": 3},
    )

    assert response.status_code == 200
    indexed_lifecycle.request_sync.assert_not_called()
    indexed_lifecycle.disable.assert_not_called()


def test_workspace_operations_list_scoped_and_ordered(api_bundle) -> None:
    client, repo, service, workspace_id, _ = api_bundle
    other = service.create_workspace(tenant_id=_TENANT, name="Other")
    older = datetime(2026, 8, 7, 10, 0, tzinfo=UTC)
    newer = datetime(2026, 8, 8, 11, 0, tzinfo=UTC)
    repo.put_operation(
        WorkspaceOperation(
            operation_id="op-old",
            tenant_id=_TENANT,
            workspace_id=workspace_id,
            source_id="source-1",
            operation_type=WorkspaceOperationType.SOURCE_SYNC,
            status=WorkspaceOperationStatus.COMPLETED,
            created_at=older,
            error_code="sync_timeout",
        )
    )
    repo.put_operation(
        WorkspaceOperation(
            operation_id="op-new",
            tenant_id=_TENANT,
            workspace_id=workspace_id,
            source_id="source-1",
            operation_type=WorkspaceOperationType.KNOWLEDGE_INGESTION,
            status=WorkspaceOperationStatus.FAILED,
            created_at=newer,
            error_code="ingestion_failed",
        )
    )
    repo.put_operation(
        WorkspaceOperation(
            operation_id="op-other",
            tenant_id=_TENANT,
            workspace_id=other.workspace_id,
            source_id="source-2",
            operation_type=WorkspaceOperationType.SOURCE_SYNC,
            status=WorkspaceOperationStatus.COMPLETED,
            created_at=newer,
        )
    )

    response = client.get(
        _operations_list_path(workspace_id),
        headers=_headers(),
        params={"limit": 1},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["workspace_id"] == workspace_id
    assert len(body["operations"]) == 1
    assert body["operations"][0]["operation_id"] == "op-new"
    assert body["operations"][0]["error_code"] == "ingestion_failed"


def test_workspace_operations_tenant_isolation(api_bundle) -> None:
    client, repo, _, workspace_id, _ = api_bundle
    repo.put_operation(
        WorkspaceOperation(
            operation_id="op-tenant-a",
            tenant_id=_TENANT,
            workspace_id=workspace_id,
            source_id="source-1",
            operation_type=WorkspaceOperationType.SOURCE_SYNC,
            status=WorkspaceOperationStatus.COMPLETED,
            created_at=_NOW,
        )
    )

    foreign = client.get(
        _operations_list_path(workspace_id),
        headers=_headers(_TENANT_B),
    )
    assert foreign.status_code == 404


def test_workspace_operations_empty_list(api_bundle) -> None:
    client, _, _, workspace_id, _ = api_bundle

    response = client.get(_operations_list_path(workspace_id), headers=_headers())

    assert response.status_code == 200
    assert response.json()["operations"] == []


def test_get_operation_projects_error_code(api_bundle) -> None:
    client, repo, _, workspace_id, _ = api_bundle
    repo.put_operation(
        WorkspaceOperation(
            operation_id="op-failed",
            tenant_id=_TENANT,
            workspace_id=workspace_id,
            source_id="source-1",
            operation_type=WorkspaceOperationType.SOURCE_SYNC,
            status=WorkspaceOperationStatus.FAILED,
            created_at=_NOW,
            error="sync failed",
            error_code="sync_failed",
        )
    )

    response = client.get(f"{_PREFIX}/operations/op-failed", headers=_headers())

    assert response.status_code == 200
    body = response.json()
    assert body["error_code"] == "sync_failed"
    assert body["status"] == "failed"
    assert body["operation_id"] == "op-failed"


def test_get_operation_backward_compatible_without_error_code(api_bundle) -> None:
    client, repo, _, workspace_id, _ = api_bundle
    repo.put_operation(
        WorkspaceOperation(
            operation_id="op-legacy",
            tenant_id=_TENANT,
            workspace_id=workspace_id,
            source_id="source-1",
            operation_type=WorkspaceOperationType.SOURCE_SYNC,
            status=WorkspaceOperationStatus.COMPLETED,
            created_at=_NOW,
        )
    )

    response = client.get(f"{_PREFIX}/operations/op-legacy", headers=_headers())

    assert response.status_code == 200
    body = response.json()
    assert body["error_code"] is None
    assert set(body) >= {
        "operation_id",
        "operation_type",
        "status",
        "workspace_id",
        "source_id",
        "files_discovered",
        "documents_indexed",
    }


def _operation_partition(tenant_id: str = _TENANT) -> str:
    return f"lkw.managed_workspace:{tenant_id}:operation"


def _workspace_operation_index_partition(tenant_id: str = _TENANT) -> str:
    return f"lkw.managed_workspace:{tenant_id}:workspace_operation_index"


def _count_workspace_operation_index_entries(
    repo: ManagedWorkspaceRepository,
    *,
    workspace_id: str,
    tenant_id: str = _TENANT,
) -> int:
    page = repo.document_store.query(
        _workspace_operation_index_partition(tenant_id),
        limit=500,
        row_key_prefix=f"{workspace_id}:",
    )
    return len(page.documents)


def _workspace_operation_index_row_keys(
    repo: ManagedWorkspaceRepository,
    *,
    workspace_id: str,
    tenant_id: str = _TENANT,
) -> list[str]:
    page = repo.document_store.query(
        _workspace_operation_index_partition(tenant_id),
        limit=500,
        row_key_prefix=f"{workspace_id}:",
    )
    return [record.row_key for record in page.documents]


def _operation_without_created_at(
    *,
    operation_id: str,
    workspace_id: str,
    source_id: str = "source-1",
    status: WorkspaceOperationStatus = WorkspaceOperationStatus.QUEUED,
    operation_type: WorkspaceOperationType = WorkspaceOperationType.KNOWLEDGE_INGESTION,
) -> WorkspaceOperation:
    return WorkspaceOperation(
        operation_id=operation_id,
        tenant_id=_TENANT,
        workspace_id=workspace_id,
        source_id=source_id,
        operation_type=operation_type,
        status=status,
        created_at=None,
    )


def test_workspace_operations_list_uses_workspace_index_not_tenant_scan(
    api_bundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, repo, service, workspace_id, _ = api_bundle
    other = service.create_workspace(tenant_id=_TENANT, name="Bulk")
    for index in range(30):
        repo.put_operation(
            WorkspaceOperation(
                operation_id=f"op-b-{index:02d}",
                tenant_id=_TENANT,
                workspace_id=other.workspace_id,
                source_id="source-b",
                operation_type=WorkspaceOperationType.SOURCE_SYNC,
                status=WorkspaceOperationStatus.COMPLETED,
                created_at=datetime(2026, 8, 1, index % 24, 0, tzinfo=UTC),
            )
        )
    repo.put_operation(
        WorkspaceOperation(
            operation_id="op-a-only",
            tenant_id=_TENANT,
            workspace_id=workspace_id,
            source_id="source-1",
            operation_type=WorkspaceOperationType.SOURCE_SYNC,
            status=WorkspaceOperationStatus.COMPLETED,
            created_at=_NOW,
        )
    )

    list_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    original_list = repo.list_operations

    def tracked_list(*args: object, **kwargs: object) -> list[WorkspaceOperation]:
        list_calls.append((args, kwargs))
        return original_list(*args, **kwargs)

    query_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    original_query = repo.document_store.query

    def tracked_query(*args: object, **kwargs: object) -> object:
        query_calls.append((args, kwargs))
        return original_query(*args, **kwargs)

    monkeypatch.setattr(repo, "list_operations", tracked_list)
    monkeypatch.setattr(repo.document_store, "query", tracked_query)

    response = client.get(
        _operations_list_path(workspace_id),
        headers=_headers(),
        params={"limit": 10},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert [item["operation_id"] for item in body["operations"]] == ["op-a-only"]
    assert list_calls == []
    assert any(
        args[0] == _workspace_operation_index_partition()
        and kwargs.get("row_key_prefix") == f"{workspace_id}:"
        for args, kwargs in query_calls
    )


def test_workspace_operations_service_limit_caps_at_100(api_bundle) -> None:
    _, repo, service, workspace_id, _ = api_bundle
    for index in range(120):
        repo.put_operation(
            WorkspaceOperation(
                operation_id=f"op-cap-{index:03d}",
                tenant_id=_TENANT,
                workspace_id=workspace_id,
                source_id="source-1",
                operation_type=WorkspaceOperationType.SOURCE_SYNC,
                status=WorkspaceOperationStatus.COMPLETED,
                created_at=datetime(2026, 8, 1, 0, index % 60, tzinfo=UTC),
            )
        )

    operations = service.list_workspace_operations(
        tenant_id=_TENANT,
        workspace_id=workspace_id,
        limit=200,
    )

    assert len(operations) == 100


def test_workspace_operations_deterministic_tie_ordering(api_bundle) -> None:
    client, repo, _, workspace_id, _ = api_bundle
    tied = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
    repo.put_operation(
        WorkspaceOperation(
            operation_id="op-tie-b",
            tenant_id=_TENANT,
            workspace_id=workspace_id,
            source_id="source-1",
            operation_type=WorkspaceOperationType.SOURCE_SYNC,
            status=WorkspaceOperationStatus.COMPLETED,
            created_at=tied,
        )
    )
    repo.put_operation(
        WorkspaceOperation(
            operation_id="op-tie-a",
            tenant_id=_TENANT,
            workspace_id=workspace_id,
            source_id="source-1",
            operation_type=WorkspaceOperationType.SOURCE_SYNC,
            status=WorkspaceOperationStatus.COMPLETED,
            created_at=tied,
        )
    )

    first = client.get(
        _operations_list_path(workspace_id),
        headers=_headers(),
    ).json()["operations"]
    second = client.get(
        _operations_list_path(workspace_id),
        headers=_headers(),
    ).json()["operations"]

    assert [item["operation_id"] for item in first] == ["op-tie-a", "op-tie-b"]
    assert first == second


def test_workspace_operation_status_update_does_not_duplicate_index(api_bundle) -> None:
    client, repo, _, workspace_id, _ = api_bundle
    operation = WorkspaceOperation(
        operation_id="op-update",
        tenant_id=_TENANT,
        workspace_id=workspace_id,
        source_id="source-1",
        operation_type=WorkspaceOperationType.KNOWLEDGE_INGESTION,
        status=WorkspaceOperationStatus.QUEUED,
        created_at=_NOW,
    )
    repo.put_operation(operation)
    assert _count_workspace_operation_index_entries(repo, workspace_id=workspace_id) == 1

    repo.put_operation(
        operation.model_copy(
            update={
                "status": WorkspaceOperationStatus.FAILED,
                "error_code": "ingestion_failed",
            }
        )
    )
    assert _count_workspace_operation_index_entries(repo, workspace_id=workspace_id) == 1

    response = client.get(
        _operations_list_path(workspace_id),
        headers=_headers(),
    )
    assert response.status_code == 200
    listed = response.json()["operations"][0]
    assert listed["operation_id"] == "op-update"
    assert listed["status"] == "failed"
    assert listed["error_code"] == "ingestion_failed"


def test_workspace_operations_orphan_index_entry_skipped(api_bundle) -> None:
    client, repo, _, workspace_id, _ = api_bundle
    orphan = WorkspaceOperationIndexEntryV1(
        tenant_id=_TENANT,
        workspace_id=workspace_id,
        operation_id="op-orphan",
        created_at=_NOW,
    )
    repo.document_store.put(
        DocumentRecord(
            partition_key=_workspace_operation_index_partition(),
            row_key=orphan.index_row_key(),
            data=orphan.model_dump(mode="json"),
        )
    )
    repo.put_operation(
        WorkspaceOperation(
            operation_id="op-live",
            tenant_id=_TENANT,
            workspace_id=workspace_id,
            source_id="source-1",
            operation_type=WorkspaceOperationType.SOURCE_SYNC,
            status=WorkspaceOperationStatus.COMPLETED,
            created_at=_NOW,
        )
    )

    response = client.get(_operations_list_path(workspace_id), headers=_headers())

    assert response.status_code == 200
    assert [item["operation_id"] for item in response.json()["operations"]] == ["op-live"]


def test_workspace_delete_cleans_operation_index(api_bundle) -> None:
    _, repo, service, workspace_id, _ = api_bundle
    repo.put_operation(
        WorkspaceOperation(
            operation_id="op-delete-me",
            tenant_id=_TENANT,
            workspace_id=workspace_id,
            source_id="source-1",
            operation_type=WorkspaceOperationType.SOURCE_SYNC,
            status=WorkspaceOperationStatus.COMPLETED,
            created_at=_NOW,
        )
    )
    assert _count_workspace_operation_index_entries(repo, workspace_id=workspace_id) == 1

    assert service.delete_workspace(tenant_id=_TENANT, workspace_id=workspace_id) is True
    assert _count_workspace_operation_index_entries(repo, workspace_id=workspace_id) == 0
    assert (
        repo.get_operation(tenant_id=_TENANT, operation_id="op-delete-me") is None
    )


def test_workspace_operation_created_at_none_assigns_primary_and_index(api_bundle) -> None:
    _, repo, _, workspace_id, _ = api_bundle
    stored = repo.put_operation(
        _operation_without_created_at(operation_id="op-none-create", workspace_id=workspace_id)
    )

    assert stored.created_at is not None
    assert stored.created_at.tzinfo is UTC
    assert _count_workspace_operation_index_entries(repo, workspace_id=workspace_id) == 1


def test_workspace_operation_created_at_none_update_preserves_index_key(api_bundle) -> None:
    client, repo, _, workspace_id, _ = api_bundle
    created = repo.put_operation(
        _operation_without_created_at(operation_id="op-none-update", workspace_id=workspace_id)
    )
    index_keys_after_create = _workspace_operation_index_row_keys(
        repo,
        workspace_id=workspace_id,
    )

    updated = repo.put_operation(
        _operation_without_created_at(
            operation_id="op-none-update",
            workspace_id=workspace_id,
            status=WorkspaceOperationStatus.FAILED,
        ).model_copy(update={"error_code": "ingestion_failed"})
    )

    assert updated.created_at == created.created_at
    assert _count_workspace_operation_index_entries(repo, workspace_id=workspace_id) == 1
    assert _workspace_operation_index_row_keys(repo, workspace_id=workspace_id) == index_keys_after_create

    listed = client.get(_operations_list_path(workspace_id), headers=_headers()).json()["operations"][0]
    assert listed["operation_id"] == "op-none-update"
    assert listed["status"] == "failed"
    assert listed["error_code"] == "ingestion_failed"


def test_workspace_operation_repeated_none_created_at_updates_keep_single_index(api_bundle) -> None:
    _, repo, _, workspace_id, _ = api_bundle
    operation_id = "op-none-repeat"
    first = repo.put_operation(
        _operation_without_created_at(operation_id=operation_id, workspace_id=workspace_id)
    )
    index_keys_after_create = _workspace_operation_index_row_keys(
        repo,
        workspace_id=workspace_id,
    )

    for status in (
        WorkspaceOperationStatus.RUNNING,
        WorkspaceOperationStatus.PROCESSING,
        WorkspaceOperationStatus.COMPLETED,
    ):
        stored = repo.put_operation(
            _operation_without_created_at(
                operation_id=operation_id,
                workspace_id=workspace_id,
                status=status,
            )
        )
        assert stored.created_at == first.created_at

    assert _count_workspace_operation_index_entries(repo, workspace_id=workspace_id) == 1
    assert _workspace_operation_index_row_keys(repo, workspace_id=workspace_id) == index_keys_after_create


def test_workspace_operation_same_created_at_update_accepted(api_bundle) -> None:
    _, repo, _, workspace_id, _ = api_bundle
    created_at = datetime(2026, 8, 9, 9, 30, tzinfo=UTC)
    operation = WorkspaceOperation(
        operation_id="op-same-created-at",
        tenant_id=_TENANT,
        workspace_id=workspace_id,
        source_id="source-1",
        operation_type=WorkspaceOperationType.KNOWLEDGE_INGESTION,
        status=WorkspaceOperationStatus.QUEUED,
        created_at=created_at,
    )
    repo.put_operation(operation)
    index_keys_after_create = _workspace_operation_index_row_keys(
        repo,
        workspace_id=workspace_id,
    )

    updated = repo.put_operation(
        operation.model_copy(
            update={
                "status": WorkspaceOperationStatus.COMPLETED,
                "created_at": created_at,
            }
        )
    )

    assert updated.created_at == created_at
    assert _count_workspace_operation_index_entries(repo, workspace_id=workspace_id) == 1
    assert _workspace_operation_index_row_keys(repo, workspace_id=workspace_id) == index_keys_after_create


def test_workspace_operation_created_at_conflict_is_fail_closed(api_bundle) -> None:
    _, repo, _, workspace_id, _ = api_bundle
    created_at = datetime(2026, 8, 9, 9, 30, tzinfo=UTC)
    operation = WorkspaceOperation(
        operation_id="op-created-at-conflict",
        tenant_id=_TENANT,
        workspace_id=workspace_id,
        source_id="source-1",
        operation_type=WorkspaceOperationType.KNOWLEDGE_INGESTION,
        status=WorkspaceOperationStatus.QUEUED,
        created_at=created_at,
    )
    repo.put_operation(operation)
    index_keys_after_create = _workspace_operation_index_row_keys(
        repo,
        workspace_id=workspace_id,
    )

    with pytest.raises(RuntimeError, match="workspace_operation_created_at_conflict"):
        repo.put_operation(
            operation.model_copy(
                update={
                    "status": WorkspaceOperationStatus.FAILED,
                    "created_at": datetime(2026, 8, 9, 10, 0, tzinfo=UTC),
                }
            )
        )

    stored = repo.get_operation(tenant_id=_TENANT, operation_id="op-created-at-conflict")
    assert stored is not None
    assert stored.created_at == created_at
    assert stored.status is WorkspaceOperationStatus.QUEUED
    assert _count_workspace_operation_index_entries(repo, workspace_id=workspace_id) == 1
    assert _workspace_operation_index_row_keys(repo, workspace_id=workspace_id) == index_keys_after_create


def test_workspace_operation_identity_conflict_is_fail_closed(api_bundle) -> None:
    _, repo, service, workspace_id, _ = api_bundle
    other_workspace = service.create_workspace(tenant_id=_TENANT, name="Other")
    operation = WorkspaceOperation(
        operation_id="op-identity-conflict",
        tenant_id=_TENANT,
        workspace_id=workspace_id,
        source_id="source-1",
        operation_type=WorkspaceOperationType.KNOWLEDGE_INGESTION,
        status=WorkspaceOperationStatus.QUEUED,
        created_at=_NOW,
    )
    repo.put_operation(operation)
    index_keys_after_create = _workspace_operation_index_row_keys(
        repo,
        workspace_id=workspace_id,
    )

    with pytest.raises(RuntimeError, match="workspace_operation_identity_conflict"):
        repo.put_operation(
            operation.model_copy(update={"workspace_id": other_workspace.workspace_id})
        )

    stored = repo.get_operation(tenant_id=_TENANT, operation_id="op-identity-conflict")
    assert stored is not None
    assert stored.workspace_id == workspace_id
    assert _count_workspace_operation_index_entries(repo, workspace_id=workspace_id) == 1
    assert _count_workspace_operation_index_entries(
        repo,
        workspace_id=other_workspace.workspace_id,
    ) == 0
    assert _workspace_operation_index_row_keys(repo, workspace_id=workspace_id) == index_keys_after_create


def test_historical_operation_direct_get_without_index(api_bundle) -> None:
    client, repo, _, workspace_id, _ = api_bundle
    historical = WorkspaceOperation(
        operation_id="op-historical",
        tenant_id=_TENANT,
        workspace_id=workspace_id,
        source_id="source-1",
        operation_type=WorkspaceOperationType.SOURCE_SYNC,
        status=WorkspaceOperationStatus.COMPLETED,
        created_at=_NOW,
    )
    repo.document_store.put(
        DocumentRecord(
            partition_key=_operation_partition(),
            row_key=historical.operation_id,
            data=historical.model_dump(mode="json"),
        )
    )

    get_response = client.get(f"{_PREFIX}/operations/op-historical", headers=_headers())
    list_response = client.get(_operations_list_path(workspace_id), headers=_headers())

    assert get_response.status_code == 200
    assert get_response.json()["operation_id"] == "op-historical"
    assert list_response.status_code == 200
    assert list_response.json()["operations"] == []

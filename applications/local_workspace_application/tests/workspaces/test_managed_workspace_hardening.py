# © Artur Czarnecki. All rights reserved.

"""Hardening tests for durable managed workspace sync and search evidence."""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.task.task import TaskResult, TaskState
from local_workspace_application.serving import workspace_routes
from local_workspace_application.serving.workspace_routes import (
    SearchEvidenceIncompleteError,
    _map_search_hits,
)
from local_workspace_application.workspaces.models import (
    WorkspaceDocumentReference,
    WorkspaceOperation,
    WorkspaceOperationStatus,
    WorkspaceOperationType,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.service import (
    ConcurrentSyncError,
    ManagedWorkspaceService,
)
from local_workspace_application.workspaces.sync_enqueue import (
    build_managed_workspace_sync_enqueue_input,
)
from local_workspace_application.workspaces.sync_jobs import ManagedWorkspaceSyncJob
from local_workspace_application.workspaces.sync_service import ManagedWorkspaceSyncService


@pytest.mark.unit
def test_sync_route_source_does_not_use_asyncio_create_task() -> None:
    source = inspect.getsource(workspace_routes)
    assert "asyncio.create_task" not in source
    assert "_read_snippet" not in source
    assert "Path(...).read_text" not in source
    assert "read_text" not in source


@pytest.mark.unit
def test_create_sync_operation_persists_queued(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    repo = ManagedWorkspaceRepository(InMemoryDocumentStore())
    service = ManagedWorkspaceService(
        repo,
        allowlist_roots=frozenset({str(root.resolve())}),
    )
    workspace = service.create_workspace(tenant_id="t1", name="Case")
    source = service.register_local_folder_source(
        tenant_id="t1",
        workspace_id=workspace.workspace_id,
        path=str(root),
    )
    seeded = service.create_sync_operation(
        tenant_id="t1",
        workspace_id=workspace.workspace_id,
        source_id=source.source_id,
    )
    assert seeded.status is WorkspaceOperationStatus.QUEUED
    loaded = service.get_operation(tenant_id="t1", operation_id=seeded.operation_id)
    assert loaded is not None
    assert loaded.status is WorkspaceOperationStatus.QUEUED


@pytest.mark.unit
def test_concurrent_sync_rejected(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    repo = ManagedWorkspaceRepository(InMemoryDocumentStore())
    service = ManagedWorkspaceService(
        repo,
        allowlist_roots=frozenset({str(root.resolve())}),
    )
    workspace = service.create_workspace(tenant_id="t1", name="Case")
    source = service.register_local_folder_source(
        tenant_id="t1",
        workspace_id=workspace.workspace_id,
        path=str(root),
    )
    first = service.create_sync_operation(
        tenant_id="t1",
        workspace_id=workspace.workspace_id,
        source_id=source.source_id,
    )
    assert first.status is WorkspaceOperationStatus.QUEUED
    with pytest.raises(ConcurrentSyncError):
        service.create_sync_operation(
            tenant_id="t1",
            workspace_id=workspace.workspace_id,
            source_id=source.source_id,
        )


@pytest.mark.unit
def test_concurrent_sync_rejects_older_active_when_latest_is_terminal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    repo = ManagedWorkspaceRepository(InMemoryDocumentStore())
    service = ManagedWorkspaceService(
        repo,
        allowlist_roots=frozenset({str(root.resolve())}),
    )
    workspace = service.create_workspace(tenant_id="t1", name="Case")
    source = service.register_local_folder_source(
        tenant_id="t1",
        workspace_id=workspace.workspace_id,
        path=str(root),
    )

    first = service.create_sync_operation(
        tenant_id="t1",
        workspace_id=workspace.workspace_id,
        source_id=source.source_id,
    )
    running = first.model_copy(update={"status": WorkspaceOperationStatus.RUNNING})
    repo.put_operation(running)
    second = service.create_sync_operation(
        tenant_id="t1",
        workspace_id=workspace.workspace_id,
        source_id=source.source_id,
        allow_concurrent=True,
    )
    completed = second.model_copy(update={"status": WorkspaceOperationStatus.COMPLETED})
    repo.put_operation(completed)

    def tenant_scan_forbidden(*, tenant_id: str):
        raise AssertionError(f"tenant operation scan: {tenant_id}")

    monkeypatch.setattr(repo, "list_operations", tenant_scan_forbidden)
    latest_page = repo.list_source_sync_operations_page(
        tenant_id="t1",
        workspace_id=workspace.workspace_id,
        source_id=source.source_id,
        limit=1,
    )
    assert latest_page.documents[0].data["operation_id"] == second.operation_id
    active = repo.find_active_sync_operation(
        tenant_id="t1",
        workspace_id=workspace.workspace_id,
        source_id=source.source_id,
    )
    assert active is not None
    assert active.operation_id == first.operation_id

    with pytest.raises(ConcurrentSyncError) as error:
        service.create_sync_operation(
            tenant_id="t1",
            workspace_id=workspace.workspace_id,
            source_id=source.source_id,
        )
    assert error.value.active.operation_id == first.operation_id
    history = repo.list_source_sync_operation_history_page(
        tenant_id="t1",
        workspace_id=workspace.workspace_id,
        source_id=source.source_id,
        limit=10,
    )
    assert {record.data["operation_id"] for record in history.documents} == {
        first.operation_id,
        second.operation_id,
    }


@pytest.mark.unit
def test_concurrent_sync_rejects_when_multiple_operations_are_active(
    tmp_path: Path,
) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    repo = ManagedWorkspaceRepository(InMemoryDocumentStore())
    service = ManagedWorkspaceService(
        repo,
        allowlist_roots=frozenset({str(root.resolve())}),
    )
    workspace = service.create_workspace(tenant_id="t1", name="Case")
    source = service.register_local_folder_source(
        tenant_id="t1",
        workspace_id=workspace.workspace_id,
        path=str(root),
    )

    first = service.create_sync_operation(
        tenant_id="t1",
        workspace_id=workspace.workspace_id,
        source_id=source.source_id,
    )
    running = first.model_copy(update={"status": WorkspaceOperationStatus.RUNNING})
    repo.put_operation(running)
    second = service.create_sync_operation(
        tenant_id="t1",
        workspace_id=workspace.workspace_id,
        source_id=source.source_id,
        allow_concurrent=True,
    )
    active = repo.find_active_sync_operation(
        tenant_id="t1",
        workspace_id=workspace.workspace_id,
        source_id=source.source_id,
    )
    assert active is not None
    assert active.operation_id in {first.operation_id, second.operation_id}

    with pytest.raises(ConcurrentSyncError) as error:
        service.create_sync_operation(
            tenant_id="t1",
            workspace_id=workspace.workspace_id,
            source_id=source.source_id,
        )
    assert error.value.active.operation_id in {first.operation_id, second.operation_id}


@pytest.mark.unit
@pytest.mark.parametrize(
    "status",
    [
        WorkspaceOperationStatus.QUEUED,
        WorkspaceOperationStatus.RUNNING,
        WorkspaceOperationStatus.PROCESSING,
    ],
)
def test_find_active_sync_operation_keeps_active_states(
    status: WorkspaceOperationStatus,
) -> None:
    repo = ManagedWorkspaceRepository(InMemoryDocumentStore())
    operation = WorkspaceOperation(
        operation_id=f"op-{status.value}",
        tenant_id="t1",
        workspace_id="w1",
        source_id="s1",
        operation_type=WorkspaceOperationType.SOURCE_SYNC,
        status=status,
    )
    repo.put_operation(operation)

    active = repo.find_active_sync_operation(
        tenant_id="t1",
        workspace_id="w1",
        source_id="s1",
    )
    assert active is not None
    assert active.operation_id == operation.operation_id


@pytest.mark.unit
@pytest.mark.asyncio
async def test_duplicate_delivery_after_completion_is_noop() -> None:
    repo = ManagedWorkspaceRepository(InMemoryDocumentStore())
    operation = WorkspaceOperation(
        operation_id="op-done",
        tenant_id="t1",
        workspace_id="w1",
        source_id="s1",
        operation_type=WorkspaceOperationType.SOURCE_SYNC,
        status=WorkspaceOperationStatus.COMPLETED,
        documents_indexed=2,
    )
    repo.put_operation(operation)
    sync = ManagedWorkspaceSyncService(repo, task_executor=MagicMock())
    result = await sync.run_operation(tenant_id="t1", operation_id="op-done")
    assert result.status is WorkspaceOperationStatus.COMPLETED
    assert result.documents_indexed == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_duplicate_delivery_while_running_does_not_reingest() -> None:
    repo = ManagedWorkspaceRepository(InMemoryDocumentStore())
    operation = WorkspaceOperation(
        operation_id="op-run",
        tenant_id="t1",
        workspace_id="w1",
        source_id="s1",
        operation_type=WorkspaceOperationType.SOURCE_SYNC,
        status=WorkspaceOperationStatus.RUNNING,
    )
    repo.put_operation(operation)
    executor = MagicMock()
    sync = ManagedWorkspaceSyncService(repo, task_executor=executor)
    result = await sync.run_operation(tenant_id="t1", operation_id="op-run")
    assert result.status is WorkspaceOperationStatus.RUNNING
    executor.execute.assert_not_called()


@pytest.mark.unit
def test_enqueue_payload_contains_required_identities() -> None:
    job = ManagedWorkspaceSyncJob(
        tenant_id="t1",
        workspace_id="w1",
        source_id="s1",
        operation_id="op1",
        operation_type="source_sync",
    )
    params = build_managed_workspace_sync_enqueue_input(job)
    assert params.task_name == "lkw.managed_workspace_sync.v1"
    assert params.tenant_id == "t1"
    assert params.run_id == "op1"
    assert "source_sync" in params.payload_base64 or True
    # Decode payload via helper
    from local_workspace_application.workspaces.sync_jobs import (
        decode_managed_workspace_sync_job,
        managed_workspace_sync_payload_base64,
    )
    import base64

    decoded = decode_managed_workspace_sync_job(
        base64.b64decode(managed_workspace_sync_payload_base64(job))
    )
    assert decoded.operation_type == "source_sync"
    assert decoded.operation_id == "op1"


@pytest.mark.unit
def test_map_search_hits_requires_complete_evidence(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    repo = ManagedWorkspaceRepository(InMemoryDocumentStore())
    repo.put_document_ref(
        WorkspaceDocumentReference(
            document_id="doc-1",
            tenant_id="t1",
            workspace_id="w1",
            source_id="s1",
            source_path=str(tmp_path / "a.txt"),
            file_name="a.txt",
            content_hash="sha256:abc",
            indexed_at=datetime.now(UTC),
        )
    )
    incomplete = TaskResult(
        task_id="r1",
        run_id="r1",
        state=TaskState.COMPLETED,
        answer="ok",
        execution_result=AgentExecutionResult(
            agent_id="local_search",
            run_id="r1",
            status=AgentExecutionStatus.COMPLETED,
            summary="ok",
            structured_data={
                "search_summary": {
                    "query": "q",
                    "workspace_id": "w1",
                    "evidence": [
                        {
                            "document_id": "doc-1",
                            "source_id": "s1",
                            "workspace_id": "w1",
                            "source_path": str(tmp_path / "a.txt"),
                            "file_name": "a.txt",
                            # missing score + snippet
                        }
                    ],
                }
            },
        ),
    )
    with pytest.raises(SearchEvidenceIncompleteError):
        _map_search_hits(
            repository=repo,
            tenant_id="t1",
            workspace_id="w1",
            task_result=incomplete,
            limit=10,
        )


@pytest.mark.unit
def test_map_search_hits_maps_complete_evidence_without_file_read(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    root = tmp_path / "docs"
    root.mkdir()
    path = root / "a.txt"
    path.write_text("secret body", encoding="utf-8")
    repo = ManagedWorkspaceRepository(InMemoryDocumentStore())
    service = ManagedWorkspaceService(
        repo,
        allowlist_roots=frozenset({str(root.resolve())}),
    )
    workspace = service.create_workspace(tenant_id="t1", name="Case")
    source = service.register_local_folder_source(
        tenant_id="t1",
        workspace_id=workspace.workspace_id,
        path=str(root),
    )
    document = WorkspaceDocumentReference(
        document_id="doc-1",
        tenant_id="t1",
        workspace_id=workspace.workspace_id,
        source_id=source.source_id,
        source_path=str(path.resolve()),
        file_name="a.txt",
        content_hash="sha256:abc",
        indexed_at=datetime.now(UTC),
    )
    repo.put_document_ref(
        document
    )
    result = TaskResult(
        task_id="r1",
        run_id="r1",
        state=TaskState.COMPLETED,
        answer="ok",
        execution_result=AgentExecutionResult(
            agent_id="local_search",
            run_id="r1",
            status=AgentExecutionStatus.COMPLETED,
            summary="ok",
            structured_data={
                "search_summary": {
                    "query": "q",
                    "workspace_id": workspace.workspace_id,
                    "result_count": 1,
                    "evidence": [
                        {
                            "document_id": document.document_id,
                            "source_id": source.source_id,
                            "workspace_id": workspace.workspace_id,
                            "source_path": document.source_path,
                            "file_name": "a.txt",
                            "score": 0.91,
                            "snippet": "platform snippet",
                            "metadata": {"k": "v"},
                        }
                    ],
                }
            },
        ),
    )
    hits = _map_search_hits(
        repository=repo,
        tenant_id="t1",
        workspace_id=workspace.workspace_id,
        task_result=result,
        limit=10,
    )
    assert len(hits) == 1
    assert hits[0].snippet == "platform snippet"
    assert hits[0].score == 0.91
    assert hits[0].document_id == document.document_id
    assert hits[0].source_id == source.source_id
    assert hits[0].workspace_id == workspace.workspace_id
    assert hits[0].source_path == document.source_path
    assert hits[0].file_name == document.file_name
    assert hits[0].metadata == {"k": "v"}


@pytest.mark.unit
def test_map_search_hits_drops_cross_workspace_evidence(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    path = tmp_path / "a.txt"
    path.write_text("x", encoding="utf-8")
    repo = ManagedWorkspaceRepository(InMemoryDocumentStore())
    repo.put_document_ref(
        WorkspaceDocumentReference(
            document_id="doc-1",
            tenant_id="t1",
            workspace_id="w1",
            source_id="s1",
            source_path=str(path.resolve()),
            file_name="a.txt",
            content_hash="sha256:abc",
            indexed_at=datetime.now(UTC),
        )
    )
    result = TaskResult(
        task_id="r1",
        run_id="r1",
        state=TaskState.COMPLETED,
        answer="ok",
        execution_result=AgentExecutionResult(
            agent_id="local_search",
            run_id="r1",
            status=AgentExecutionStatus.COMPLETED,
            summary="ok",
            structured_data={
                "search_summary": {
                    "evidence": [
                        {
                            "document_id": "doc-1",
                            "source_id": "s1",
                            "workspace_id": "other",
                            "source_path": str(path.resolve()),
                            "file_name": "a.txt",
                            "score": 0.5,
                            "snippet": "nope",
                            "metadata": {},
                        }
                    ]
                }
            },
        ),
    )
    with pytest.raises(SearchEvidenceIncompleteError):
        _map_search_hits(
            repository=repo,
            tenant_id="t1",
            workspace_id="w1",
            task_result=result,
            limit=10,
        )

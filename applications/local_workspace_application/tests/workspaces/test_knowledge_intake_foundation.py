# © Artur Czarnecki. All rights reserved.

"""Durable Knowledge Intake foundation (LKW-WORKSPACE-CONTENTS-1B-1)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.queueing.contracts.task_queue import TaskHandle, TaskQueue, TaskRequest, TaskResult, TaskStatus
from intergrax.queueing.providers.document_store import DocumentStoreTaskQueue
from intergrax.queueing.providers.document_store.colocated_worker import DocumentStoreTaskWorker
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.tools.registry.wiring import ToolWiringContext
from local_workspace_application.workspaces.ask_repository import WorkspaceAskRepository
from local_workspace_application.workspaces.knowledge_ingestion import (
    KnowledgeIngestionResult,
    KnowledgeIngestionService,
    register_knowledge_ingestion_worker_handler,
)
from local_workspace_application.workspaces.knowledge_intake import (
    KnowledgeInputIdempotencyConflict,
    KnowledgeIntakeDispatchError,
    KnowledgeIntakeService,
)
from local_workspace_application.workspaces.models import (
    KnowledgeInput,
    KnowledgeInputKind,
    KnowledgeInputStatus,
    Workspace,
    WorkspaceOperation,
    WorkspaceOperationStatus,
    WorkspaceOperationType,
    WorkspaceSource,
    WorkspaceSourceStatus,
    WorkspaceSourceType,
    WorkspaceStatus,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.service import ManagedWorkspaceService

pytestmark = [pytest.mark.unit, pytest.mark.gate]

TENANT = "tenant-a"
OTHER_TENANT = "tenant-b"
WORKSPACE = "workspace-a"


def _now() -> datetime:
    return datetime.now(UTC)


def _seed_workspace(repo: ManagedWorkspaceRepository, *, tenant_id: str = TENANT) -> Workspace:
    workspace = Workspace(
        workspace_id=WORKSPACE,
        tenant_id=tenant_id,
        name="Demo",
        status=WorkspaceStatus.ACTIVE,
        created_at=_now(),
        updated_at=_now(),
    )
    return repo.put_workspace(workspace)


class _FakeResolver:
    def __init__(self) -> None:
        self.calls = 0
        self.created_ids: list[str] = []

    def resolve(
        self,
        *,
        knowledge_input: KnowledgeInput,
        suggested_source_id: str,
    ) -> WorkspaceSource:
        self.calls += 1
        self.created_ids.append(suggested_source_id)
        return WorkspaceSource(
            source_id=suggested_source_id,
            workspace_id=knowledge_input.workspace_id,
            tenant_id=knowledge_input.tenant_id,
            source_type=WorkspaceSourceType.MANAGED_UPLOAD,
            path="",
            recursive=False,
            status=WorkspaceSourceStatus.REGISTERED,
            created_at=_now(),
        )


class _FakeProcessor:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    async def process(
        self,
        *,
        knowledge_input: KnowledgeInput,
        source: WorkspaceSource,
        operation: WorkspaceOperation,
    ) -> KnowledgeIngestionResult:
        _ = knowledge_input, source, operation
        self.calls += 1
        if self.fail:
            raise RuntimeError("processor boom\ntraceback-line")
        return KnowledgeIngestionResult(
            files_processed=1,
            files_failed=0,
            documents_indexed=1,
            documents_unchanged=0,
        )


class _RaisingMessageBus(TaskQueue):
    def enqueue(self, request: TaskRequest) -> TaskHandle:
        _ = request
        raise RuntimeError("bus down")

    def get_status(self, handle: TaskHandle) -> TaskStatus:
        _ = handle
        return TaskStatus.PENDING

    def get_result(self, handle: TaskHandle) -> TaskResult | None:
        _ = handle
        return None


def _build_stack(
    *,
    store: InMemoryDocumentStore | None = None,
    processor: _FakeProcessor | None = None,
    resolver: _FakeResolver | None = None,
    message_bus: TaskQueue | None = None,
) -> tuple[
    InMemoryDocumentStore,
    ManagedWorkspaceRepository,
    KnowledgeIntakeService,
    KnowledgeIngestionService,
    DocumentStoreTaskQueue,
    DocumentStoreTaskWorker,
    _FakeResolver,
    _FakeProcessor,
]:
    store = store or InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    queue = message_bus if isinstance(message_bus, DocumentStoreTaskQueue) else DocumentStoreTaskQueue(store)
    bus = message_bus or queue
    ctx = ToolWiringContext(message_bus=bus)
    resolver = resolver or _FakeResolver()
    processor = processor or _FakeProcessor()
    intake = KnowledgeIntakeService(repo, resolver, ctx)
    ingestion = KnowledgeIngestionService(repo, processor)
    registry = TaskExecutionRegistry()
    register_knowledge_ingestion_worker_handler(registry, ingestion)
    worker = DocumentStoreTaskWorker(queue if isinstance(bus, DocumentStoreTaskQueue) else queue, registry)
    if not isinstance(bus, DocumentStoreTaskQueue):
        # Raising bus path still needs a DocumentStoreTaskQueue reference for typing; unused.
        queue = DocumentStoreTaskQueue(store)
        worker = DocumentStoreTaskWorker(queue, registry)
    return store, repo, intake, ingestion, queue, worker, resolver, processor


def test_model_compatibility_local_folder_and_source_sync() -> None:
    source = WorkspaceSource(
        source_id="s1",
        workspace_id=WORKSPACE,
        tenant_id=TENANT,
        source_type=WorkspaceSourceType.LOCAL_FOLDER,
        path="D:/docs",
        recursive=True,
        status=WorkspaceSourceStatus.REGISTERED,
        created_at=_now(),
    )
    assert source.path == "D:/docs"

    non_local = WorkspaceSource(
        source_id="s2",
        workspace_id=WORKSPACE,
        tenant_id=TENANT,
        source_type=WorkspaceSourceType.MANAGED_UPLOAD,
        path="",
        recursive=False,
        status=WorkspaceSourceStatus.REGISTERED,
        created_at=_now(),
    )
    assert non_local.source_type is WorkspaceSourceType.MANAGED_UPLOAD

    with pytest.raises(ValidationError):
        WorkspaceSource(
            source_id="s3",
            workspace_id=WORKSPACE,
            tenant_id=TENANT,
            source_type=WorkspaceSourceType.MANAGED_UPLOAD,
            path="C:/leak",
            recursive=False,
            status=WorkspaceSourceStatus.REGISTERED,
            created_at=_now(),
        )

    with pytest.raises(ValidationError):
        WorkspaceSource(
            source_id="s4",
            workspace_id=WORKSPACE,
            tenant_id=TENANT,
            source_type=WorkspaceSourceType.WEB_RESOURCE,
            path="",
            recursive=True,
            status=WorkspaceSourceStatus.REGISTERED,
            created_at=_now(),
        )

    operation = WorkspaceOperation(
        operation_id="op-1",
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        source_id="s1",
        operation_type=WorkspaceOperationType.SOURCE_SYNC,
        status=WorkspaceOperationStatus.QUEUED,
    )
    assert operation.input_id is None
    assert operation.created_at is None


def test_durable_acceptance_persists_queued_operation() -> None:
    _, repo, intake, _, queue, _, _, _ = _build_stack()
    _seed_workspace(repo)

    result = intake.accept(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        input_kind=KnowledgeInputKind.MANAGED_FILE,
        idempotency_key="key-1",
        submission_metadata={"label": "demo"},
    )

    loaded_input = repo.get_knowledge_input(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        input_id=result.knowledge_input.input_id,
    )
    loaded_source = repo.get_source(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        source_id=result.source.source_id,
    )
    loaded_op = repo.get_operation(
        tenant_id=TENANT,
        operation_id=result.operation.operation_id,
    )
    assert loaded_input is not None
    assert loaded_source is not None
    assert loaded_op is not None
    assert loaded_op.status is WorkspaceOperationStatus.QUEUED
    assert loaded_op.queue_task_id
    assert loaded_op.queue_provider
    assert loaded_op.input_id == loaded_input.input_id
    assert queue.get_status(
        TaskHandle(
            task_id=loaded_op.queue_task_id,
            provider=loaded_op.queue_provider or "",
            tenant_id=TENANT,
        )
    ) is TaskStatus.PENDING


def test_deterministic_idempotency() -> None:
    _, repo, intake, _, _, _, resolver, _ = _build_stack()
    _seed_workspace(repo)

    first = intake.accept(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        input_kind=KnowledgeInputKind.MANAGED_FILE,
        idempotency_key="same-key",
        submission_metadata={"label": "a"},
    )
    second = intake.accept(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        input_kind=KnowledgeInputKind.MANAGED_FILE,
        idempotency_key="same-key",
        submission_metadata={"label": "a"},
    )

    assert first.knowledge_input.input_id == second.knowledge_input.input_id
    assert first.source.source_id == second.source.source_id
    assert first.operation.operation_id == second.operation.operation_id
    assert first.operation.queue_task_id == second.operation.queue_task_id
    assert resolver.calls == 1
    assert len(repo.list_knowledge_inputs(tenant_id=TENANT, workspace_id=WORKSPACE)) == 1
    assert (
        len(
            repo.list_ingestion_operations(
                tenant_id=TENANT,
                workspace_id=WORKSPACE,
            )
        )
        == 1
    )


def test_idempotency_conflict_different_kind_or_metadata() -> None:
    _, repo, intake, _, _, _, _, _ = _build_stack()
    _seed_workspace(repo)

    intake.accept(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        input_kind=KnowledgeInputKind.MANAGED_FILE,
        idempotency_key="conflict-key",
        submission_metadata={"label": "a"},
    )
    with pytest.raises(KnowledgeInputIdempotencyConflict):
        intake.accept(
            tenant_id=TENANT,
            workspace_id=WORKSPACE,
            input_kind=KnowledgeInputKind.WEB_URL,
            idempotency_key="conflict-key",
            submission_metadata={"label": "a"},
        )
    with pytest.raises(KnowledgeInputIdempotencyConflict):
        intake.accept(
            tenant_id=TENANT,
            workspace_id=WORKSPACE,
            input_kind=KnowledgeInputKind.MANAGED_FILE,
            idempotency_key="conflict-key",
            submission_metadata={"label": "b"},
        )
    assert len(repo.list_knowledge_inputs(tenant_id=TENANT, workspace_id=WORKSPACE)) == 1
    assert len(repo.list_ingestion_operations(tenant_id=TENANT, workspace_id=WORKSPACE)) == 1


def test_end_to_end_queue_execution() -> None:
    _, repo, intake, _, queue, worker, _, processor = _build_stack()
    _seed_workspace(repo)

    accepted = intake.accept(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        input_kind=KnowledgeInputKind.MANAGED_FILE,
        idempotency_key="e2e",
        submission_metadata={"label": "e2e"},
    )
    handle = TaskHandle(
        task_id=accepted.operation.queue_task_id or "",
        provider=accepted.operation.queue_provider or "",
        tenant_id=TENANT,
    )
    assert queue.get_status(handle) is TaskStatus.PENDING
    assert worker.drain_once() == 1
    assert processor.calls == 1

    operation = repo.get_operation(
        tenant_id=TENANT,
        operation_id=accepted.operation.operation_id,
    )
    source = repo.get_source(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        source_id=accepted.source.source_id,
    )
    assert operation is not None
    assert operation.status is WorkspaceOperationStatus.COMPLETED
    assert source is not None
    assert source.status is WorkspaceSourceStatus.READY
    assert queue.get_status(handle) is TaskStatus.SUCCEEDED


def test_processor_failure_persists_failed_operation() -> None:
    _, repo, intake, _, queue, worker, _, processor = _build_stack(
        processor=_FakeProcessor(fail=True),
    )
    _seed_workspace(repo)

    accepted = intake.accept(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        input_kind=KnowledgeInputKind.MANAGED_FILE,
        idempotency_key="fail",
        submission_metadata={"label": "fail"},
    )
    handle = TaskHandle(
        task_id=accepted.operation.queue_task_id or "",
        provider=accepted.operation.queue_provider or "",
        tenant_id=TENANT,
    )
    assert worker.drain_once() == 1
    assert processor.calls == 1

    operation = repo.get_operation(
        tenant_id=TENANT,
        operation_id=accepted.operation.operation_id,
    )
    source = repo.get_source(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        source_id=accepted.source.source_id,
    )
    assert operation is not None
    assert operation.status is WorkspaceOperationStatus.FAILED
    assert operation.error_code == "processor_failed"
    assert operation.error is not None
    assert "traceback" not in operation.error
    assert "\n" not in operation.error
    assert source is not None
    assert source.status is WorkspaceSourceStatus.ERROR
    assert queue.get_status(handle) in {TaskStatus.SUCCEEDED, TaskStatus.FAILED}


def test_duplicate_delivery_completed_skips_processor() -> None:
    _, repo, intake, ingestion, _, worker, _, processor = _build_stack()
    _seed_workspace(repo)
    accepted = intake.accept(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        input_kind=KnowledgeInputKind.MANAGED_FILE,
        idempotency_key="dup-completed",
    )
    assert worker.drain_once() == 1
    assert processor.calls == 1

    again = asyncio_run(
        ingestion.run_operation(
            tenant_id=TENANT,
            operation_id=accepted.operation.operation_id,
        )
    )
    assert again.status is WorkspaceOperationStatus.COMPLETED
    assert processor.calls == 1


def test_duplicate_delivery_processing_skips_processor() -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    _seed_workspace(repo)
    processor = _FakeProcessor()
    ingestion = KnowledgeIngestionService(repo, processor)

    knowledge_input = KnowledgeInput(
        input_id="ki:processing",
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        input_kind=KnowledgeInputKind.MANAGED_FILE,
        idempotency_key="proc",
        operation_id="op:processing",
        source_id="src:processing",
        status=KnowledgeInputStatus.RESOLVED,
        submission_metadata={},
        created_at=_now(),
        updated_at=_now(),
    )
    source = WorkspaceSource(
        source_id="src:processing",
        workspace_id=WORKSPACE,
        tenant_id=TENANT,
        source_type=WorkspaceSourceType.MANAGED_UPLOAD,
        status=WorkspaceSourceStatus.PROCESSING,
        created_at=_now(),
    )
    operation = WorkspaceOperation(
        operation_id="op:processing",
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        source_id="src:processing",
        operation_type=WorkspaceOperationType.KNOWLEDGE_INGESTION,
        status=WorkspaceOperationStatus.PROCESSING,
        input_id="ki:processing",
        created_at=_now(),
        started_at=_now(),
    )
    repo.put_knowledge_input(knowledge_input)
    repo.put_source(source)
    repo.put_operation(operation)

    result = asyncio_run(
        ingestion.run_operation(tenant_id=TENANT, operation_id="op:processing")
    )
    assert result.status is WorkspaceOperationStatus.PROCESSING
    assert processor.calls == 0


def test_tenant_isolation() -> None:
    _, repo, intake, _, _, _, _, _ = _build_stack()
    _seed_workspace(repo)

    accepted = intake.accept(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        input_kind=KnowledgeInputKind.MANAGED_FILE,
        idempotency_key="iso",
    )
    assert (
        repo.get_knowledge_input(
            tenant_id=OTHER_TENANT,
            workspace_id=WORKSPACE,
            input_id=accepted.knowledge_input.input_id,
        )
        is None
    )
    assert (
        repo.get_source(
            tenant_id=OTHER_TENANT,
            workspace_id=WORKSPACE,
            source_id=accepted.source.source_id,
        )
        is None
    )
    assert (
        repo.get_operation(
            tenant_id=OTHER_TENANT,
            operation_id=accepted.operation.operation_id,
        )
        is None
    )


def test_enqueue_failure_leaves_durable_failed_operation() -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    _seed_workspace(repo)
    resolver = _FakeResolver()
    ctx = ToolWiringContext(message_bus=_RaisingMessageBus())
    intake = KnowledgeIntakeService(repo, resolver, ctx)

    with pytest.raises(KnowledgeIntakeDispatchError):
        intake.accept(
            tenant_id=TENANT,
            workspace_id=WORKSPACE,
            input_kind=KnowledgeInputKind.MANAGED_FILE,
            idempotency_key="enqueue-fail",
            submission_metadata={"label": "x"},
        )

    inputs = repo.list_knowledge_inputs(tenant_id=TENANT, workspace_id=WORKSPACE)
    sources = repo.list_sources(tenant_id=TENANT, workspace_id=WORKSPACE)
    operations = repo.list_ingestion_operations(tenant_id=TENANT, workspace_id=WORKSPACE)
    assert len(inputs) == 1
    assert len(sources) == 1
    assert len(operations) == 1
    assert operations[0].status is WorkspaceOperationStatus.FAILED
    assert operations[0].error_code == "enqueue_failed"


def test_reconcile_workspace_requeues_accepted_operation() -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    _seed_workspace(repo)
    resolver = _FakeResolver()
    processor = _FakeProcessor()

    knowledge_input = KnowledgeInput(
        input_id="ki:reconcile",
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        input_kind=KnowledgeInputKind.MANAGED_FILE,
        idempotency_key="reconcile",
        operation_id="op:reconcile",
        source_id="src:reconcile",
        status=KnowledgeInputStatus.RESOLVED,
        submission_metadata={},
        created_at=_now(),
        updated_at=_now(),
    )
    source = WorkspaceSource(
        source_id="src:reconcile",
        workspace_id=WORKSPACE,
        tenant_id=TENANT,
        source_type=WorkspaceSourceType.MANAGED_UPLOAD,
        status=WorkspaceSourceStatus.REGISTERED,
        created_at=_now(),
    )
    operation = WorkspaceOperation(
        operation_id="op:reconcile",
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        source_id="src:reconcile",
        operation_type=WorkspaceOperationType.KNOWLEDGE_INGESTION,
        status=WorkspaceOperationStatus.ACCEPTED,
        input_id="ki:reconcile",
        created_at=_now(),
    )
    repo.put_knowledge_input(knowledge_input)
    repo.put_source(source)
    repo.put_operation(operation)

    queue = DocumentStoreTaskQueue(store)
    ctx = ToolWiringContext(message_bus=queue)
    intake = KnowledgeIntakeService(repo, resolver, ctx)
    ingestion = KnowledgeIngestionService(repo, processor)
    registry = TaskExecutionRegistry()
    register_knowledge_ingestion_worker_handler(registry, ingestion)
    worker = DocumentStoreTaskWorker(queue, registry)

    resumed = intake.reconcile_workspace(tenant_id=TENANT, workspace_id=WORKSPACE)
    assert resumed >= 1
    loaded = repo.get_operation(tenant_id=TENANT, operation_id="op:reconcile")
    assert loaded is not None
    assert loaded.status is WorkspaceOperationStatus.QUEUED
    assert loaded.queue_task_id
    first_task_id = loaded.queue_task_id

    resumed_again = intake.reconcile_workspace(tenant_id=TENANT, workspace_id=WORKSPACE)
    loaded_again = repo.get_operation(tenant_id=TENANT, operation_id="op:reconcile")
    assert loaded_again is not None
    assert loaded_again.queue_task_id == first_task_id
    assert resumed_again == 0 or loaded_again.queue_task_id == first_task_id

    assert worker.drain_once() == 1
    final = repo.get_operation(tenant_id=TENANT, operation_id="op:reconcile")
    assert final is not None
    assert final.status is WorkspaceOperationStatus.COMPLETED


def test_workspace_deletion_removes_knowledge_inputs() -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    ask_repo = WorkspaceAskRepository(store)
    service = ManagedWorkspaceService(repo, ask_repository=ask_repo)
    _seed_workspace(repo)

    queue = DocumentStoreTaskQueue(store)
    intake = KnowledgeIntakeService(
        repo,
        _FakeResolver(),
        ToolWiringContext(message_bus=queue),
    )
    accepted = intake.accept(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        input_kind=KnowledgeInputKind.MANAGED_FILE,
        idempotency_key="delete-me",
    )
    assert service.delete_workspace(tenant_id=TENANT, workspace_id=WORKSPACE) is True
    assert (
        repo.get_knowledge_input(
            tenant_id=TENANT,
            workspace_id=WORKSPACE,
            input_id=accepted.knowledge_input.input_id,
        )
        is None
    )


def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)

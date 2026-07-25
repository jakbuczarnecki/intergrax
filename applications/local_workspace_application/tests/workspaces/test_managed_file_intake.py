# © Artur Czarnecki. All rights reserved.

"""Managed-file Knowledge Intake (LKW-WORKSPACE-CONTENTS-1B-2)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.object_storage import StoredObject
from intergrax.queueing.contracts.task_queue import TaskHandle, TaskStatus
from intergrax.queueing.providers.document_store import DocumentStoreTaskQueue
from intergrax.queueing.providers.document_store.colocated_worker import DocumentStoreTaskWorker
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.tools.registry.wiring import ToolWiringContext
from local_workspace_application.workspaces.document_indexing import (
    WorkspaceDocumentIndexingResult,
)
from local_workspace_application.workspaces.ingestion_recovery import (
    KnowledgeIngestionRecoveryService,
)
from local_workspace_application.workspaces.knowledge_ingestion import (
    KnowledgeIngestionService,
    LKW_KNOWLEDGE_INGESTION_TASK_NAME,
    register_knowledge_ingestion_worker_handler,
)
from local_workspace_application.workspaces.knowledge_intake import KnowledgeIntakeService
from local_workspace_application.workspaces.managed_file_ingestion import (
    ManagedFileKnowledgeIngestionProcessor,
    ManagedObjectMaterializer,
)
from local_workspace_application.workspaces.managed_files import (
    IntakeBatchIdempotencyConflict,
    ManagedFileIdempotencyConflict,
    ManagedFileIntakeService,
    ManagedFileObjectCleanup,
    ManagedFileSourceResolver,
    ManagedFileUpload,
    ManagedFileValidationError,
)
from local_workspace_application.workspaces.models import (
    ActiveKnowledgeIngestionLocator,
    IntakeBatchStatus,
    ManagedFileObjectStatus,
    Workspace,
    WorkspaceOperationStatus,
    WorkspaceOperationType,
    WorkspaceSourceStatus,
    WorkspaceStatus,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.service import ManagedWorkspaceService
from local_workspace_application.workspaces.sync_jobs import LKW_MANAGED_WORKSPACE_SYNC_TASK_NAME
from local_workspace_application.workspaces.sync_runtime import build_managed_workspace_sync_runtime
from local_workspace_application.workspaces.sync_service import ManagedWorkspaceSyncService

pytestmark = [pytest.mark.unit, pytest.mark.gate]

TENANT = "tenant-a"
WORKSPACE = "workspace-a"


def _now() -> datetime:
    return datetime.now(UTC)


def _seed_workspace(repo: ManagedWorkspaceRepository) -> Workspace:
    workspace = Workspace(
        workspace_id=WORKSPACE,
        tenant_id=TENANT,
        name="Demo",
        status=WorkspaceStatus.ACTIVE,
        created_at=_now(),
        updated_at=_now(),
    )
    return repo.put_workspace(workspace)


class FakeObjectStorage:
    def __init__(self) -> None:
        self.objects: dict[str, StoredObject] = {}
        self.fail_put = False
        self.fail_delete_keys: set[str] = set()

    def put(
        self,
        key: str,
        body: bytes,
        *,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> None:
        if self.fail_put:
            raise RuntimeError("storage unavailable")
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
        if key in self.fail_delete_keys:
            raise RuntimeError("delete failed")
        self.objects.pop(key, None)

    def presigned_url(
        self,
        key: str,
        *,
        expires_in_seconds: int = 3600,
        method: str = "GET",
    ) -> str:
        _ = expires_in_seconds, method
        return f"memory://{key}"

    def close(self) -> None:
        return None


class SpyIndexingService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def index_one(self, **kwargs: object) -> WorkspaceDocumentIndexingResult:
        self.calls.append(kwargs)
        return WorkspaceDocumentIndexingResult(
            indexed=True,
            unchanged=False,
            document_id="doc-1",
            documents_indexed=1,
            num_chunks=2,
            reason="ingest_complete",
        )


def _build_intake(
    *,
    storage: FakeObjectStorage | None = None,
    indexing: SpyIndexingService | None = None,
    max_bytes: int = 1024 * 1024,
    max_batch_files: int = 20,
) -> tuple[
    ManagedWorkspaceRepository,
    FakeObjectStorage,
    ManagedFileIntakeService,
    DocumentStoreTaskQueue,
    DocumentStoreTaskWorker,
    KnowledgeIngestionService,
    SpyIndexingService,
]:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    storage = storage or FakeObjectStorage()
    queue = DocumentStoreTaskQueue(store)
    ctx = ToolWiringContext(message_bus=queue)
    resolver = ManagedFileSourceResolver(repo)
    intake_svc = KnowledgeIntakeService(repo, resolver, ctx)
    indexing = indexing or SpyIndexingService()
    materializer = ManagedObjectMaterializer(storage, Path("build/test_staging").resolve())
    processor = ManagedFileKnowledgeIngestionProcessor(repo, materializer, indexing)  # type: ignore[arg-type]
    ingestion = KnowledgeIngestionService(repo, processor)
    registry = TaskExecutionRegistry()
    register_knowledge_ingestion_worker_handler(registry, ingestion)
    worker = DocumentStoreTaskWorker(queue, registry)
    managed = ManagedFileIntakeService(
        repo,
        storage,
        intake_svc,
        max_bytes=max_bytes,
        max_batch_files=max_batch_files,
    )
    _seed_workspace(repo)
    return repo, storage, managed, queue, worker, ingestion, indexing


def test_validation_safe_names_and_rejects() -> None:
    _, _, managed, _, _, _, _ = _build_intake()
    managed.accept_one(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="k-pdf",
        upload=ManagedFileUpload("contract.pdf", "application/pdf", b"%PDF-1"),
    )
    managed.accept_one(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="k-docx",
        upload=ManagedFileUpload(
            "brief.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            b"PK",
        ),
    )
    with pytest.raises(ManagedFileValidationError, match="managed_file_name_unsafe"):
        managed.accept_one(
            tenant_id=TENANT,
            workspace_id=WORKSPACE,
            idempotency_key="slash",
            upload=ManagedFileUpload("a/b.pdf", "application/pdf", b"x"),
        )
    with pytest.raises(ManagedFileValidationError, match="managed_file_name_unsafe"):
        managed.accept_one(
            tenant_id=TENANT,
            workspace_id=WORKSPACE,
            idempotency_key="bslash",
            upload=ManagedFileUpload("a\\b.pdf", "application/pdf", b"x"),
        )
    with pytest.raises(ManagedFileValidationError, match="managed_file_name_unsafe"):
        managed.accept_one(
            tenant_id=TENANT,
            workspace_id=WORKSPACE,
            idempotency_key="dot",
            upload=ManagedFileUpload(".", "application/pdf", b"x"),
        )
    with pytest.raises(ManagedFileValidationError, match="managed_file_name_unsafe"):
        managed.accept_one(
            tenant_id=TENANT,
            workspace_id=WORKSPACE,
            idempotency_key="ddot",
            upload=ManagedFileUpload("..", "application/pdf", b"x"),
        )
    with pytest.raises(ManagedFileValidationError, match="managed_file_extension_required"):
        managed.accept_one(
            tenant_id=TENANT,
            workspace_id=WORKSPACE,
            idempotency_key="noext",
            upload=ManagedFileUpload("readme", "text/plain", b"x"),
        )
    with pytest.raises(ManagedFileValidationError, match="managed_file_empty"):
        managed.accept_one(
            tenant_id=TENANT,
            workspace_id=WORKSPACE,
            idempotency_key="empty",
            upload=ManagedFileUpload("a.pdf", "application/pdf", b""),
        )
    _, _, managed_small, _, _, _, _ = _build_intake(max_bytes=5)
    with pytest.raises(ManagedFileValidationError, match="managed_file_too_large"):
        managed_small.accept_one(
            tenant_id=TENANT,
            workspace_id=WORKSPACE,
            idempotency_key="big2",
            upload=ManagedFileUpload("a.pdf", "application/pdf", b"xxxxxx"),
        )
    with pytest.raises(ManagedFileValidationError, match="managed_file_content_type_invalid"):
        managed.accept_one(
            tenant_id=TENANT,
            workspace_id=WORKSPACE,
            idempotency_key="ctype",
            upload=ManagedFileUpload("a.pdf", "text/plain\x00evil", b"x"),
        )


def test_single_file_persistence_and_queue() -> None:
    repo, storage, managed, queue, _, _, _ = _build_intake()
    body = b"%PDF-demo"
    acceptance = managed.accept_one(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="one",
        upload=ManagedFileUpload("contract.pdf", "application/pdf", body),
    )
    stored = storage.get(acceptance.managed_file.storage_key)
    assert stored is not None
    assert stored.body == body
    assert "contract.pdf" not in acceptance.managed_file.storage_key
    assert repo.get_managed_file(
        tenant_id=TENANT, workspace_id=WORKSPACE, input_id=acceptance.knowledge_input.input_id
    ) is not None
    assert repo.get_knowledge_input(
        tenant_id=TENANT, workspace_id=WORKSPACE, input_id=acceptance.knowledge_input.input_id
    ) is not None
    assert acceptance.source.path == ""
    assert acceptance.operation.status in {
        WorkspaceOperationStatus.QUEUED,
        WorkspaceOperationStatus.ACCEPTED,
    }
    assert acceptance.operation.queue_task_id
    assert (
        queue.get_status(
            TaskHandle(
                task_id=acceptance.operation.queue_task_id,
                provider=acceptance.operation.queue_provider or "",
                tenant_id=TENANT,
            )
        )
        is TaskStatus.PENDING
    )


def test_atomic_retry_and_conflict() -> None:
    repo, storage, managed, _, _, _, _ = _build_intake()
    upload = ManagedFileUpload("contract.pdf", "application/pdf", b"%PDF-1")
    first = managed.accept_one(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="retry",
        upload=upload,
    )
    second = managed.accept_one(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="retry",
        upload=upload,
    )
    assert first.managed_file.object_id == second.managed_file.object_id
    assert first.managed_file.storage_key == second.managed_file.storage_key
    assert first.knowledge_input.input_id == second.knowledge_input.input_id
    assert first.source.source_id == second.source.source_id
    assert first.operation.operation_id == second.operation.operation_id
    assert first.operation.queue_task_id == second.operation.queue_task_id
    assert len(repo.list_sources(tenant_id=TENANT, workspace_id=WORKSPACE)) == 1
    original = storage.get(first.managed_file.storage_key)
    assert original is not None
    with pytest.raises(ManagedFileIdempotencyConflict):
        managed.accept_one(
            tenant_id=TENANT,
            workspace_id=WORKSPACE,
            idempotency_key="retry",
            upload=ManagedFileUpload("contract.pdf", "application/pdf", b"%PDF-2"),
        )
    assert storage.get(first.managed_file.storage_key).body == original.body  # type: ignore[union-attr]


def test_missing_object_recovery() -> None:
    _, storage, managed, _, _, _, _ = _build_intake()
    upload = ManagedFileUpload("contract.pdf", "application/pdf", b"%PDF-1")
    first = managed.accept_one(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="missing",
        upload=upload,
    )
    storage.objects.pop(first.managed_file.storage_key)
    second = managed.accept_one(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="missing",
        upload=upload,
    )
    assert second.managed_file.storage_key == first.managed_file.storage_key
    assert storage.get(first.managed_file.storage_key) is not None
    assert second.source.source_id == first.source.source_id
    assert second.operation.operation_id == first.operation.operation_id


def test_object_storage_write_failure_no_domain_records() -> None:
    storage = FakeObjectStorage()
    storage.fail_put = True
    repo, _, managed, queue, _, _, _ = _build_intake(storage=storage)
    with pytest.raises(ManagedFileValidationError, match="managed_file_storage_write_failed"):
        managed.accept_one(
            tenant_id=TENANT,
            workspace_id=WORKSPACE,
            idempotency_key="fail-put",
            upload=ManagedFileUpload("a.pdf", "application/pdf", b"x"),
        )
    assert repo.list_knowledge_inputs(tenant_id=TENANT, workspace_id=WORKSPACE) == []
    assert repo.list_sources(tenant_id=TENANT, workspace_id=WORKSPACE) == []
    assert [
        op
        for op in repo.list_operations(tenant_id=TENANT)
        if op.operation_type is WorkspaceOperationType.KNOWLEDGE_INGESTION
    ] == []
    assert (
        [
            op
            for op in repo.list_operations(tenant_id=TENANT)
            if op.queue_task_id
        ]
        == []
    )


def test_multi_file_and_partial_batch() -> None:
    repo, _, managed, queue, _, _, _ = _build_intake()
    batch = managed.accept_many(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="batch-3",
        uploads=[
            ManagedFileUpload("a.pdf", "application/pdf", b"a"),
            ManagedFileUpload("b.pdf", "application/pdf", b"b"),
            ManagedFileUpload("c.pdf", "application/pdf", b"c"),
        ],
    )
    assert batch.status is IntakeBatchStatus.ACCEPTED
    assert len(batch.items) == 3
    assert len(repo.list_managed_files(tenant_id=TENANT, workspace_id=WORKSPACE)) == 3
    assert len(repo.list_knowledge_inputs(tenant_id=TENANT, workspace_id=WORKSPACE)) == 3
    assert len(repo.list_sources(tenant_id=TENANT, workspace_id=WORKSPACE)) == 3
    ops = [
        op
        for op in repo.list_operations(tenant_id=TENANT)
        if op.operation_type is WorkspaceOperationType.KNOWLEDGE_INGESTION
    ]
    assert len(ops) == 3
    assert len({op.operation_id for op in ops}) == 3
    assert len({op.queue_task_id for op in ops}) == 3

    partial = managed.accept_many(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="batch-partial",
        uploads=[
            ManagedFileUpload("ok1.pdf", "application/pdf", b"1"),
            ManagedFileUpload("bad.pdf", "application/pdf", b""),
            ManagedFileUpload("ok2.pdf", "application/pdf", b"2"),
        ],
    )
    assert partial.status is IntakeBatchStatus.PARTIAL
    assert sum(1 for i in partial.items if i.status.value == "accepted") == 2
    assert sum(1 for i in partial.items if i.status.value == "failed") == 1
    failed = next(i for i in partial.items if i.status.value == "failed")
    assert failed.error_code == "managed_file_empty"


def test_batch_retry_and_conflict() -> None:
    _, storage, managed, _, _, _, _ = _build_intake()
    uploads = [
        ManagedFileUpload("a.pdf", "application/pdf", b"a"),
        ManagedFileUpload("b.pdf", "application/pdf", b"b"),
    ]
    first = managed.accept_many(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="batch-retry",
        uploads=uploads,
    )
    second = managed.accept_many(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="batch-retry",
        uploads=uploads,
    )
    assert first.batch_id == second.batch_id
    assert [i.input_id for i in first.items] == [i.input_id for i in second.items]
    key = first.items[0].operation_id
    assert key
    with pytest.raises(IntakeBatchIdempotencyConflict):
        managed.accept_many(
            tenant_id=TENANT,
            workspace_id=WORKSPACE,
            idempotency_key="batch-retry",
            uploads=[
                ManagedFileUpload("a.pdf", "application/pdf", b"CHANGED"),
                ManagedFileUpload("b.pdf", "application/pdf", b"b"),
            ],
        )
    # accepted object bytes unchanged
    mf = storage.get(
        managed._repository.get_managed_file(  # noqa: SLF001
            tenant_id=TENANT,
            workspace_id=WORKSPACE,
            input_id=first.items[0].input_id or "",
        ).storage_key  # type: ignore[union-attr]
    )
    assert mf is not None
    assert mf.body == b"a"


def test_materialization_success_and_cleanup(tmp_path: Path) -> None:
    storage = FakeObjectStorage()
    staging = tmp_path / "staging"
    staging.mkdir()
    materializer = ManagedObjectMaterializer(storage, staging)
    _, _, managed, _, _, _, _ = _build_intake(storage=storage)
    acceptance = managed.accept_one(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="mat",
        upload=ManagedFileUpload("contract.pdf", "application/pdf", b"%PDF"),
    )
    seen: list[Path] = []
    with materializer.materialize(managed_file=acceptance.managed_file) as path:
        seen.append(path)
        assert path.exists()
        assert path.name == "contract.pdf"
        assert path.read_bytes() == b"%PDF"
        assert staging.resolve() in path.resolve().parents
    assert not seen[0].exists()
    assert not any(staging.iterdir())


def test_hash_mismatch_and_missing_object_processor(tmp_path: Path) -> None:
    repo, storage, managed, queue, worker, _, indexing = _build_intake()
    materializer = ManagedObjectMaterializer(storage, tmp_path / "staging")
    processor = ManagedFileKnowledgeIngestionProcessor(repo, materializer, indexing)  # type: ignore[arg-type]
    ingestion = KnowledgeIngestionService(repo, processor)
    registry = TaskExecutionRegistry()
    register_knowledge_ingestion_worker_handler(registry, ingestion)
    worker = DocumentStoreTaskWorker(queue, registry)

    acceptance = managed.accept_one(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="hash",
        upload=ManagedFileUpload("contract.pdf", "application/pdf", b"%PDF"),
    )
    storage.objects[acceptance.managed_file.storage_key] = StoredObject(
        key=acceptance.managed_file.storage_key,
        body=b"XXXX",  # same length as b"%PDF", different bytes
        content_type="application/pdf",
        size_bytes=4,
    )
    assert worker.drain_once() == 1
    op = repo.get_operation(tenant_id=TENANT, operation_id=acceptance.operation.operation_id)
    assert op is not None
    assert op.status is WorkspaceOperationStatus.FAILED
    assert op.error_code == "managed_object_hash_mismatch"
    source = repo.get_source(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        source_id=acceptance.source.source_id,
    )
    assert source is not None
    assert source.status is WorkspaceSourceStatus.ERROR
    assert (
        queue.get_status(
            TaskHandle(
                task_id=acceptance.operation.queue_task_id or "",
                provider=acceptance.operation.queue_provider or "",
                tenant_id=TENANT,
            )
        )
        is TaskStatus.SUCCEEDED
    )
    assert not any(tmp_path.rglob("contract.pdf"))

    acceptance2 = managed.accept_one(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="missing-obj",
        upload=ManagedFileUpload("gone.pdf", "application/pdf", b"%PDF2"),
    )
    storage.objects.pop(acceptance2.managed_file.storage_key)
    assert worker.drain_once() == 1
    op2 = repo.get_operation(tenant_id=TENANT, operation_id=acceptance2.operation.operation_id)
    assert op2 is not None
    assert op2.error_code == "managed_object_missing"
    mf = repo.get_managed_file(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        input_id=acceptance2.knowledge_input.input_id,
    )
    assert mf is not None
    assert mf.status is ManagedFileObjectStatus.MISSING


def test_processor_uses_shared_indexing_only(tmp_path: Path) -> None:
    repo, storage, managed, queue, _, _, indexing = _build_intake()
    materializer = ManagedObjectMaterializer(storage, tmp_path / "staging")
    processor = ManagedFileKnowledgeIngestionProcessor(repo, materializer, indexing)  # type: ignore[arg-type]
    ingestion = KnowledgeIngestionService(repo, processor)
    registry = TaskExecutionRegistry()
    register_knowledge_ingestion_worker_handler(registry, ingestion)
    worker = DocumentStoreTaskWorker(queue, registry)
    acceptance = managed.accept_one(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="e2e",
        upload=ManagedFileUpload("contract.pdf", "application/pdf", b"%PDF"),
    )
    assert worker.drain_once() == 1
    assert len(indexing.calls) == 1
    op = repo.get_operation(tenant_id=TENANT, operation_id=acceptance.operation.operation_id)
    assert op is not None
    assert op.status is WorkspaceOperationStatus.COMPLETED
    source = repo.get_source(
        tenant_id=TENANT, workspace_id=WORKSPACE, source_id=acceptance.source.source_id
    )
    assert source is not None
    assert source.status is WorkspaceSourceStatus.READY
    assert op.files_processed == 1
    assert op.documents_indexed == 1
    locators = [
        loc
        for loc in repo.list_active_ingestion_locators()
        if loc.operation_id == acceptance.operation.operation_id
    ]
    assert locators == []


def test_shared_runtime_registers_both_handlers() -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    _seed_workspace(repo)
    sync = ManagedWorkspaceSyncService(repo, type("E", (), {"execute": lambda *a, **k: None})())
    runtime = build_managed_workspace_sync_runtime(
        document_store=store,
        sync_service=sync,
        repository=repo,
    )
    storage = FakeObjectStorage()
    indexing = SpyIndexingService()
    processor = ManagedFileKnowledgeIngestionProcessor(
        repo,
        ManagedObjectMaterializer(storage, Path("build/staging").resolve()),
        indexing,  # type: ignore[arg-type]
    )
    ingestion = KnowledgeIngestionService(repo, processor)
    runtime.register_knowledge_ingestion_service(ingestion)
    # Both handlers share one registry/worker/queue.
    assert runtime.registry.get_handler(LKW_MANAGED_WORKSPACE_SYNC_TASK_NAME) is not None
    assert runtime.registry.get_handler(LKW_KNOWLEDGE_INGESTION_TASK_NAME) is not None
    with pytest.raises(RuntimeError, match="knowledge_ingestion_already_registered"):
        runtime.register_knowledge_ingestion_service(ingestion)


def test_startup_recovery() -> None:
    repo, storage, managed, queue, _, ingestion, indexing = _build_intake()
    intake = KnowledgeIntakeService(
        repo,
        ManagedFileSourceResolver(repo),
        ToolWiringContext(message_bus=queue),
    )
    recovery = KnowledgeIngestionRecoveryService(repo, intake)

    a = managed.accept_one(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="rec-a",
        upload=ManagedFileUpload("a.pdf", "application/pdf", b"a"),
    )
    q = managed.accept_one(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="rec-q",
        upload=ManagedFileUpload("q.pdf", "application/pdf", b"q"),
    )
    p = managed.accept_one(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="rec-p",
        upload=ManagedFileUpload("p.pdf", "application/pdf", b"p"),
    )
    repo.put_operation(
        p.operation.model_copy(update={"status": WorkspaceOperationStatus.PROCESSING})
    )
    completed = managed.accept_one(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="rec-c",
        upload=ManagedFileUpload("c.pdf", "application/pdf", b"c"),
    )
    repo.put_operation(
        completed.operation.model_copy(
            update={
                "status": WorkspaceOperationStatus.COMPLETED,
                "completed_at": _now(),
            }
        )
    )
    repo.put_active_ingestion_locator(
        ActiveKnowledgeIngestionLocator(
            operation_id=completed.operation.operation_id,
            tenant_id=TENANT,
            workspace_id=WORKSPACE,
            created_at=_now(),
        )
    )
    repo.put_active_ingestion_locator(
        ActiveKnowledgeIngestionLocator(
            operation_id="missing-op",
            tenant_id=TENANT,
            workspace_id=WORKSPACE,
            created_at=_now(),
        )
    )

    before_tasks = {
        op.queue_task_id
        for op in repo.list_operations(tenant_id=TENANT)
        if op.queue_task_id
    }
    result = recovery.recover_all()
    assert result.locators_seen >= 4
    assert result.processing_failed >= 1
    assert result.stale_locators_removed >= 2
    failed_p = repo.get_operation(tenant_id=TENANT, operation_id=p.operation.operation_id)
    assert failed_p is not None
    assert failed_p.status is WorkspaceOperationStatus.FAILED
    assert failed_p.error_code == "interrupted_by_host_restart"
    after_tasks = {
        op.queue_task_id
        for op in repo.list_operations(tenant_id=TENANT)
        if op.queue_task_id
    }
    # reconcile may keep same idempotent task ids
    assert a.operation.queue_task_id in after_tasks
    assert q.operation.queue_task_id in after_tasks
    _ = before_tasks, ingestion, indexing, storage


def test_workspace_deletion_cleanup() -> None:
    repo, storage, managed, _, _, _, _ = _build_intake()
    managed.accept_one(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="del-1",
        upload=ManagedFileUpload("a.pdf", "application/pdf", b"a"),
    )
    managed.accept_one(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="del-2",
        upload=ManagedFileUpload("b.pdf", "application/pdf", b"b"),
    )
    assert len(storage.objects) == 2
    cleanup = ManagedFileObjectCleanup(repo, storage)
    service = ManagedWorkspaceService(repo, managed_file_cleanup=cleanup)
    assert service.delete_workspace(tenant_id=TENANT, workspace_id=WORKSPACE) is True
    assert storage.objects == {}
    assert repo.list_managed_files(tenant_id=TENANT, workspace_id=WORKSPACE) == []
    assert repo.list_intake_batches(tenant_id=TENANT, workspace_id=WORKSPACE) == []
    assert repo.get_workspace(tenant_id=TENANT, workspace_id=WORKSPACE) is None

    _seed_workspace(repo)
    managed.accept_one(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="del-fail",
        upload=ManagedFileUpload("c.pdf", "application/pdf", b"c"),
    )
    key = next(iter(storage.objects))
    storage.fail_delete_keys.add(key)
    cleanup2 = ManagedFileObjectCleanup(repo, storage)
    service2 = ManagedWorkspaceService(repo, managed_file_cleanup=cleanup2)
    with pytest.raises(RuntimeError, match="workspace_managed_file_cleanup_failed"):
        service2.delete_workspace(tenant_id=TENANT, workspace_id=WORKSPACE)
    assert repo.get_workspace(tenant_id=TENANT, workspace_id=WORKSPACE) is not None
    assert repo.list_sources(tenant_id=TENANT, workspace_id=WORKSPACE)
    assert repo.list_knowledge_inputs(tenant_id=TENANT, workspace_id=WORKSPACE)


def test_sync_service_uses_indexing_service() -> None:
    repo = ManagedWorkspaceRepository(InMemoryDocumentStore())
    spy = SpyIndexingService()
    sync = ManagedWorkspaceSyncService(
        repo,
        type("E", (), {"execute": lambda *a, **k: None})(),
        indexing_service=spy,  # type: ignore[arg-type]
    )
    assert sync._indexing_service is spy  # noqa: SLF001

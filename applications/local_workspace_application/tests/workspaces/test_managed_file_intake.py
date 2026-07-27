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
    WorkspaceDocumentIndexingError,
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
from local_workspace_application.workspaces.knowledge_intake import (
    KnowledgeInputResolutionError,
    KnowledgeIntakeService,
)
from local_workspace_application.workspaces.managed_file_ingestion import (
    ManagedFileKnowledgeIngestionProcessor,
    ManagedObjectMaterializer,
)
from local_workspace_application.workspaces.managed_files import (
    IntakeBatchIdempotencyConflict,
    ManagedFileBatchCandidate,
    ManagedFileIdempotencyConflict,
    ManagedFileIntakeService,
    ManagedFileObjectCleanup,
    ManagedFileSourceResolver,
    ManagedFileUpload,
    ManagedFileValidationError,
    managed_file_request_fingerprint,
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


def test_failed_empty_item_changed_on_retry_conflicts() -> None:
    repo, storage, managed, _, _, _, _ = _build_intake()
    first = managed.accept_many(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="failed-retry",
        uploads=[ManagedFileUpload("a.pdf", "application/pdf", b"")],
    )
    assert first.status is IntakeBatchStatus.FAILED
    assert first.items[0].error_code == "managed_file_empty"
    assert first.items[0].request_fingerprint.startswith("sha256:")
    fp = first.items[0].request_fingerprint
    assert "a.pdf" not in fp
    assert "application/pdf" not in fp

    with pytest.raises(IntakeBatchIdempotencyConflict):
        managed.accept_many(
            tenant_id=TENANT,
            workspace_id=WORKSPACE,
            idempotency_key="failed-retry",
            uploads=[ManagedFileUpload("a.pdf", "application/pdf", b"%PDF-nonempty")],
        )
    assert repo.list_managed_files(tenant_id=TENANT, workspace_id=WORKSPACE) == []
    assert repo.list_sources(tenant_id=TENANT, workspace_id=WORKSPACE) == []
    assert [
        op
        for op in repo.list_operations(tenant_id=TENANT)
        if op.operation_type is WorkspaceOperationType.KNOWLEDGE_INGESTION
    ] == []
    assert storage.objects == {}


def test_failed_invalid_filename_changed_on_retry_conflicts() -> None:
    _, _, managed, _, _, _, _ = _build_intake()
    first = managed.accept_many(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="bad-name-retry",
        uploads=[ManagedFileUpload("bad/name.pdf", "application/pdf", b"x")],
    )
    assert first.items[0].error_code == "managed_file_name_unsafe"
    assert first.items[0].safe_file_name == "rejected-item-0.bin"
    with pytest.raises(IntakeBatchIdempotencyConflict):
        managed.accept_many(
            tenant_id=TENANT,
            workspace_id=WORKSPACE,
            idempotency_key="bad-name-retry",
            uploads=[ManagedFileUpload("different/name.pdf", "application/pdf", b"x")],
        )


def test_exact_failed_retry_is_idempotent() -> None:
    repo, _, managed, _, _, _, _ = _build_intake()
    upload = ManagedFileUpload("a.pdf", "application/pdf", b"")
    first = managed.accept_many(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="exact-fail",
        uploads=[upload],
    )
    second = managed.accept_many(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="exact-fail",
        uploads=[upload],
    )
    assert first.batch_id == second.batch_id
    assert first.items[0].item_id == second.items[0].item_id
    assert first.items[0].request_fingerprint == second.items[0].request_fingerprint
    assert first.items[0].error_code == second.items[0].error_code == "managed_file_empty"
    assert repo.list_managed_files(tenant_id=TENANT, workspace_id=WORKSPACE) == []
    assert [
        op
        for op in repo.list_operations(tenant_id=TENANT)
        if op.operation_type is WorkspaceOperationType.KNOWLEDGE_INGESTION
    ] == []


def test_content_type_conflict_under_same_batch_key() -> None:
    _, _, managed, _, _, _, _ = _build_intake()
    managed.accept_many(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="ctype-conflict",
        uploads=[ManagedFileUpload("a.pdf", "application/pdf", b"abc")],
    )
    with pytest.raises(IntakeBatchIdempotencyConflict):
        managed.accept_many(
            tenant_id=TENANT,
            workspace_id=WORKSPACE,
            idempotency_key="ctype-conflict",
            uploads=[ManagedFileUpload("a.pdf", "text/plain", b"abc")],
        )


def test_input_boundary_rejects_non_bytes_and_forced_error_code() -> None:
    _, _, managed, _, _, _, _ = _build_intake()
    assert "forced_error_code" not in ManagedFileUpload.__dataclass_fields__
    for body in (bytearray(b"x"), memoryview(b"x"), "x"):  # type: ignore[arg-type]
        with pytest.raises(ManagedFileValidationError, match="managed_file_body_required"):
            managed.accept_one(
                tenant_id=TENANT,
                workspace_id=WORKSPACE,
                idempotency_key=f"body-{type(body).__name__}",
                upload=ManagedFileUpload("a.pdf", "application/pdf", body),  # type: ignore[arg-type]
            )


def test_unsafe_filename_not_persisted_or_returned() -> None:
    repo, _, managed, _, _, _, _ = _build_intake()
    unsafe = "evil/../secret.pdf"
    batch = managed.accept_many(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="unsafe-name",
        uploads=[ManagedFileUpload(unsafe, "application/pdf", b"x")],
    )
    assert batch.items[0].safe_file_name == "rejected-item-0.bin"
    dumped = str(batch.model_dump())
    assert unsafe not in dumped
    assert "evil" not in dumped
    assert "secret.pdf" not in dumped
    assert repo.list_knowledge_inputs(tenant_id=TENANT, workspace_id=WORKSPACE) == []
    assert repo.list_sources(tenant_id=TENANT, workspace_id=WORKSPACE) == []


def test_unexpected_acceptance_exception_is_safe() -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    storage = FakeObjectStorage()
    _seed_workspace(repo)

    class BoomIntake:
        def accept(self, **kwargs: object) -> object:
            _ = kwargs
            raise RuntimeError("s3://private-bucket/key credential=secret")

    managed = ManagedFileIntakeService(
        repo,
        storage,
        BoomIntake(),  # type: ignore[arg-type]
        max_bytes=1024 * 1024,
        max_batch_files=20,
    )
    body = b"%PDF"
    body_hash = f"sha256:{__import__('hashlib').sha256(body).hexdigest()}"
    batch = managed.accept_prepared_many(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="boom-batch",
        candidates=[
            ManagedFileBatchCandidate(
                raw_file_name="a.pdf",
                raw_content_type="application/pdf",
                body=body,
                size_bytes=len(body),
                body_hash=body_hash,
                request_fingerprint=managed_file_request_fingerprint(
                    raw_file_name="a.pdf",
                    raw_content_type="application/pdf",
                    size_bytes=len(body),
                    body_hash=body_hash,
                ),
            )
        ],
    )
    assert batch.items[0].error_code == "managed_file_accept_failed"
    text = str(batch.model_dump())
    for forbidden in ("s3://", "private-bucket", "credential", "secret", "RuntimeError"):
        assert forbidden not in text


def test_existing_object_storage_read_failure() -> None:
    class BoomGetStorage(FakeObjectStorage):
        def get(self, key: str) -> StoredObject | None:
            _ = key
            raise RuntimeError("bucket=private key=lkw/managed/x token=secret")

    storage = BoomGetStorage()
    _, _, managed, _, _, _, _ = _build_intake(storage=storage)
    upload = ManagedFileUpload("a.pdf", "application/pdf", b"%PDF-1")
    first = managed.accept_one(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="read-fail",
        upload=upload,
    )
    # Force retry path against existing managed object.
    with pytest.raises(ManagedFileValidationError, match="managed_file_storage_read_failed"):
        managed.accept_one(
            tenant_id=TENANT,
            workspace_id=WORKSPACE,
            idempotency_key="read-fail",
            upload=upload,
        )
    mf = managed._repository.get_managed_file(  # noqa: SLF001
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        input_id=first.knowledge_input.input_id,
    )
    assert mf is not None
    assert mf.status is ManagedFileObjectStatus.ERROR
    assert mf.error_code == "managed_file_storage_read_failed"
    assert "secret" not in str(mf.model_dump())
    assert "bucket=private" not in str(mf.model_dump())


def test_materializer_storage_read_failure_through_worker(tmp_path: Path) -> None:
    class BoomGetStorage(FakeObjectStorage):
        def get(self, key: str) -> StoredObject | None:
            _ = key
            raise RuntimeError("bucket=private key=lkw/managed/x token=secret")

    storage = BoomGetStorage()
    repo, _, managed, queue, worker, _, _ = _build_intake(storage=storage)
    acceptance = managed.accept_one(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="mat-read-fail",
        upload=ManagedFileUpload("contract.pdf", "application/pdf", b"%PDF"),
    )
    assert worker.drain_once() == 1
    _ = tmp_path
    op = repo.get_operation(tenant_id=TENANT, operation_id=acceptance.operation.operation_id)
    assert op is not None
    assert op.status is WorkspaceOperationStatus.FAILED
    assert op.error_code == "managed_object_read_failed"
    assert op.error == "managed_object_read_failed"
    source = repo.get_source(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        source_id=acceptance.source.source_id,
    )
    assert source is not None
    assert source.status is WorkspaceSourceStatus.ERROR
    mf = repo.get_managed_file(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        input_id=acceptance.knowledge_input.input_id,
    )
    assert mf is not None
    assert mf.status is ManagedFileObjectStatus.ERROR
    assert mf.error_code == "managed_object_read_failed"
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
    assert not repo.list_active_ingestion_locators()
    durable = (
        str(op.error)
        + str(op.error_code)
        + str(mf.error_code)
        + str(source.status.value)
    )
    for forbidden in ("bucket=private", "token=secret", "RuntimeError"):
        assert forbidden not in durable


def test_malformed_locator_recovery_isolation() -> None:
    from intergrax.integrations.contracts.document_store import DocumentRecord

    repo, _, managed, queue, _, _, _ = _build_intake()
    intake = KnowledgeIntakeService(
        repo,
        ManagedFileSourceResolver(repo),
        ToolWiringContext(message_bus=queue),
    )
    recovery = KnowledgeIngestionRecoveryService(repo, intake)

    accepted = managed.accept_one(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="rec-ok",
        upload=ManagedFileUpload("ok.pdf", "application/pdf", b"ok"),
    )
    processing = managed.accept_one(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="rec-proc",
        upload=ManagedFileUpload("p.pdf", "application/pdf", b"p"),
    )
    repo.put_operation(
        processing.operation.model_copy(update={"status": WorkspaceOperationStatus.PROCESSING})
    )
    terminal = managed.accept_one(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="rec-term",
        upload=ManagedFileUpload("t.pdf", "application/pdf", b"t"),
    )
    repo.put_operation(
        terminal.operation.model_copy(
            update={
                "status": WorkspaceOperationStatus.COMPLETED,
                "completed_at": _now(),
            }
        )
    )
    repo.put_active_ingestion_locator(
        ActiveKnowledgeIngestionLocator(
            operation_id=terminal.operation.operation_id,
            tenant_id=TENANT,
            workspace_id=WORKSPACE,
            created_at=_now(),
        )
    )
    partition = "lkw.managed_workspace:active_knowledge_ingestion"
    repo.document_store.put(
        DocumentRecord(
            partition_key=partition,
            row_key="malformed-missing-tenant",
            data={"operation_id": "malformed-1", "workspace_id": WORKSPACE},
        )
    )

    result = recovery.recover_all()
    assert result.errors >= 1
    assert result.stale_locators_removed >= 1
    assert result.processing_failed >= 1
    failed_p = repo.get_operation(
        tenant_id=TENANT, operation_id=processing.operation.operation_id
    )
    assert failed_p is not None
    assert failed_p.status is WorkspaceOperationStatus.FAILED
    assert failed_p.error_code == "interrupted_by_host_restart"
    assert accepted.operation.queue_task_id
    remaining = {
        loc.operation_id for loc in repo.list_active_ingestion_locators()
    }
    assert "malformed-missing-tenant" not in remaining
    assert terminal.operation.operation_id not in remaining


def test_malformed_locator_delete_failure_still_recovers() -> None:
    from intergrax.integrations.contracts.document_store import DocumentRecord

    store = InMemoryDocumentStore()
    original_delete = store.delete

    def flaky_delete(partition_key: str, row_key: str) -> None:
        if row_key == "malformed-boom":
            raise RuntimeError("delete-failed")
        return original_delete(partition_key, row_key)

    store.delete = flaky_delete  # type: ignore[method-assign]
    repo = ManagedWorkspaceRepository(store)
    storage = FakeObjectStorage()
    queue = DocumentStoreTaskQueue(store)
    ctx = ToolWiringContext(message_bus=queue)
    resolver = ManagedFileSourceResolver(repo)
    intake_svc = KnowledgeIntakeService(repo, resolver, ctx)
    managed = ManagedFileIntakeService(
        repo,
        storage,
        intake_svc,
        max_bytes=1024 * 1024,
        max_batch_files=20,
    )
    _seed_workspace(repo)
    recovery = KnowledgeIngestionRecoveryService(repo, intake_svc)

    processing = managed.accept_one(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="rec-proc-2",
        upload=ManagedFileUpload("p.pdf", "application/pdf", b"p"),
    )
    repo.put_operation(
        processing.operation.model_copy(update={"status": WorkspaceOperationStatus.PROCESSING})
    )
    partition = "lkw.managed_workspace:active_knowledge_ingestion"
    repo.document_store.put(
        DocumentRecord(
            partition_key=partition,
            row_key="malformed-boom",
            data={
                "operation_id": "malformed-boom",
                "tenant_id": TENANT,
                "workspace_id": WORKSPACE,
                "created_at": "not-a-timestamp",
            },
        )
    )
    result = recovery.recover_all()
    assert result.errors >= 1
    failed_p = repo.get_operation(
        tenant_id=TENANT, operation_id=processing.operation.operation_id
    )
    assert failed_p is not None
    assert failed_p.status is WorkspaceOperationStatus.FAILED


def test_managed_source_correlation_bind_reuse_and_conflict() -> None:
    repo, _, managed, _, _, _, _ = _build_intake()
    acceptance = managed.accept_one(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="src-bind",
        upload=ManagedFileUpload("a.pdf", "application/pdf", b"a"),
    )
    resolver = ManagedFileSourceResolver(repo)
    expected = acceptance.source.source_id

    # Bind from None
    mf = repo.get_managed_file(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        input_id=acceptance.knowledge_input.input_id,
    )
    assert mf is not None
    repo.put_managed_file(mf.model_copy(update={"source_id": None}))
    ki = acceptance.knowledge_input.model_copy(update={"source_id": None})
    resolved = resolver.resolve(knowledge_input=ki, suggested_source_id=expected)
    assert resolved.source_id == expected
    rebound = repo.get_managed_file(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        input_id=acceptance.knowledge_input.input_id,
    )
    assert rebound is not None
    assert rebound.source_id == expected

    # Matching ID reused
    again = resolver.resolve(
        knowledge_input=acceptance.knowledge_input,
        suggested_source_id=expected,
    )
    assert again.source_id == expected
    same = repo.get_managed_file(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        input_id=acceptance.knowledge_input.input_id,
    )
    assert same is not None
    assert same.source_id == expected

    # Conflicting managed source_id
    repo.put_managed_file(same.model_copy(update={"source_id": "wrong-source"}))
    before_sources = repo.list_sources(tenant_id=TENANT, workspace_id=WORKSPACE)
    with pytest.raises(KnowledgeInputResolutionError, match="managed_file_source_conflict"):
        resolver.resolve(
            knowledge_input=acceptance.knowledge_input.model_copy(update={"source_id": None}),
            suggested_source_id=expected,
        )
    conflicted = repo.get_managed_file(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        input_id=acceptance.knowledge_input.input_id,
    )
    assert conflicted is not None
    assert conflicted.source_id == "wrong-source"
    assert repo.list_sources(tenant_id=TENANT, workspace_id=WORKSPACE) == before_sources

    # KnowledgeInput mismatch
    with pytest.raises(KnowledgeInputResolutionError, match="managed_file_source_conflict"):
        resolver.resolve(
            knowledge_input=acceptance.knowledge_input.model_copy(
                update={"source_id": "other-source"}
            ),
            suggested_source_id=expected,
        )
    still = repo.get_managed_file(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        input_id=acceptance.knowledge_input.input_id,
    )
    assert still is not None
    assert still.source_id == "wrong-source"


def test_deterministic_operation_id_helper_matches_accept() -> None:
    from local_workspace_application.workspaces.knowledge_intake import (
        deterministic_knowledge_operation_id,
    )

    _, _, managed, _, _, _, _ = _build_intake()
    acceptance = managed.accept_one(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="op-id",
        upload=ManagedFileUpload("a.pdf", "application/pdf", b"a"),
    )
    assert acceptance.operation.operation_id == deterministic_knowledge_operation_id(
        input_id=acceptance.knowledge_input.input_id
    )
    assert acceptance.managed_file.operation_id == acceptance.operation.operation_id


def test_fingerprint_canonical_json_removes_nul_ambiguity() -> None:
    import hashlib
    import json
    import re

    body_hash = f"sha256:{hashlib.sha256(b"x").hexdigest()}"
    fp_a = managed_file_request_fingerprint(
        raw_file_name="a\0b",
        raw_content_type="c",
        size_bytes=1,
        body_hash=body_hash,
        request_state="complete",
    )
    fp_b = managed_file_request_fingerprint(
        raw_file_name="a",
        raw_content_type="b\0c",
        size_bytes=1,
        body_hash=body_hash,
        request_state="complete",
    )
    assert fp_a != fp_b
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", fp_a)
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", fp_b)
    again = managed_file_request_fingerprint(
        raw_file_name="a\0b",
        raw_content_type="c",
        size_bytes=1,
        body_hash=body_hash,
        request_state="complete",
    )
    assert again == fp_a
    payload = {
        "body_hash": body_hash,
        "raw_content_type": "c",
        "raw_file_name": "a\0b",
        "request_state": "complete",
        "size_bytes": 1,
        "version": 2,
    }
    expected = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    )
    assert fp_a == expected
    with pytest.raises(ValueError, match="request_state_invalid"):
        managed_file_request_fingerprint(
            raw_file_name="a.pdf",
            raw_content_type="application/pdf",
            size_bytes=1,
            body_hash=body_hash,
            request_state="unknown",
        )


def test_candidate_invariants_accept_and_reject() -> None:
    import hashlib

    empty_hash = f"sha256:{hashlib.sha256(b"").hexdigest()}"
    body = b"%PDF"
    body_hash = f"sha256:{hashlib.sha256(body).hexdigest()}"
    ok_fp = managed_file_request_fingerprint(
        raw_file_name="a.pdf",
        raw_content_type="application/pdf",
        size_bytes=len(body),
        body_hash=body_hash,
    )
    ManagedFileBatchCandidate(
        raw_file_name="a.pdf",
        raw_content_type="application/pdf",
        body=body,
        size_bytes=len(body),
        body_hash=body_hash,
        request_fingerprint=ok_fp,
    )
    empty_fp = managed_file_request_fingerprint(
        raw_file_name="a.pdf",
        raw_content_type="application/pdf",
        size_bytes=0,
        body_hash=empty_hash,
    )
    ManagedFileBatchCandidate(
        raw_file_name="a.pdf",
        raw_content_type="application/pdf",
        body=b"",
        size_bytes=0,
        body_hash=empty_hash,
        request_fingerprint=empty_fp,
    )
    oversized = b"x" * 10
    oversized_hash = f"sha256:{hashlib.sha256(oversized).hexdigest()}"
    ManagedFileBatchCandidate(
        raw_file_name="a.pdf",
        raw_content_type="application/pdf",
        body=None,
        size_bytes=len(oversized),
        body_hash=oversized_hash,
        request_fingerprint=managed_file_request_fingerprint(
            raw_file_name="a.pdf",
            raw_content_type="application/pdf",
            size_bytes=len(oversized),
            body_hash=oversized_hash,
            request_state="complete",
        ),
        preflight_error_code="managed_file_too_large",
    )
    ManagedFileBatchCandidate(
        raw_file_name="a.pdf",
        raw_content_type="application/pdf",
        body=None,
        size_bytes=3,
        body_hash=f"sha256:{hashlib.sha256(b"abc").hexdigest()}",
        request_fingerprint=managed_file_request_fingerprint(
            raw_file_name="a.pdf",
            raw_content_type="application/pdf",
            size_bytes=3,
            body_hash=f"sha256:{hashlib.sha256(b"abc").hexdigest()}",
            request_state="read_failed",
        ),
        preflight_error_code="managed_file_upload_read_failed",
    )

    with pytest.raises(ValueError, match="candidate_state_invalid"):
        ManagedFileBatchCandidate(
            raw_file_name="a.pdf",
            raw_content_type="application/pdf",
            body=body,
            size_bytes=len(body),
            body_hash=body_hash,
            request_fingerprint=ok_fp,
            preflight_error_code="managed_file_empty",
        )
    with pytest.raises(ValueError, match="candidate_state_invalid"):
        ManagedFileBatchCandidate(
            raw_file_name="a.pdf",
            raw_content_type="application/pdf",
            body=None,
            size_bytes=0,
            body_hash=empty_hash,
            request_fingerprint=empty_fp,
            preflight_error_code=None,
        )
    with pytest.raises(ValueError, match="preflight_error_code_invalid"):
        ManagedFileBatchCandidate(
            raw_file_name="a.pdf",
            raw_content_type="application/pdf",
            body=None,
            size_bytes=0,
            body_hash=empty_hash,
            request_fingerprint=empty_fp,
            preflight_error_code="not_a_public_code",
        )
    with pytest.raises(ValueError, match="body_size_mismatch"):
        ManagedFileBatchCandidate(
            raw_file_name="a.pdf",
            raw_content_type="application/pdf",
            body=body,
            size_bytes=99,
            body_hash=body_hash,
            request_fingerprint=ok_fp,
        )
    with pytest.raises(ValueError, match="body_hash_mismatch"):
        ManagedFileBatchCandidate(
            raw_file_name="a.pdf",
            raw_content_type="application/pdf",
            body=body,
            size_bytes=len(body),
            body_hash=empty_hash,
            request_fingerprint=ok_fp,
        )
    with pytest.raises(ValueError, match="request_fingerprint_invalid"):
        ManagedFileBatchCandidate(
            raw_file_name="a.pdf",
            raw_content_type="application/pdf",
            body=body,
            size_bytes=len(body),
            body_hash=body_hash,
            request_fingerprint="not-a-digest",
        )
    with pytest.raises(ValueError, match="size_bytes_invalid"):
        ManagedFileBatchCandidate(
            raw_file_name="a.pdf",
            raw_content_type="application/pdf",
            body=None,
            size_bytes=-1,
            body_hash=empty_hash,
            request_fingerprint=empty_fp,
            preflight_error_code="managed_file_empty",
        )


def test_invalid_body_type_fingerprint_differs_from_empty_bytes() -> None:
    _, _, managed, _, _, _, _ = _build_intake()
    invalid = ManagedFileUpload("a.pdf", "application/pdf", bytearray(b""))  # type: ignore[arg-type]
    empty = ManagedFileUpload("a.pdf", "application/pdf", b"")
    first = managed.accept_many(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="inv-body",
        uploads=[invalid],
    )
    second = managed.accept_many(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="inv-body",
        uploads=[invalid],
    )
    assert first.items[0].error_code == "managed_file_body_required"
    assert second.batch_id == first.batch_id
    with pytest.raises(IntakeBatchIdempotencyConflict):
        managed.accept_many(
            tenant_id=TENANT,
            workspace_id=WORKSPACE,
            idempotency_key="inv-body",
            uploads=[empty],
        )
    empty_batch = managed.accept_many(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="empty-body",
        uploads=[empty],
    )
    assert empty_batch.items[0].error_code == "managed_file_empty"
    assert empty_batch.items[0].request_fingerprint != first.items[0].request_fingerprint


def test_intake_service_validates_managed_source_on_existing_source() -> None:
    repo, _, managed, queue, _, _, _ = _build_intake()
    intake = managed._knowledge_intake  # noqa: SLF001
    acceptance = managed.accept_one(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="src-svc",
        upload=ManagedFileUpload("a.pdf", "application/pdf", b"a"),
    )
    expected = acceptance.source.source_id
    ops_before = len(repo.list_operations(tenant_id=TENANT))
    sources_before = repo.list_sources(tenant_id=TENANT, workspace_id=WORKSPACE)
    task_before = acceptance.operation.queue_task_id

    mf = repo.get_managed_file(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        input_id=acceptance.knowledge_input.input_id,
    )
    assert mf is not None
    repo.put_managed_file(mf.model_copy(update={"source_id": "wrong-source"}))

    with pytest.raises(KnowledgeInputResolutionError, match="managed_file_source_conflict"):
        intake.reconcile_workspace(tenant_id=TENANT, workspace_id=WORKSPACE)

    conflicted = repo.get_managed_file(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        input_id=acceptance.knowledge_input.input_id,
    )
    assert conflicted is not None
    assert conflicted.source_id == "wrong-source"
    ki = repo.get_knowledge_input(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        input_id=acceptance.knowledge_input.input_id,
    )
    assert ki is not None
    assert ki.source_id == expected
    assert repo.list_sources(tenant_id=TENANT, workspace_id=WORKSPACE) == sources_before
    assert len(repo.list_operations(tenant_id=TENANT)) == ops_before
    loaded_op = repo.get_operation(
        tenant_id=TENANT, operation_id=acceptance.operation.operation_id
    )
    assert loaded_op is not None
    assert loaded_op.queue_task_id == task_before
    _ = queue


def test_intake_service_knowledge_input_source_conflict() -> None:
    repo, _, managed, _, _, _, _ = _build_intake()
    intake = managed._knowledge_intake  # noqa: SLF001
    first = managed.accept_one(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="ki-conflict-a",
        upload=ManagedFileUpload("a.pdf", "application/pdf", b"a"),
    )
    second = managed.accept_one(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="ki-conflict-b",
        upload=ManagedFileUpload("b.pdf", "application/pdf", b"b"),
    )
    mf_before = repo.get_managed_file(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        input_id=first.knowledge_input.input_id,
    )
    assert mf_before is not None
    repo.put_knowledge_input(
        first.knowledge_input.model_copy(update={"source_id": second.source.source_id})
    )
    with pytest.raises(KnowledgeInputResolutionError, match="managed_file_source_conflict"):
        intake.reconcile_workspace(tenant_id=TENANT, workspace_id=WORKSPACE)
    ki = repo.get_knowledge_input(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        input_id=first.knowledge_input.input_id,
    )
    assert ki is not None
    assert ki.source_id == second.source.source_id
    mf = repo.get_managed_file(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        input_id=first.knowledge_input.input_id,
    )
    assert mf is not None
    assert mf.source_id == mf_before.source_id
    assert mf.source_id != second.source.source_id


def test_intake_service_valid_existing_source_reuse() -> None:
    repo, _, managed, _, _, _, _ = _build_intake()
    first = managed.accept_one(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="reuse-ok",
        upload=ManagedFileUpload("a.pdf", "application/pdf", b"a"),
    )
    second = managed.accept_one(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="reuse-ok",
        upload=ManagedFileUpload("a.pdf", "application/pdf", b"a"),
    )
    assert second.source.source_id == first.source.source_id
    assert second.operation.operation_id == first.operation.operation_id
    assert second.operation.queue_task_id == first.operation.queue_task_id
    assert (
        len(repo.list_sources(tenant_id=TENANT, workspace_id=WORKSPACE))
        == len({first.source.source_id})
    )


def test_materialization_failure_marks_managed_object_error(tmp_path: Path) -> None:
    repo, storage, managed, queue, _, _, indexing = _build_intake()
    # Staging root outside itself via symlink-style escape is hard; force mkdir/write fail
    # by pointing materializer at a file path instead of a directory.
    staging_as_file = tmp_path / "not-a-dir"
    staging_as_file.write_text("x", encoding="utf-8")
    materializer = ManagedObjectMaterializer(storage, staging_as_file)
    processor = ManagedFileKnowledgeIngestionProcessor(repo, materializer, indexing)  # type: ignore[arg-type]
    ingestion = KnowledgeIngestionService(repo, processor)
    registry = TaskExecutionRegistry()
    register_knowledge_ingestion_worker_handler(registry, ingestion)
    worker = DocumentStoreTaskWorker(queue, registry)

    acceptance = managed.accept_one(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="mat-fail",
        upload=ManagedFileUpload("contract.pdf", "application/pdf", b"%PDF"),
    )
    assert worker.drain_once() == 1
    op = repo.get_operation(tenant_id=TENANT, operation_id=acceptance.operation.operation_id)
    assert op is not None
    assert op.status is WorkspaceOperationStatus.FAILED
    assert op.error_code == "managed_object_materialization_failed"
    assert op.error == "managed_object_materialization_failed"
    source = repo.get_source(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        source_id=acceptance.source.source_id,
    )
    assert source is not None
    assert source.status is WorkspaceSourceStatus.ERROR
    mf = repo.get_managed_file(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        input_id=acceptance.knowledge_input.input_id,
    )
    assert mf is not None
    assert mf.status is ManagedFileObjectStatus.ERROR
    assert mf.error_code == "managed_object_materialization_failed"
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
    assert not repo.list_active_ingestion_locators()
    durable = str(op.error) + str(op.error_code) + str(mf.error_code)
    assert str(staging_as_file) not in durable
    assert "RuntimeError" not in durable


def test_indexing_failure_marks_managed_object_error(tmp_path: Path) -> None:
    class BoomIndexing:
        async def index_one(self, **kwargs: object) -> WorkspaceDocumentIndexingResult:
            _ = kwargs
            raise WorkspaceDocumentIndexingError("parser_boom_detail")

    repo, storage, managed, queue, _, _, _ = _build_intake()
    materializer = ManagedObjectMaterializer(storage, tmp_path / "staging")
    processor = ManagedFileKnowledgeIngestionProcessor(
        repo, materializer, BoomIndexing()  # type: ignore[arg-type]
    )
    ingestion = KnowledgeIngestionService(repo, processor)
    registry = TaskExecutionRegistry()
    register_knowledge_ingestion_worker_handler(registry, ingestion)
    worker = DocumentStoreTaskWorker(queue, registry)

    acceptance = managed.accept_one(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        idempotency_key="idx-fail",
        upload=ManagedFileUpload("contract.pdf", "application/pdf", b"%PDF"),
    )
    assert worker.drain_once() == 1
    op = repo.get_operation(tenant_id=TENANT, operation_id=acceptance.operation.operation_id)
    assert op is not None
    assert op.status is WorkspaceOperationStatus.FAILED
    assert op.error_code == "managed_file_indexing_failed"
    assert op.error == "managed_file_indexing_failed"
    source = repo.get_source(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        source_id=acceptance.source.source_id,
    )
    assert source is not None
    assert source.status is WorkspaceSourceStatus.ERROR
    mf = repo.get_managed_file(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        input_id=acceptance.knowledge_input.input_id,
    )
    assert mf is not None
    assert mf.status is ManagedFileObjectStatus.ERROR
    assert mf.error_code == "managed_file_indexing_failed"
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
    assert not repo.list_active_ingestion_locators()
    durable = str(op.error) + str(op.error_code) + str(mf.error_code)
    assert "parser_boom_detail" not in durable
    assert "WorkspaceDocumentIndexingError" not in durable

# © Artur Czarnecki. All rights reserved.

"""WEB_URL resolver, processor and queue lifecycle tests."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.queueing.contracts.task_queue import TaskHandle, TaskStatus
from intergrax.queueing.providers.document_store import DocumentStoreTaskQueue
from intergrax.queueing.providers.document_store.colocated_worker import DocumentStoreTaskWorker
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.tools.registry.wiring import ToolWiringContext
from intergrax.websearch.capture.contracts import (
    CapturedWebContent,
    WebContentCaptureRequest,
)
from intergrax.websearch.capture.url_policy import WebUrlAccessPolicy
from local_workspace_application.workspaces.document_indexing import (
    WorkspaceDocumentIndexingError,
    WorkspaceDocumentIndexingResult,
)
from local_workspace_application.workspaces.ingestion_recovery import (
    KnowledgeIngestionRecoveryService,
)
from local_workspace_application.workspaces.knowledge_ingestion import (
    KnowledgeIngestionProcessorError,
    KnowledgeIngestionService,
    register_knowledge_ingestion_worker_handler,
)
from local_workspace_application.workspaces.knowledge_intake import (
    KnowledgeInputResolutionError,
    KnowledgeIntakeService,
)
from local_workspace_application.workspaces.models import (
    KnowledgeInput,
    KnowledgeInputKind,
    KnowledgeInputStatus,
    Workspace,
    WorkspaceOperationStatus,
    WorkspaceSource,
    WorkspaceSourceStatus,
    WorkspaceSourceType,
    WorkspaceStatus,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.web_url_ingestion import (
    WebUrlIntakeService,
    WebUrlKnowledgeIngestionProcessor,
    WebUrlSourceResolver,
    WebUrlTextMaterializer,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

TENANT = "tenant-a"
WORKSPACE = "workspace-a"
FIXTURE_TEXT = "The Oriole warehouse verification code is BLUE-7319."


async def _resolve_public(_host: str) -> tuple[str, ...]:
    return ("93.184.216.34",)


def _now() -> datetime:
    return datetime.now(UTC)


def _seed_workspace(repo: ManagedWorkspaceRepository) -> None:
    repo.put_workspace(
        Workspace(
            workspace_id=WORKSPACE,
            tenant_id=TENANT,
            name="Demo",
            status=WorkspaceStatus.ACTIVE,
            created_at=_now(),
            updated_at=_now(),
        )
    )


class FakeWebContentCapture:
    def __init__(self, *, text: str = FIXTURE_TEXT, error_code: str | None = None) -> None:
        self.text = text
        self.error_code = error_code
        self.calls: list[WebContentCaptureRequest] = []

    async def capture(self, request: WebContentCaptureRequest) -> CapturedWebContent:
        self.calls.append(request)
        if self.error_code:
            from intergrax.websearch.capture.contracts import (
                WebContentCaptureError,
                WebContentCaptureErrorCode,
            )

            raise WebContentCaptureError(WebContentCaptureErrorCode(self.error_code))
        now = _now()
        return CapturedWebContent(
            safe_display_url="https://example.com/docs",
            requested_url_fingerprint="sha256:" + "f" * 64,
            final_url_fingerprint="sha256:" + "e" * 64,
            final_host_changed=False,
            title="Docs",
            text=self.text,
            content_type="text/html",
            content_hash="sha256:" + "d" * 64,
            status_code=200,
            redirect_count=0,
            content_bytes=len(self.text.encode()),
            text_chars=len(self.text),
            capture_mode="http",
            extraction_method="basic",
            fetched_at=now,
        )


class SpyIndexingService:
    def __init__(self, *, fail: bool = False, unchanged: bool = False) -> None:
        self.calls: list[dict[str, object]] = []
        self.fail = fail
        self.unchanged = unchanged

    async def index_one(self, **kwargs: object) -> WorkspaceDocumentIndexingResult:
        self.calls.append(kwargs)
        if self.fail:
            raise WorkspaceDocumentIndexingError("index_failed")
        return WorkspaceDocumentIndexingResult(
            indexed=not self.unchanged,
            unchanged=self.unchanged,
            document_id="doc-1",
            documents_indexed=0 if self.unchanged else 1,
            num_chunks=1,
            reason="ingest_complete",
        )


def _build_stack(
    tmp_path: Path,
    *,
    indexing: SpyIndexingService | None = None,
    capture: FakeWebContentCapture | None = None,
) -> tuple[
    ManagedWorkspaceRepository,
    WebUrlIntakeService,
    DocumentStoreTaskQueue,
    DocumentStoreTaskWorker,
    KnowledgeIngestionService,
    SpyIndexingService,
    FakeWebContentCapture,
]:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    queue = DocumentStoreTaskQueue(store)
    policy = WebUrlAccessPolicy(dns_resolver=_resolve_public)
    resolver = WebUrlSourceResolver(repo)
    intake_svc = KnowledgeIntakeService(repo, resolver, ToolWiringContext(message_bus=queue))
    web_intake = WebUrlIntakeService(repo, intake_svc, policy)
    indexing = indexing or SpyIndexingService()
    capture = capture or FakeWebContentCapture()
    materializer = WebUrlTextMaterializer(tmp_path / "staging")
    processor = WebUrlKnowledgeIngestionProcessor(
        repo,
        capture,
        indexing,  # type: ignore[arg-type]
        materializer,
    )
    ingestion = KnowledgeIngestionService(repo, processor)
    registry = TaskExecutionRegistry()
    register_knowledge_ingestion_worker_handler(registry, ingestion)
    worker = DocumentStoreTaskWorker(queue, registry)
    _seed_workspace(repo)
    return repo, web_intake, queue, worker, ingestion, indexing, capture


def _accept(web_intake: WebUrlIntakeService, **kwargs: object):
    return asyncio.run(web_intake.accept(**kwargs))


def test_resolver_creates_web_resource_source(tmp_path: Path) -> None:
    repo, web_intake, _, _, _, _, _ = _build_stack(tmp_path)
    accepted = _accept(
        web_intake,
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        raw_url="https://example.com/docs",
        idempotency_key="resolver-1",
    )
    source = repo.get_source(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        source_id=accepted.source_id,
    )
    assert source is not None
    assert source.source_type is WorkspaceSourceType.WEB_RESOURCE
    assert source.path == ""
    assert source.recursive is False


def test_processor_indexes_and_updates_locator(tmp_path: Path) -> None:
    repo, web_intake, _, worker, _, indexing, capture = _build_stack(tmp_path)
    accepted = _accept(
        web_intake,
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        raw_url="https://example.com/docs?secret=1",
        idempotency_key="proc-1",
    )
    assert worker.drain_once() == 1
    assert len(indexing.calls) == 1
    call = indexing.calls[0]
    assert "secret=1" not in str(call)
    assert call["logical_source_path"] == f"web/{accepted.source_id}/content.txt"
    assert call["safe_file_name"] == "https://example.com/docs"
    assert not any(tmp_path.rglob("web-content.txt"))
    op = repo.get_operation(tenant_id=TENANT, operation_id=accepted.operation_id)
    assert op is not None
    assert op.status is WorkspaceOperationStatus.COMPLETED
    assert op.files_discovered == 1
    assert op.documents_indexed == 1
    source = repo.get_source(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        source_id=accepted.source_id,
    )
    assert source is not None
    assert source.status is WorkspaceSourceStatus.READY
    fingerprint = repo.list_knowledge_inputs(tenant_id=TENANT, workspace_id=WORKSPACE)[
        0
    ].submission_metadata["source_fingerprint"]
    locator = repo.get_web_url_locator(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        requested_url_fingerprint=fingerprint,
    )
    assert locator is not None
    assert locator.final_url_fingerprint is not None
    assert "?secret=1" in locator.canonical_private_url
    assert len(capture.calls) == 1


def test_processor_capture_error_maps_to_stable_code(tmp_path: Path) -> None:
    repo, web_intake, _, worker, _, _, _ = _build_stack(
        tmp_path,
        capture=FakeWebContentCapture(error_code="web_url_timeout"),
    )
    accepted = _accept(
        web_intake,
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        raw_url="https://example.com/docs",
        idempotency_key="cap-fail",
    )
    assert worker.drain_once() == 1
    op = repo.get_operation(tenant_id=TENANT, operation_id=accepted.operation_id)
    assert op is not None
    assert op.error_code == "web_url_timeout"
    assert "https://" not in (op.error or "")


def test_recovery_redispatches_without_duplicate_operation(tmp_path: Path) -> None:
    repo, web_intake, queue, worker, _, _, _ = _build_stack(tmp_path)
    accepted = _accept(
        web_intake,
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        raw_url="https://example.com/docs",
        idempotency_key="recovery-1",
    )
    op_before = repo.get_operation(tenant_id=TENANT, operation_id=accepted.operation_id)
    assert op_before is not None
    task_id = op_before.queue_task_id
    recovery = KnowledgeIngestionRecoveryService(
        repo,
        KnowledgeIntakeService(
            repo,
            WebUrlSourceResolver(repo),
            ToolWiringContext(message_bus=queue),
        ),
    )
    recovery.recover_all()
    op_after = repo.get_operation(tenant_id=TENANT, operation_id=accepted.operation_id)
    assert op_after is not None
    assert op_after.operation_id == accepted.operation_id
    assert worker.drain_once() == 1
    op_after = repo.get_operation(tenant_id=TENANT, operation_id=accepted.operation_id)
    assert op_after is not None
    assert op_after.status is WorkspaceOperationStatus.COMPLETED


def test_worker_payload_has_identities_only(tmp_path: Path) -> None:
    repo, web_intake, _, _, _, _, _ = _build_stack(tmp_path)
    accepted = _accept(
        web_intake,
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        raw_url="https://example.com/docs?x=1",
        idempotency_key="payload-1",
    )
    op = repo.get_operation(tenant_id=TENANT, operation_id=accepted.operation_id)
    assert op is not None
    from local_workspace_application.workspaces.knowledge_ingestion import (
        decode_knowledge_ingestion_job,
        encode_knowledge_ingestion_job,
        KnowledgeIngestionJob,
    )

    job = KnowledgeIngestionJob(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        input_id=accepted.input_id,
        source_id=accepted.source_id,
        operation_id=accepted.operation_id,
    )
    payload = encode_knowledge_ingestion_job(job).decode("utf-8")
    assert "https://" not in payload
    assert "x=1" not in payload
    decoded = decode_knowledge_ingestion_job(encode_knowledge_ingestion_job(job))
    assert decoded.input_id == accepted.input_id


def test_resolver_rejects_wrong_kind() -> None:
    repo = ManagedWorkspaceRepository(InMemoryDocumentStore())
    resolver = WebUrlSourceResolver(repo)
    knowledge_input = KnowledgeInput(
        input_id="ki:1",
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        input_kind=KnowledgeInputKind.MANAGED_FILE,
        idempotency_key="k",
        operation_id="op:1",
        status=KnowledgeInputStatus.ACCEPTED,
        submission_metadata={"source_fingerprint": "sha256:" + "a" * 64},
        created_at=_now(),
        updated_at=_now(),
    )
    with pytest.raises(KnowledgeInputResolutionError):
        resolver.resolve(knowledge_input=knowledge_input, suggested_source_id="src:1")

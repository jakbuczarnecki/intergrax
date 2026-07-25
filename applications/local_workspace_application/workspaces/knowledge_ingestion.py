# © Artur Czarnecki. All rights reserved.

"""Knowledge Ingestion job, processor port, operation service and worker handler."""

from __future__ import annotations

import asyncio
import base64
import json
from datetime import UTC, datetime
from typing import Callable, Literal, Optional, Protocol

from pydantic import BaseModel, Field

from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.tools.execution_models import ToolExecutionResult
from intergrax.tools.providers.message_bus.contracts import MessageBusEnqueueInput
from local_workspace_application.workspaces.models import (
    KnowledgeInput,
    WorkspaceOperation,
    WorkspaceOperationStatus,
    WorkspaceOperationType,
    WorkspaceSource,
    WorkspaceSourceStatus,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository

LKW_KNOWLEDGE_INGESTION_TASK_NAME = "lkw.knowledge_ingestion.v1"
LKW_KNOWLEDGE_INGESTION_SCHEMA = "lkw.knowledge_ingestion_job.v1"

MainLoopProvider = Callable[[], asyncio.AbstractEventLoop | None]


class KnowledgeIngestionJob(BaseModel):
    """Queue payload — identities only; worker reloads durable state."""

    schema_version: Literal["lkw.knowledge_ingestion_job.v1"] = LKW_KNOWLEDGE_INGESTION_SCHEMA
    tenant_id: str = Field(..., min_length=1)
    workspace_id: str = Field(..., min_length=1)
    input_id: str = Field(..., min_length=1)
    source_id: str = Field(..., min_length=1)
    operation_id: str = Field(..., min_length=1)


class KnowledgeIngestionResult(BaseModel):
    files_processed: int = Field(default=0, ge=0)
    files_failed: int = Field(default=0, ge=0)
    documents_indexed: int = Field(default=0, ge=0)
    documents_unchanged: int = Field(default=0, ge=0)


class KnowledgeIngestionProcessor(Protocol):
    async def process(
        self,
        *,
        knowledge_input: KnowledgeInput,
        source: WorkspaceSource,
        operation: WorkspaceOperation,
    ) -> KnowledgeIngestionResult: ...


class KnowledgeIngestionWorkerOutput(BaseModel):
    operation_id: str
    status: str
    schema_version: str = "lkw.knowledge_ingestion_worker.v1"
    metadata: dict[str, object] = Field(default_factory=dict)


def encode_knowledge_ingestion_job(job: KnowledgeIngestionJob) -> bytes:
    return json.dumps(
        job.model_dump(mode="json"),
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def decode_knowledge_ingestion_job(payload: bytes) -> KnowledgeIngestionJob:
    raw = json.loads(payload.decode("utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("knowledge_ingestion_job must be a JSON object")
    return KnowledgeIngestionJob.model_validate(raw)


def knowledge_ingestion_payload_base64(job: KnowledgeIngestionJob) -> str:
    return base64.b64encode(encode_knowledge_ingestion_job(job)).decode("ascii")


def knowledge_ingestion_idempotency_key(job: KnowledgeIngestionJob) -> str:
    return f"{LKW_KNOWLEDGE_INGESTION_TASK_NAME}:{job.operation_id}"


def build_knowledge_ingestion_enqueue_input(job: KnowledgeIngestionJob) -> MessageBusEnqueueInput:
    return MessageBusEnqueueInput(
        tenant_id=job.tenant_id,
        run_id=job.operation_id,
        task_name=LKW_KNOWLEDGE_INGESTION_TASK_NAME,
        payload_base64=knowledge_ingestion_payload_base64(job),
        idempotency_key=knowledge_ingestion_idempotency_key(job),
    )


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _safe_error_message(exc: BaseException) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    first_line = message.splitlines()[0].strip()
    return first_line[:500]


class KnowledgeIngestionProcessorError(RuntimeError):
    """Stable processor failure with a durable domain error code."""

    def __init__(self, error_code: str) -> None:
        code = (error_code or "").strip()
        if not code:
            raise ValueError("error_code_required")
        self.error_code = code
        super().__init__(code)


class KnowledgeIngestionService:
    """Executes a durable Knowledge Ingestion operation via a processor port."""

    def __init__(
        self,
        repository: ManagedWorkspaceRepository,
        processor: KnowledgeIngestionProcessor,
    ) -> None:
        self._repository = repository
        self._processor = processor

    async def run_operation(
        self,
        *,
        tenant_id: str,
        operation_id: str,
    ) -> WorkspaceOperation:
        operation = self._repository.get_operation(tenant_id=tenant_id, operation_id=operation_id)
        if operation is None:
            raise LookupError("operation_not_found")

        if operation.operation_type is not WorkspaceOperationType.KNOWLEDGE_INGESTION:
            return self._fail(operation, error_code="invalid_operation_type")

        if operation.status in {
            WorkspaceOperationStatus.COMPLETED,
            WorkspaceOperationStatus.FAILED,
            WorkspaceOperationStatus.PROCESSING,
        }:
            return operation

        if operation.status not in {
            WorkspaceOperationStatus.ACCEPTED,
            WorkspaceOperationStatus.QUEUED,
        }:
            return self._fail(
                operation,
                error_code=f"unexpected_operation_status:{operation.status.value}",
            )

        if not operation.input_id:
            return self._fail(operation, error_code="input_id_missing")

        knowledge_input = self._repository.get_knowledge_input(
            tenant_id=tenant_id,
            workspace_id=operation.workspace_id,
            input_id=operation.input_id,
        )
        source = self._repository.get_source(
            tenant_id=tenant_id,
            workspace_id=operation.workspace_id,
            source_id=operation.source_id,
        )
        if knowledge_input is None or source is None:
            return self._fail(operation, error_code="durable_state_missing", source=source)

        if (
            knowledge_input.operation_id != operation.operation_id
            or knowledge_input.source_id != operation.source_id
            or knowledge_input.tenant_id != operation.tenant_id
            or knowledge_input.workspace_id != operation.workspace_id
            or source.tenant_id != operation.tenant_id
            or source.workspace_id != operation.workspace_id
        ):
            return self._fail(operation, error_code="durable_state_inconsistent", source=source)

        started = operation.started_at or _utc_now()
        operation = operation.model_copy(
            update={
                "status": WorkspaceOperationStatus.PROCESSING,
                "started_at": started,
                "error": None,
                "error_code": None,
            }
        )
        self._repository.put_operation(operation)
        self._ensure_active_locator(operation)
        self._repository.put_source(
            source.model_copy(update={"status": WorkspaceSourceStatus.PROCESSING})
        )

        try:
            result = await self._processor.process(
                knowledge_input=knowledge_input,
                source=source,
                operation=operation,
            )
        except KnowledgeIngestionProcessorError as exc:
            return self._fail(
                operation,
                error_code=exc.error_code,
                error=exc.error_code,
                source=source,
            )
        except Exception as exc:  # noqa: BLE001 - persist fail-closed product state
            return self._fail(
                operation,
                error_code="processor_failed",
                error=_safe_error_message(exc),
                source=source,
            )

        if not isinstance(result, KnowledgeIngestionResult):
            return self._fail(
                operation,
                error_code="invalid_processor_result",
                source=source,
            )

        completed = _utc_now()
        operation = operation.model_copy(
            update={
                "status": WorkspaceOperationStatus.COMPLETED,
                "files_processed": result.files_processed,
                "files_failed": result.files_failed,
                "documents_indexed": result.documents_indexed,
                "documents_unchanged": result.documents_unchanged,
                "completed_at": completed,
                "error": None,
                "error_code": None,
            }
        )
        self._repository.put_operation(operation)
        self._repository.put_source(
            source.model_copy(
                update={
                    "status": WorkspaceSourceStatus.READY,
                    "last_sync_at": completed,
                }
            )
        )
        self._clear_active_locator(operation.operation_id)
        return operation

    def _ensure_active_locator(self, operation: WorkspaceOperation) -> None:
        from local_workspace_application.workspaces.models import ActiveKnowledgeIngestionLocator

        if operation.status not in {
            WorkspaceOperationStatus.ACCEPTED,
            WorkspaceOperationStatus.QUEUED,
            WorkspaceOperationStatus.PROCESSING,
        }:
            return
        self._repository.put_active_ingestion_locator(
            ActiveKnowledgeIngestionLocator(
                operation_id=operation.operation_id,
                tenant_id=operation.tenant_id,
                workspace_id=operation.workspace_id,
                created_at=operation.created_at or _utc_now(),
            )
        )

    def _clear_active_locator(self, operation_id: str) -> None:
        self._repository.delete_active_ingestion_locator(operation_id)

    def _fail(
        self,
        operation: WorkspaceOperation,
        *,
        error_code: str,
        error: str | None = None,
        source: WorkspaceSource | None = None,
    ) -> WorkspaceOperation:
        failed = operation.model_copy(
            update={
                "status": WorkspaceOperationStatus.FAILED,
                "error_code": error_code,
                "error": error or error_code,
                "completed_at": _utc_now(),
            }
        )
        self._repository.put_operation(failed)
        if source is not None:
            self._repository.put_source(
                source.model_copy(update={"status": WorkspaceSourceStatus.ERROR})
            )
        self._clear_active_locator(failed.operation_id)
        return failed


def make_knowledge_ingestion_worker_handler(
    ingestion_service: KnowledgeIngestionService,
    *,
    main_loop_provider: MainLoopProvider | None = None,
):
    def handler(
        *,
        tenant_id: str,
        run_id: str,
        payload: bytes,
        idempotency_key: Optional[str] = None,
    ) -> ToolExecutionResult[KnowledgeIngestionWorkerOutput]:
        _ = idempotency_key
        try:
            job = decode_knowledge_ingestion_job(payload)
            if job.tenant_id != tenant_id:
                return ToolExecutionResult.fail(
                    "knowledge_ingestion_tenant_mismatch",
                    "knowledge_ingestion_tenant_mismatch",
                )
            if run_id != job.operation_id:
                return ToolExecutionResult.fail(
                    "knowledge_ingestion_run_id_mismatch",
                    "knowledge_ingestion_run_id_mismatch",
                )
            coro = ingestion_service.run_operation(
                tenant_id=job.tenant_id,
                operation_id=job.operation_id,
            )
            main_loop = main_loop_provider() if main_loop_provider is not None else None
            if main_loop is not None and main_loop.is_running():
                operation = asyncio.run_coroutine_threadsafe(coro, main_loop).result(timeout=600)
            else:
                operation = asyncio.run(coro)
            if not isinstance(operation, WorkspaceOperation):
                return ToolExecutionResult.fail(
                    "knowledge_ingestion_invalid_result",
                    "knowledge_ingestion_invalid_result",
                )
            # A durably recorded domain failure is a successfully handled queue task.
            return ToolExecutionResult.ok(
                KnowledgeIngestionWorkerOutput(
                    operation_id=operation.operation_id,
                    status=operation.status.value,
                    metadata={
                        "workspace_id": operation.workspace_id,
                        "source_id": operation.source_id,
                        "input_id": operation.input_id,
                        "files_processed": operation.files_processed,
                        "files_failed": operation.files_failed,
                        "documents_indexed": operation.documents_indexed,
                        "documents_unchanged": operation.documents_unchanged,
                        "error_code": operation.error_code,
                    },
                )
            )
        except Exception as exc:  # noqa: BLE001 - worker plane normalizes failures
            return ToolExecutionResult.fail(type(exc).__name__, str(exc))

    return handler


def register_knowledge_ingestion_worker_handler(
    registry: TaskExecutionRegistry,
    ingestion_service: KnowledgeIngestionService,
    *,
    logical_task_name: str = LKW_KNOWLEDGE_INGESTION_TASK_NAME,
    main_loop_provider: MainLoopProvider | None = None,
) -> None:
    registry.register(
        logical_task_name,
        make_knowledge_ingestion_worker_handler(
            ingestion_service,
            main_loop_provider=main_loop_provider,
        ),
    )

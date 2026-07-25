# © Artur Czarnecki. All rights reserved.

"""Channel-neutral Knowledge Intake acceptance, Source resolution and dispatch."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Mapping, Protocol

from intergrax.tools.providers.message_bus.service import message_bus_enqueue
from intergrax.tools.registry.wiring import ToolWiringContext
from local_workspace_application.workspaces.knowledge_ingestion import (
    KnowledgeIngestionJob,
    build_knowledge_ingestion_enqueue_input,
)
from local_workspace_application.workspaces.models import (
    ActiveKnowledgeIngestionLocator,
    KnowledgeInput,
    KnowledgeInputKind,
    KnowledgeInputStatus,
    WorkspaceOperation,
    WorkspaceOperationStatus,
    WorkspaceOperationType,
    WorkspaceSource,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository


class KnowledgeInputIdempotencyConflict(Exception):
    """Same idempotency key reused with conflicting kind or metadata."""


class KnowledgeInputResolutionError(Exception):
    """Source resolver returned an invalid or inconsistent Source."""


class KnowledgeIntakeDispatchError(Exception):
    """MessageBus enqueue failed after durable operation creation."""


class KnowledgeIntakeStateConflict(Exception):
    """Existing durable state conflicts with deterministic Knowledge Intake identity."""


class KnowledgeInputSourceResolver(Protocol):
    def resolve(
        self,
        *,
        knowledge_input: KnowledgeInput,
        suggested_source_id: str,
    ) -> WorkspaceSource: ...


@dataclass(frozen=True)
class KnowledgeIntakeAcceptance:
    knowledge_input: KnowledgeInput
    source: WorkspaceSource
    operation: WorkspaceOperation


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


def deterministic_knowledge_input_id(
    *,
    tenant_id: str,
    workspace_id: str,
    idempotency_key: str,
) -> str:
    """Public stable Knowledge Input identity (same algorithm as 1B-1)."""
    return _stable_id("ki", tenant_id, workspace_id, idempotency_key)


def _input_id(*, tenant_id: str, workspace_id: str, idempotency_key: str) -> str:
    return deterministic_knowledge_input_id(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        idempotency_key=idempotency_key,
    )


def _operation_id(*, input_id: str) -> str:
    return _stable_id("op", "knowledge_ingestion", input_id)


def _suggested_source_id(*, input_id: str) -> str:
    return _stable_id("src", "knowledge_input_source", input_id)


def _metadata_equal(left: Mapping[str, str], right: Mapping[str, str]) -> bool:
    return dict(left) == dict(right)


class KnowledgeIntakeService:
    """Accepts a Knowledge Input, resolves Source, persists operation, enqueues job."""

    def __init__(
        self,
        repository: ManagedWorkspaceRepository,
        source_resolver: KnowledgeInputSourceResolver,
        wiring_context: ToolWiringContext,
    ) -> None:
        self._repository = repository
        self._source_resolver = source_resolver
        self._wiring_context = wiring_context

    def accept(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        input_kind: KnowledgeInputKind,
        idempotency_key: str,
        submission_metadata: Mapping[str, str] | None = None,
    ) -> KnowledgeIntakeAcceptance:
        tenant_id = tenant_id.strip()
        workspace_id = workspace_id.strip()
        idempotency_key = idempotency_key.strip()
        if not tenant_id or not workspace_id or not idempotency_key:
            raise ValueError("tenant_workspace_idempotency_required")

        workspace = self._repository.get_workspace(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        if workspace is None:
            raise LookupError("workspace_not_found")

        metadata = dict(submission_metadata or {})
        input_id = _input_id(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            idempotency_key=idempotency_key,
        )
        operation_id = _operation_id(input_id=input_id)
        suggested_source_id = _suggested_source_id(input_id=input_id)

        existing = self._repository.get_knowledge_input(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            input_id=input_id,
        )
        if existing is not None:
            if existing.input_kind is not input_kind or not _metadata_equal(
                existing.submission_metadata,
                metadata,
            ):
                raise KnowledgeInputIdempotencyConflict("knowledge_input_idempotency_conflict")
            return self._resume(
                knowledge_input=existing,
                suggested_source_id=suggested_source_id,
            )

        now = _utc_now()
        knowledge_input = KnowledgeInput(
            input_id=input_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            input_kind=input_kind,
            idempotency_key=idempotency_key,
            operation_id=operation_id,
            source_id=None,
            status=KnowledgeInputStatus.ACCEPTED,
            submission_metadata=metadata,
            created_at=now,
            updated_at=now,
            error_code=None,
        )
        self._repository.put_knowledge_input(knowledge_input)
        return self._resume(
            knowledge_input=knowledge_input,
            suggested_source_id=suggested_source_id,
        )

    def reconcile_workspace(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> int:
        tenant_id = tenant_id.strip()
        workspace_id = workspace_id.strip()
        if not tenant_id or not workspace_id:
            raise ValueError("tenant_workspace_required")

        resumed = 0
        for knowledge_input in self._repository.list_knowledge_inputs(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        ):
            if knowledge_input.status not in {
                KnowledgeInputStatus.ACCEPTED,
                KnowledgeInputStatus.RESOLVED,
            }:
                continue
            suggested_source_id = _suggested_source_id(input_id=knowledge_input.input_id)
            before = self._snapshot(knowledge_input)
            self._resume(
                knowledge_input=knowledge_input,
                suggested_source_id=suggested_source_id,
            )
            after_input = self._repository.get_knowledge_input(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                input_id=knowledge_input.input_id,
            )
            after_op = self._repository.get_operation(
                tenant_id=tenant_id,
                operation_id=knowledge_input.operation_id,
            )
            if before != (after_input, after_op):
                resumed += 1

        for operation in self._repository.list_ingestion_operations(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            statuses={
                WorkspaceOperationStatus.ACCEPTED,
                WorkspaceOperationStatus.QUEUED,
            },
        ):
            if not operation.input_id:
                continue
            knowledge_input = self._repository.get_knowledge_input(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                input_id=operation.input_id,
            )
            source = self._repository.get_source(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                source_id=operation.source_id,
            )
            if knowledge_input is None or source is None:
                continue
            before_task = operation.queue_task_id
            before_status = operation.status
            updated = self._dispatch(knowledge_input=knowledge_input, source=source, operation=operation)
            if updated.queue_task_id != before_task or updated.status != before_status:
                resumed += 1
        return resumed

    def _snapshot(
        self,
        knowledge_input: KnowledgeInput,
    ) -> tuple[KnowledgeInput | None, WorkspaceOperation | None]:
        return (
            knowledge_input,
            self._repository.get_operation(
                tenant_id=knowledge_input.tenant_id,
                operation_id=knowledge_input.operation_id,
            ),
        )

    def _resume(
        self,
        *,
        knowledge_input: KnowledgeInput,
        suggested_source_id: str,
    ) -> KnowledgeIntakeAcceptance:
        source = self._resolve_and_persist_source(
            knowledge_input=knowledge_input,
            suggested_source_id=suggested_source_id,
        )
        knowledge_input = self._mark_resolved(knowledge_input=knowledge_input, source=source)
        operation = self._ensure_operation(knowledge_input=knowledge_input, source=source)
        operation = self._dispatch(
            knowledge_input=knowledge_input,
            source=source,
            operation=operation,
        )
        return KnowledgeIntakeAcceptance(
            knowledge_input=knowledge_input,
            source=source,
            operation=operation,
        )

    def _resolve_and_persist_source(
        self,
        *,
        knowledge_input: KnowledgeInput,
        suggested_source_id: str,
    ) -> WorkspaceSource:
        if knowledge_input.source_id:
            existing = self._repository.get_source(
                tenant_id=knowledge_input.tenant_id,
                workspace_id=knowledge_input.workspace_id,
                source_id=knowledge_input.source_id,
            )
            if existing is not None:
                if (
                    existing.tenant_id != knowledge_input.tenant_id
                    or existing.workspace_id != knowledge_input.workspace_id
                ):
                    raise KnowledgeIntakeStateConflict("knowledge_input_source_state_conflict")
                return existing

        try:
            resolved = self._source_resolver.resolve(
                knowledge_input=knowledge_input,
                suggested_source_id=suggested_source_id,
            )
        except Exception as exc:  # noqa: BLE001 - map to domain error
            raise KnowledgeInputResolutionError(type(exc).__name__) from None

        self._validate_resolved_source(
            knowledge_input=knowledge_input,
            source=resolved,
            suggested_source_id=suggested_source_id,
        )
        return self._repository.put_source(resolved)

    def _validate_resolved_source(
        self,
        *,
        knowledge_input: KnowledgeInput,
        source: WorkspaceSource,
        suggested_source_id: str,
    ) -> None:
        if not source.source_id.strip():
            raise KnowledgeInputResolutionError("source_id_required")
        if source.tenant_id != knowledge_input.tenant_id:
            raise KnowledgeInputResolutionError("source_tenant_mismatch")
        if source.workspace_id != knowledge_input.workspace_id:
            raise KnowledgeInputResolutionError("source_workspace_mismatch")

        existing = self._repository.get_source(
            tenant_id=knowledge_input.tenant_id,
            workspace_id=knowledge_input.workspace_id,
            source_id=source.source_id,
        )
        if existing is None and source.source_id != suggested_source_id:
            raise KnowledgeInputResolutionError("suggested_source_id_required")

    def _mark_resolved(
        self,
        *,
        knowledge_input: KnowledgeInput,
        source: WorkspaceSource,
    ) -> KnowledgeInput:
        if (
            knowledge_input.source_id == source.source_id
            and knowledge_input.status is KnowledgeInputStatus.RESOLVED
        ):
            return knowledge_input
        updated = knowledge_input.model_copy(
            update={
                "source_id": source.source_id,
                "status": KnowledgeInputStatus.RESOLVED,
                "updated_at": _utc_now(),
            }
        )
        return self._repository.put_knowledge_input(updated)

    def _ensure_operation(
        self,
        *,
        knowledge_input: KnowledgeInput,
        source: WorkspaceSource,
    ) -> WorkspaceOperation:
        expected_operation_id = _operation_id(input_id=knowledge_input.input_id)
        if knowledge_input.operation_id != expected_operation_id:
            raise KnowledgeIntakeStateConflict("knowledge_input_operation_id_conflict")

        existing = self._repository.get_operation(
            tenant_id=knowledge_input.tenant_id,
            operation_id=knowledge_input.operation_id,
        )
        if existing is not None:
            if (
                existing.tenant_id != knowledge_input.tenant_id
                or existing.workspace_id != knowledge_input.workspace_id
                or existing.operation_type is not WorkspaceOperationType.KNOWLEDGE_INGESTION
                or existing.input_id != knowledge_input.input_id
                or existing.source_id != source.source_id
            ):
                raise KnowledgeIntakeStateConflict("knowledge_intake_operation_state_conflict")
            self._ensure_active_locator(existing)
            return existing
        now = _utc_now()
        operation = WorkspaceOperation(
            operation_id=knowledge_input.operation_id,
            tenant_id=knowledge_input.tenant_id,
            workspace_id=knowledge_input.workspace_id,
            source_id=source.source_id,
            operation_type=WorkspaceOperationType.KNOWLEDGE_INGESTION,
            status=WorkspaceOperationStatus.ACCEPTED,
            input_id=knowledge_input.input_id,
            created_at=now,
        )
        persisted = self._repository.put_operation(operation)
        self._ensure_active_locator(persisted)
        return persisted

    def _ensure_active_locator(self, operation: WorkspaceOperation) -> None:
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

    def _dispatch(
        self,
        *,
        knowledge_input: KnowledgeInput,
        source: WorkspaceSource,
        operation: WorkspaceOperation,
    ) -> WorkspaceOperation:
        if operation.status in {
            WorkspaceOperationStatus.COMPLETED,
            WorkspaceOperationStatus.FAILED,
            WorkspaceOperationStatus.PROCESSING,
        }:
            if operation.status is WorkspaceOperationStatus.PROCESSING:
                self._ensure_active_locator(operation)
            return operation

        if operation.status not in {
            WorkspaceOperationStatus.ACCEPTED,
            WorkspaceOperationStatus.QUEUED,
        }:
            return operation

        self._ensure_active_locator(operation)

        job = KnowledgeIngestionJob(
            tenant_id=knowledge_input.tenant_id,
            workspace_id=knowledge_input.workspace_id,
            input_id=knowledge_input.input_id,
            source_id=source.source_id,
            operation_id=operation.operation_id,
        )
        try:
            output = message_bus_enqueue(
                self._wiring_context,
                build_knowledge_ingestion_enqueue_input(job),
            )
        except Exception as exc:  # noqa: BLE001 - persist fail-closed evidence
            failed = operation.model_copy(
                update={
                    "status": WorkspaceOperationStatus.FAILED,
                    "error_code": "enqueue_failed",
                    "error": type(exc).__name__,
                    "completed_at": _utc_now(),
                }
            )
            self._repository.put_operation(failed)
            self._clear_active_locator(failed.operation_id)
            raise KnowledgeIntakeDispatchError(type(exc).__name__) from None

        queued = operation.model_copy(
            update={
                "status": WorkspaceOperationStatus.QUEUED,
                "queue_task_id": output.task_id,
                "queue_provider": output.provider,
                "error": None,
                "error_code": None,
                "completed_at": None,
            }
        )
        persisted = self._repository.put_operation(queued)
        self._ensure_active_locator(persisted)
        return persisted

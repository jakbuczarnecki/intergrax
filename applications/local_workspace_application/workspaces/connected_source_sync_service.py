# © Artur Czarnecki. All rights reserved.

"""Connected workspace source synchronization orchestration."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from intergrax.integrations.contracts.document_store import ConditionalDocumentStore
from intergrax.runtime.vendor_knowledge.bindings import KnowledgeSourceBindingService
from intergrax.runtime.vendor_knowledge.contracts import VendorKnowledgeFacade
from intergrax.runtime.vendor_knowledge.sync_coordinator import VendorKnowledgeSyncCoordinator
from intergrax.runtime.vendor_knowledge.sync_document_store import (
    DocumentStoreKnowledgeRemoteItemStateRepository,
    DocumentStoreKnowledgeSourceLeaseRepository,
    DocumentStoreKnowledgeSyncCheckpointRepository,
)
from intergrax.runtime.vendor_knowledge.sync_models import KnowledgeSyncRunStatus
from local_workspace_application.workspaces.connected_source_sync_sink import (
    ConnectedSourceSyncSinkContext,
    WorkspaceConnectedSourceKnowledgeSyncSink,
)
from local_workspace_application.workspaces.document_indexing import WorkspaceDocumentIndexingService
from local_workspace_application.workspaces.knowledge_configuration_models import (
    WorkspaceIndexedSourceBindingStatusV1,
)
from local_workspace_application.workspaces.knowledge_configuration_service import (
    WorkspaceKnowledgeConfigurationService,
)
from local_workspace_application.workspaces.models import (
    WorkspaceOperation,
    WorkspaceOperationStatus,
    WorkspaceSource,
    WorkspaceSourceStatus,
    WorkspaceSourceType,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.sync_jobs import ManagedWorkspaceSyncJob

logger = logging.getLogger(__name__)


class ConnectedSourceSyncContinuationPort(Protocol):
    def requeue(self, job: ManagedWorkspaceSyncJob) -> None:
        ...


@dataclass(frozen=True, slots=True)
class ConnectedSourceSyncDependencies:
    binding_service: KnowledgeSourceBindingService
    facade: VendorKnowledgeFacade
    owner_id: str


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _safe_error_message(exc: BaseException) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    first_line = message.splitlines()[0].strip()
    return first_line[:500]


class ManagedWorkspaceConnectedSourceSyncService:
    def __init__(
        self,
        repository: ManagedWorkspaceRepository,
        indexing_service: WorkspaceDocumentIndexingService,
        configuration_reader: WorkspaceKnowledgeConfigurationService,
        dependencies_factory: Callable[[str], ConnectedSourceSyncDependencies],
        *,
        page_size: int = 50,
        max_pages_per_operation: int = 8,
        continuation: ConnectedSourceSyncContinuationPort | None = None,
        lease_ttl_seconds: int = 60,
    ) -> None:
        self._repository = repository
        self._indexing_service = indexing_service
        self._configuration_reader = configuration_reader
        self._dependencies_factory = dependencies_factory
        self._page_size = page_size
        self._max_pages_per_operation = max_pages_per_operation
        self._continuation = continuation
        self._lease_ttl_seconds = lease_ttl_seconds

    async def run_operation(self, *, tenant_id: str, operation_id: str) -> WorkspaceOperation:
        operation = self._repository.get_operation(tenant_id=tenant_id, operation_id=operation_id)
        if operation is None:
            raise LookupError("operation_not_found")

        if operation.status in {
            WorkspaceOperationStatus.COMPLETED,
            WorkspaceOperationStatus.FAILED,
        }:
            return operation
        if operation.status is WorkspaceOperationStatus.RUNNING:
            logger.warning(
                "connected_source_sync_duplicate_while_running operation_id=%s",
                operation.operation_id,
            )
            return operation
        if operation.status is not WorkspaceOperationStatus.QUEUED:
            return self._fail(operation, f"unexpected_operation_status:{operation.status.value}")

        source = self._repository.get_source(
            tenant_id=tenant_id,
            workspace_id=operation.workspace_id,
            source_id=operation.source_id,
        )
        if source is None:
            return self._fail(operation, "source_not_found")
        if source.source_type is not WorkspaceSourceType.CONNECTED_SOURCE:
            return self._fail(operation, "source_sync_unsupported_for_source_type")

        binding_ref, indexed_binding_id = self._resolve_indexed_source_binding(
            tenant_id=tenant_id,
            workspace_id=operation.workspace_id,
            source_id=source.source_id,
        )
        if binding_ref is None:
            return self._fail(operation, "indexed_source_binding_not_found")

        started = _utc_now()
        operation = operation.model_copy(
            update={
                "status": WorkspaceOperationStatus.RUNNING,
                "started_at": started,
                "error": None,
            }
        )
        self._repository.put_operation(operation)
        self._repository.put_source(
            source.model_copy(update={"status": WorkspaceSourceStatus.SYNCING})
        )

        documents_indexed = operation.documents_indexed
        documents_unchanged = operation.documents_unchanged
        has_more = False

        try:
            dependencies = self._dependencies_factory(tenant_id)
            sink_context = ConnectedSourceSyncSinkContext(
                tenant_id=tenant_id,
                workspace_id=operation.workspace_id,
                source_id=source.source_id,
                indexed_source_binding_id=indexed_binding_id,
                knowledge_source_binding_ref=binding_ref,
                operation_id=operation.operation_id,
            )
            sink = WorkspaceConnectedSourceKnowledgeSyncSink(
                repository=self._repository,
                indexing_service=self._indexing_service,
                context=sink_context,
            )
            coordinator = self._build_coordinator(
                tenant_id=tenant_id,
                dependencies=dependencies,
                sink=sink,
            )

            restart = True
            for _ in range(self._max_pages_per_operation):
                result = await coordinator.reconcile_once(
                    binding_id=binding_ref,
                    page_size=self._page_size,
                    restart=restart,
                )
                restart = False
                if result.status is KnowledgeSyncRunStatus.LEASE_BUSY:
                    return self._requeue(operation, source)
                documents_indexed += result.active_count
                documents_unchanged += max(
                    0,
                    result.changes_count - result.active_count - result.tombstone_count,
                )
                if not result.has_more:
                    has_more = False
                    break
                has_more = True
        except Exception as exc:
            logger.exception(
                "connected_source_sync_failed operation_id=%s",
                operation.operation_id,
            )
            failed = self._fail(operation, _safe_error_message(exc))
            self._repository.put_source(
                source.model_copy(update={"status": WorkspaceSourceStatus.ERROR})
            )
            return failed

        if has_more:
            return self._requeue(
                operation.model_copy(
                    update={
                        "documents_indexed": documents_indexed,
                        "documents_unchanged": documents_unchanged,
                    }
                ),
                source,
            )

        completed = _utc_now()
        operation = operation.model_copy(
            update={
                "status": WorkspaceOperationStatus.COMPLETED,
                "documents_indexed": documents_indexed,
                "documents_unchanged": documents_unchanged,
                "completed_at": completed,
                "error": None,
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
        return operation

    def _build_coordinator(
        self,
        *,
        tenant_id: str,
        dependencies: ConnectedSourceSyncDependencies,
        sink: WorkspaceConnectedSourceKnowledgeSyncSink,
    ) -> VendorKnowledgeSyncCoordinator:
        document_store = self._repository.document_store
        if not isinstance(document_store, ConditionalDocumentStore):
            raise RuntimeError("connected_source_sync_requires_conditional_document_store")
        return VendorKnowledgeSyncCoordinator(
            tenant_id=tenant_id,
            owner_id=dependencies.owner_id,
            binding_service=dependencies.binding_service,
            facade=dependencies.facade,
            lease_repository=DocumentStoreKnowledgeSourceLeaseRepository(document_store),
            checkpoint_repository=DocumentStoreKnowledgeSyncCheckpointRepository(document_store),
            item_state_repository=DocumentStoreKnowledgeRemoteItemStateRepository(document_store),
            sink=sink,
            lease_ttl_seconds=self._lease_ttl_seconds,
        )

    def _resolve_indexed_source_binding(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
    ) -> tuple[str | None, str]:
        configuration = self._configuration_reader.get_configuration(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        if configuration is None:
            return None, ""
        for binding in configuration.indexed_sources:
            if binding.source_id != source_id:
                continue
            if binding.status is not WorkspaceIndexedSourceBindingStatusV1.ACTIVE:
                return None, binding.indexed_source_binding_id
            return binding.knowledge_source_binding_ref, binding.indexed_source_binding_id
        return None, ""

    def _requeue(self, operation: WorkspaceOperation, source: WorkspaceSource) -> WorkspaceOperation:
        requeued = operation.model_copy(
            update={
                "status": WorkspaceOperationStatus.QUEUED,
                "error": None,
            }
        )
        self._repository.put_operation(requeued)
        self._repository.put_source(
            source.model_copy(update={"status": WorkspaceSourceStatus.SYNCING})
        )
        if self._continuation is not None:
            self._continuation.requeue(
                ManagedWorkspaceSyncJob(
                    tenant_id=operation.tenant_id,
                    workspace_id=operation.workspace_id,
                    source_id=operation.source_id,
                    operation_id=operation.operation_id,
                )
            )
        return requeued

    def _fail(self, operation: WorkspaceOperation, error: str) -> WorkspaceOperation:
        failed = operation.model_copy(
            update={
                "status": WorkspaceOperationStatus.FAILED,
                "error": error,
                "started_at": operation.started_at or _utc_now(),
                "completed_at": _utc_now(),
            }
        )
        self._repository.put_operation(failed)
        return failed

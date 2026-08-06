# © Artur Czarnecki. All rights reserved.

"""Connected workspace source synchronization orchestration."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from local_workspace_application.workspaces.connected_source_delivery import (
    ConnectedSourceDeliveryApplyResult,
)
from local_workspace_application.workspaces.connected_source_models import (
    ConnectedSourceReconciliationStateV1,
    ConnectedSourceSyncSinkError,
)
from local_workspace_application.workspaces.connected_source_operation_accounting import (
    apply_completed_delivery_accounting,
)
from local_workspace_application.workspaces.connected_source_reconciliation import (
    resolve_connected_source_restart,
)
from local_workspace_application.workspaces.connected_source_source_projection import (
    repair_connected_source_source_projection,
)
from local_workspace_application.workspaces.connected_source_sync_enqueue import (
    durable_requeue_connected_source_operation,
)
from local_workspace_application.workspaces.connected_source_sync_sink import (
    ConnectedSourceSyncSinkContext,
    WorkspaceConnectedSourceKnowledgeSyncSink,
)
from local_workspace_application.workspaces.document_indexing import (
    WorkspaceDocumentIndexingService,
)
from local_workspace_application.workspaces.knowledge_access_service import (
    TenantKnowledgeSourceBindingPort,
)
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

from intergrax.integrations.contracts.document_store import ConditionalDocumentStore
from intergrax.runtime.vendor_knowledge.bindings import KnowledgeSourceBindingService
from intergrax.runtime.vendor_knowledge.contracts import VendorKnowledgeFacade
from intergrax.runtime.vendor_knowledge.errors import VendorKnowledgeError
from intergrax.runtime.vendor_knowledge.sync_coordinator import (
    VendorKnowledgeSyncCoordinator,
)
from intergrax.runtime.vendor_knowledge.sync_document_store import (
    DocumentStoreKnowledgeReconciliationCandidateInventoryRepository,
    DocumentStoreKnowledgeReconciliationRunRepository,
    DocumentStoreKnowledgeRemoteItemStateRepository,
    DocumentStoreKnowledgeSourceLeaseRepository,
    DocumentStoreKnowledgeSyncCheckpointRepository,
)
from intergrax.runtime.vendor_knowledge.sync_models import (
    KnowledgeReconciliationRunPhase,
    KnowledgeSyncRunStatus,
)
from intergrax.runtime.vendor_knowledge.sync_publication_fence import (
    DocumentStoreKnowledgeSyncPublicationFenceRepository,
)
from intergrax.runtime.vendor_knowledge.sync_reconciliation import (
    derive_reconciliation_run_id,
)
from intergrax.tools.registry.wiring import ToolWiringContext

logger = logging.getLogger(__name__)


class ConnectedSourceSyncContinuationPort(Protocol):
    def requeue(self, job: ManagedWorkspaceSyncJob) -> None:
        ...


@dataclass(frozen=True, slots=True)
class ConnectedSourceSyncDependencies:
    binding_service: KnowledgeSourceBindingService
    facade: VendorKnowledgeFacade
    owner_id: str


class _DeliveryCountingSink:
    def __init__(
        self,
        inner: WorkspaceConnectedSourceKnowledgeSyncSink,
        *,
        on_apply: Callable[[str, ConnectedSourceDeliveryApplyResult], None],
    ) -> None:
        self._inner = inner
        self._on_apply = on_apply

    async def apply_batch(self, *, batch) -> ConnectedSourceDeliveryApplyResult:
        result = await self._inner.apply_batch(batch=batch)
        self._on_apply(batch.delivery_id, result)
        return result

    def set_publication_validator(self, validator) -> None:
        self._inner.set_publication_validator(validator)

    def inspect_receipt(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        delivery_id: str,
        prepared_batch_payload_fingerprint: str,
    ):
        return self._inner.inspect_receipt(
            tenant_id=tenant_id,
            binding_id=binding_id,
            delivery_id=delivery_id,
            prepared_batch_payload_fingerprint=prepared_batch_payload_fingerprint,
        )


def _utc_now() -> datetime:
    return datetime.now(UTC)


class ManagedWorkspaceConnectedSourceSyncService:
    def __init__(
        self,
        repository: ManagedWorkspaceRepository,
        indexing_service: WorkspaceDocumentIndexingService,
        configuration_reader: WorkspaceKnowledgeConfigurationService,
        tenant_binding_port: TenantKnowledgeSourceBindingPort,
        dependencies_factory: Callable[[str], ConnectedSourceSyncDependencies],
        *,
        page_size: int = 50,
        max_pages_per_operation: int = 8,
        max_duration_seconds: float = 120.0,
        continuation: ConnectedSourceSyncContinuationPort | None = None,
        sync_enqueue_context: ToolWiringContext | None = None,
        lease_ttl_seconds: int = 60,
    ) -> None:
        self._repository = repository
        self._indexing_service = indexing_service
        self._configuration_reader = configuration_reader
        self._tenant_binding_port = tenant_binding_port
        self._dependencies_factory = dependencies_factory
        self._page_size = page_size
        self._max_pages_per_operation = max_pages_per_operation
        self._max_duration_seconds = max_duration_seconds
        self._continuation = continuation
        self._sync_enqueue_context = sync_enqueue_context
        self._lease_ttl_seconds = lease_ttl_seconds

    def attach_continuation(self, continuation: ConnectedSourceSyncContinuationPort) -> None:
        self._continuation = continuation

    def attach_sync_enqueue_context(self, context: ToolWiringContext) -> None:
        self._sync_enqueue_context = context

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
            reloaded = self._repository.get_operation(tenant_id=tenant_id, operation_id=operation_id)
            return reloaded or operation
        if operation.status is not WorkspaceOperationStatus.QUEUED:
            return self._fail(operation, "unexpected_operation_status")

        source = self._repository.get_source(
            tenant_id=tenant_id,
            workspace_id=operation.workspace_id,
            source_id=operation.source_id,
        )
        if source is None:
            return self._fail(operation, "source_not_found")
        if source.source_type is not WorkspaceSourceType.CONNECTED_SOURCE:
            return self._fail(operation, "source_sync_unsupported_for_source_type")

        binding_ref, indexed_binding_id, binding_configuration_version = (
            self._resolve_indexed_source_binding(
                tenant_id=tenant_id,
                workspace_id=operation.workspace_id,
                source_id=source.source_id,
            )
        )
        if binding_ref is None:
            failed = self._fail(operation, "indexed_source_binding_not_found")
            repair_connected_source_source_projection(
                repository=self._repository,
                tenant_id=tenant_id,
                workspace_id=operation.workspace_id,
                source_id=source.source_id,
            )
            return failed

        document_store = self._repository.document_store
        if not isinstance(document_store, ConditionalDocumentStore):
            raise RuntimeError(  # noqa: TRY004
                "connected_source_sync_requires_conditional_document_store"
            )
        publication_fence_port = DocumentStoreKnowledgeSyncPublicationFenceRepository(
            document_store
        )
        checkpoint_repository = DocumentStoreKnowledgeSyncCheckpointRepository(
            document_store,
            publication_fence_port=publication_fence_port,
        )
        reconciliation_run_repository = DocumentStoreKnowledgeReconciliationRunRepository(
            document_store,
            publication_fence_port=publication_fence_port,
        )
        restart_decision = resolve_connected_source_restart(
            repository=self._repository,
            checkpoint_repository=checkpoint_repository,
            tenant_id=tenant_id,
            binding_ref=binding_ref,
            binding_configuration_version=binding_configuration_version,
            operation=operation,
        )
        operation = restart_decision.operation
        restart = restart_decision.restart
        trigger_delivery_id: str | None = None
        reconciliation_run = reconciliation_run_repository.get(
            tenant_id=tenant_id,
            binding_id=binding_ref,
        )
        expected_run_id = derive_reconciliation_run_id(
            tenant_id=tenant_id,
            binding_id=binding_ref,
            operation_id=operation.operation_id,
        )
        if reconciliation_run is None or reconciliation_run.run_id != expected_run_id:
            restart = True
        if (
            reconciliation_run is not None
            and reconciliation_run.run_id == expected_run_id
            and reconciliation_run.phase is KnowledgeReconciliationRunPhase.COMPLETED
        ):
            final_delivery_id = getattr(reconciliation_run, "final_delivery_id", None)
            if isinstance(final_delivery_id, str) and final_delivery_id:
                operation, _accounting = apply_completed_delivery_accounting(
                    repository=self._repository,
                    operation=operation,
                    delivery_id=final_delivery_id,
                )
            completed = operation.model_copy(
                update={
                    "status": WorkspaceOperationStatus.COMPLETED,
                    "completed_at": operation.completed_at or _utc_now(),
                    "error": None,
                }
            )
            self._repository.put_operation(completed)
            repair_connected_source_source_projection(
                repository=self._repository,
                tenant_id=tenant_id,
                workspace_id=completed.workspace_id,
                source_id=completed.source_id,
            )
            return completed
        if (
            reconciliation_run is not None
            and reconciliation_run.run_id == expected_run_id
            and reconciliation_run.phase
            in {
                KnowledgeReconciliationRunPhase.COLLECTING,
                KnowledgeReconciliationRunPhase.PAGE_PREPARED,
                KnowledgeReconciliationRunPhase.FINALIZING,
                KnowledgeReconciliationRunPhase.RECOVERY_REQUIRED,
            }
        ):
            trigger_delivery_id = (
                getattr(reconciliation_run, "last_applied_delivery_id", None)
                or getattr(reconciliation_run, "prepared_parent_delivery_id", None)
            )
            if (
                reconciliation_run.phase is KnowledgeReconciliationRunPhase.PAGE_PREPARED
                and trigger_delivery_id is None
            ):
                restart = True
            else:
                restart = False
            if (
                operation.connected_source_reconciliation_state
                is not ConnectedSourceReconciliationStateV1.CONTINUATION
            ):
                operation = operation.model_copy(
                    update={
                        "connected_source_reconciliation_state": (
                            ConnectedSourceReconciliationStateV1.CONTINUATION
                        )
                    }
                )
                self._repository.put_operation(operation)
        reconciliation_state = (
            operation.connected_source_reconciliation_state
            or ConnectedSourceReconciliationStateV1.NEW_RECONCILIATION
        )

        claimed = operation.model_copy(
            update={
                "status": WorkspaceOperationStatus.RUNNING,
                "started_at": operation.started_at or _utc_now(),
                "error": None,
                "connected_source_reconciliation_state": reconciliation_state,
            }
        )
        if not self._repository.claim_operation_if_queued(
            expected=operation,
            replacement=claimed,
        ):
            reloaded = self._repository.get_operation(tenant_id=tenant_id, operation_id=operation_id)
            return reloaded or operation
        operation = claimed
        self._repository.put_source(
            source.model_copy(update={"status": WorkspaceSourceStatus.SYNCING})
        )

        has_more = False
        worker_started = time.monotonic()

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
            inner_sink = WorkspaceConnectedSourceKnowledgeSyncSink(
                repository=self._repository,
                indexing_service=self._indexing_service,
                configuration_reader=self._configuration_reader,
                tenant_binding_port=self._tenant_binding_port,
                context=sink_context,
                publication_fence_port=DocumentStoreKnowledgeSyncPublicationFenceRepository(
                    document_store
                ),
            )

            def _record_delivery(delivery_id: str, result: ConnectedSourceDeliveryApplyResult) -> None:
                nonlocal operation
                operation, _accounting = apply_completed_delivery_accounting(
                    repository=self._repository,
                    operation=operation,
                    delivery_id=delivery_id,
                    sink_result=result,
                )

            sink = _DeliveryCountingSink(inner_sink, on_apply=_record_delivery)
            coordinator = self._build_coordinator(
                tenant_id=tenant_id,
                dependencies=dependencies,
                sink=sink,
            )

            for _ in range(self._max_pages_per_operation):
                if time.monotonic() - worker_started > self._max_duration_seconds:
                    has_more = True
                    break
                result = await coordinator.reconcile_once(
                    binding_id=binding_ref,
                    page_size=self._page_size,
                    restart=restart,
                    operation_id=operation.operation_id,
                    trigger_delivery_id=trigger_delivery_id,
                )
                if result.status is KnowledgeSyncRunStatus.LEASE_BUSY:
                    return self._requeue_lease_busy(operation, source)
                if result.delivery_id is not None and (
                    operation.connected_source_reconciliation_state
                    is ConnectedSourceReconciliationStateV1.NEW_RECONCILIATION
                ):
                    trigger_delivery_id = result.delivery_id
                    operation = operation.model_copy(
                        update={
                            "connected_source_reconciliation_state": (
                                ConnectedSourceReconciliationStateV1.CONTINUATION
                            )
                        }
                    )
                    self._repository.put_operation(operation)
                    restart = False
                elif operation.connected_source_reconciliation_state is (
                    ConnectedSourceReconciliationStateV1.CONTINUATION
                ):
                    trigger_delivery_id = result.delivery_id or trigger_delivery_id
                    restart = False
                if not result.has_more:
                    has_more = False
                    break
                has_more = True
        except ConnectedSourceSyncSinkError as exc:
            logger.exception(
                "connected_source_sync_sink_failed operation_id=%s error=%s",
                operation.operation_id,
                exc.error_code,
            )
            failed = self._fail(operation, exc.error_code)
            repair_connected_source_source_projection(
                repository=self._repository,
                tenant_id=tenant_id,
                workspace_id=operation.workspace_id,
                source_id=source.source_id,
            )
            return failed
        except VendorKnowledgeError as exc:
            if exc.retryable:
                logger.warning(
                    "connected_source_sync_retryable operation_id=%s code=%s",
                    operation.operation_id,
                    exc.code.value,
                )
                return self._requeue(
                    operation,
                    source,
                    error_code=exc.code.value,
                )
            logger.exception(
                "connected_source_sync_vendor_failed operation_id=%s code=%s",
                operation.operation_id,
                exc.code.value,
            )
            failed = self._fail(operation, exc.code.value)
            repair_connected_source_source_projection(
                repository=self._repository,
                tenant_id=tenant_id,
                workspace_id=operation.workspace_id,
                source_id=source.source_id,
            )
            return failed
        except Exception:
            logger.exception(
                "connected_source_sync_failed operation_id=%s",
                operation.operation_id,
            )
            failed = self._fail(operation, "connected_source_sync_failed")
            repair_connected_source_source_projection(
                repository=self._repository,
                tenant_id=tenant_id,
                workspace_id=operation.workspace_id,
                source_id=source.source_id,
            )
            return failed

        if has_more:
            return self._requeue(operation, source)

        completed = _utc_now()
        operation = operation.model_copy(
            update={
                "status": WorkspaceOperationStatus.COMPLETED,
                "completed_at": completed,
                "error": None,
            }
        )
        self._repository.put_operation(operation)
        repair_connected_source_source_projection(
            repository=self._repository,
            tenant_id=tenant_id,
            workspace_id=operation.workspace_id,
            source_id=source.source_id,
        )
        return operation

    def _build_coordinator(
        self,
        *,
        tenant_id: str,
        dependencies: ConnectedSourceSyncDependencies,
        sink: _DeliveryCountingSink,
    ) -> VendorKnowledgeSyncCoordinator:
        document_store = self._repository.document_store
        if not isinstance(document_store, ConditionalDocumentStore):
            raise RuntimeError(  # noqa: TRY004
                "connected_source_sync_requires_conditional_document_store"
            )
        publication_fence_port = DocumentStoreKnowledgeSyncPublicationFenceRepository(
            document_store
        )
        return VendorKnowledgeSyncCoordinator(
            tenant_id=tenant_id,
            owner_id=dependencies.owner_id,
            binding_service=dependencies.binding_service,
            facade=dependencies.facade,
            lease_repository=DocumentStoreKnowledgeSourceLeaseRepository(document_store),
            checkpoint_repository=DocumentStoreKnowledgeSyncCheckpointRepository(
                document_store,
                publication_fence_port=publication_fence_port,
            ),
            item_state_repository=DocumentStoreKnowledgeRemoteItemStateRepository(
                document_store,
                publication_fence_port=publication_fence_port,
            ),
            sink=sink,
            lease_ttl_seconds=self._lease_ttl_seconds,
            reconciliation_run_repository=DocumentStoreKnowledgeReconciliationRunRepository(
                document_store,
                publication_fence_port=publication_fence_port,
            ),
            candidate_inventory_repository=(
                DocumentStoreKnowledgeReconciliationCandidateInventoryRepository(document_store)
            ),
            sink_receipt_inspector=sink,
            publication_fence_port=publication_fence_port,
            require_fenced_publication=True,
        )

    def _resolve_indexed_source_binding(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
    ) -> tuple[str | None, str, int]:
        configuration = self._configuration_reader.get_configuration(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        if configuration is None:
            return None, "", 0
        for binding in configuration.indexed_sources:
            if binding.source_id != source_id:
                continue
            if binding.status is not WorkspaceIndexedSourceBindingStatusV1.ACTIVE:
                return None, binding.indexed_source_binding_id, 0
            tenant_binding = self._tenant_binding_port.get_binding(
                tenant_id=tenant_id,
                binding_id=binding.knowledge_source_binding_ref,
            )
            configuration_version = (
                tenant_binding.configuration_version if tenant_binding is not None else 0
            )
            return (
                binding.knowledge_source_binding_ref,
                binding.indexed_source_binding_id,
                configuration_version,
            )
        return None, "", 0

    def _requeue_lease_busy(
        self,
        operation: WorkspaceOperation,
        source: WorkspaceSource,
    ) -> WorkspaceOperation:
        reloaded = self._repository.get_operation(
            tenant_id=operation.tenant_id,
            operation_id=operation.operation_id,
        )
        if (
            reloaded is not None
            and reloaded.status is WorkspaceOperationStatus.RUNNING
            and (reloaded.completed_at is not None or reloaded.error is not None)
        ):
            return reloaded
        return self._requeue(operation, source, error_code="lease_busy")

    def _requeue(
        self,
        operation: WorkspaceOperation,
        source: WorkspaceSource,
        *,
        error_code: str | None = None,
    ) -> WorkspaceOperation:
        requeued, _enqueue_result = durable_requeue_connected_source_operation(
            repository=self._repository,
            wiring_context=self._sync_enqueue_context,
            operation=operation,
            source_status=WorkspaceSourceStatus.SYNCING,
            error_code=error_code,
        )
        if self._continuation is not None and self._sync_enqueue_context is None:
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

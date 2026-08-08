# © Artur Czarnecki. All rights reserved.

"""Wire durable managed-workspace sync onto the platform MessageBus contract."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.integrations.contracts.message_bus import MessageBus
from intergrax.queueing.contracts.task_queue import TaskHandle, TaskRequest
from intergrax.queueing.providers.document_store import (
    DocumentStoreTaskQueue,
    DocumentStoreTaskWorker,
)
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.tools.registry.wiring import ToolWiringContext
from local_workspace_application.workspaces.knowledge_ingestion import (
    KnowledgeIngestionService,
    register_knowledge_ingestion_worker_handler,
)
from local_workspace_application.workspaces.models import (
    WorkspaceOperationStatus,
    WorkspaceSourceType,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.sync_jobs import (
    LKW_MANAGED_WORKSPACE_SYNC_TASK_NAME,
    decode_managed_workspace_sync_job,
)
from local_workspace_application.workspaces.sync_service import ManagedWorkspaceSyncService
from local_workspace_application.workspaces.sync_worker import (
    register_managed_workspace_sync_worker_handler,
)

if TYPE_CHECKING:
    from local_workspace_application.workspaces.connected_source_recovery import (
        ConnectedSourceRecoveryService,
    )
    from local_workspace_application.workspaces.ingestion_recovery import (
        KnowledgeIngestionRecoveryService,
    )


@dataclass(slots=True)
class ManagedWorkspaceSyncRuntime:
    message_bus: MessageBus
    wiring_context: ToolWiringContext
    worker: DocumentStoreTaskWorker
    registry: TaskExecutionRegistry
    _main_loop: asyncio.AbstractEventLoop | None = field(default=None, repr=False)
    _knowledge_ingestion_registered: bool = field(default=False, repr=False)
    _recovery_service: KnowledgeIngestionRecoveryService | None = field(default=None, repr=False)
    _connected_source_recovery: ConnectedSourceRecoveryService | None = field(
        default=None,
        repr=False,
    )
    _started: bool = field(default=False, repr=False)

    def bind_main_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._main_loop = loop

    def main_loop(self) -> asyncio.AbstractEventLoop | None:
        return self._main_loop

    def register_knowledge_ingestion_service(
        self,
        service: KnowledgeIngestionService,
    ) -> None:
        if self._started:
            raise RuntimeError("knowledge_ingestion_register_after_start")
        if self._knowledge_ingestion_registered:
            raise RuntimeError("knowledge_ingestion_already_registered")
        register_knowledge_ingestion_worker_handler(
            self.registry,
            service,
            main_loop_provider=self.main_loop,
        )
        self._knowledge_ingestion_registered = True

    def attach_recovery_service(
        self,
        recovery_service: KnowledgeIngestionRecoveryService,
    ) -> None:
        if self._started:
            raise RuntimeError("knowledge_ingestion_recovery_after_start")
        self._recovery_service = recovery_service

    def attach_connected_source_recovery(
        self,
        recovery_service: ConnectedSourceRecoveryService,
    ) -> None:
        if self._started:
            raise RuntimeError("connected_source_recovery_after_start")
        self._connected_source_recovery = recovery_service

    def start(self) -> None:
        if self._started:
            raise RuntimeError("sync_runtime_already_started")
        try:
            if self._recovery_service is not None:
                self._recovery_service.recover_all()
            if self._connected_source_recovery is not None:
                self._connected_source_recovery.recover_running_operations()
            self.worker.start()
            self._started = True
        except BaseException:
            try:
                self.worker.stop()
            except Exception:
                pass
            raise

    def stop(self) -> None:
        self.worker.stop()


def build_managed_workspace_sync_runtime(
    *,
    document_store: DocumentStore,
    sync_service: ManagedWorkspaceSyncService,
    repository: ManagedWorkspaceRepository,
    existing_message_bus: MessageBus | None = None,
    connected_source_recovery_tenant_ids: tuple[str, ...] = (),
) -> ManagedWorkspaceSyncRuntime:
    """Prefer an injected MessageBus; otherwise use durable DocumentStoreTaskQueue."""
    runtime_holder: dict[str, ManagedWorkspaceSyncRuntime] = {}
    wiring_holder: dict[str, ToolWiringContext] = {}

    def _main_loop_provider() -> asyncio.AbstractEventLoop | None:
        runtime = runtime_holder.get("runtime")
        if runtime is None:
            return None
        return runtime.main_loop()

    registry = TaskExecutionRegistry()
    register_managed_workspace_sync_worker_handler(
        registry,
        sync_service,
        main_loop_provider=_main_loop_provider,
    )

    def _on_interrupted(_handle: TaskHandle, request: TaskRequest) -> None:
        if request.task_name != LKW_MANAGED_WORKSPACE_SYNC_TASK_NAME:
            return
        job = decode_managed_workspace_sync_job(request.payload)
        operation = repository.get_operation(
            tenant_id=job.tenant_id,
            operation_id=job.operation_id,
        )
        if operation is None:
            return
        if operation.status is not WorkspaceOperationStatus.RUNNING:
            return
        source = repository.get_source(
            tenant_id=job.tenant_id,
            workspace_id=job.workspace_id,
            source_id=job.source_id,
        )
        if source is not None and source.source_type is WorkspaceSourceType.CONNECTED_SOURCE:
            from local_workspace_application.workspaces.connected_source_sync_enqueue import (
                durable_requeue_connected_source_operation,
            )

            context = wiring_holder.get("context")
            durable_requeue_connected_source_operation(
                repository=repository,
                wiring_context=context,
                operation=operation,
            )
            return
        repository.put_operation(
            operation.model_copy(
                update={
                    "status": WorkspaceOperationStatus.FAILED,
                    "error": "interrupted_by_host_restart",
                    "completed_at": datetime.now(UTC),
                }
            )
        )

    if existing_message_bus is not None:
        wiring_context = ToolWiringContext(message_bus=existing_message_bus)
        wiring_holder["context"] = wiring_context
        placeholder = DocumentStoreTaskQueue(document_store)
        worker = DocumentStoreTaskWorker(
            placeholder,
            registry,
            on_interrupted=_on_interrupted,
        )
        runtime = ManagedWorkspaceSyncRuntime(
            message_bus=existing_message_bus,
            wiring_context=wiring_context,
            worker=worker,
            registry=registry,
        )
        runtime_holder["runtime"] = runtime
        if connected_source_recovery_tenant_ids:
            from local_workspace_application.workspaces.connected_source_recovery import (
                ConnectedSourceRecoveryService,
            )

            recovery = ConnectedSourceRecoveryService(
                repository,
                wiring_context,
                tenant_ids=connected_source_recovery_tenant_ids,
            )
            runtime.attach_connected_source_recovery(recovery)
        return runtime

    durable_queue = DocumentStoreTaskQueue(document_store)
    wiring_context = ToolWiringContext(message_bus=durable_queue)
    wiring_holder["context"] = wiring_context
    worker = DocumentStoreTaskWorker(
        durable_queue,
        registry,
        on_interrupted=_on_interrupted,
    )
    runtime = ManagedWorkspaceSyncRuntime(
        message_bus=durable_queue,
        wiring_context=wiring_context,
        worker=worker,
        registry=registry,
    )
    runtime_holder["runtime"] = runtime
    if connected_source_recovery_tenant_ids:
        from local_workspace_application.workspaces.connected_source_recovery import (
            ConnectedSourceRecoveryService,
        )

        recovery = ConnectedSourceRecoveryService(
            repository,
            wiring_context,
            tenant_ids=connected_source_recovery_tenant_ids,
        )
        runtime.attach_connected_source_recovery(recovery)
    return runtime

# © Artur Czarnecki. All rights reserved.

"""Wire durable managed-workspace sync onto the platform MessageBus contract."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime

from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.integrations.contracts.message_bus import MessageBus
from intergrax.queueing.contracts.task_queue import TaskHandle, TaskRequest
from intergrax.queueing.providers.document_store import (
    DocumentStoreTaskQueue,
    DocumentStoreTaskWorker,
)
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.tools.registry.wiring import ToolWiringContext
from local_workspace_application.workspaces.models import WorkspaceOperationStatus
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.sync_jobs import (
    LKW_MANAGED_WORKSPACE_SYNC_TASK_NAME,
    decode_managed_workspace_sync_job,
)
from local_workspace_application.workspaces.sync_service import ManagedWorkspaceSyncService
from local_workspace_application.workspaces.sync_worker import (
    register_managed_workspace_sync_worker_handler,
)


@dataclass(slots=True)
class ManagedWorkspaceSyncRuntime:
    message_bus: MessageBus
    wiring_context: ToolWiringContext
    worker: DocumentStoreTaskWorker
    registry: TaskExecutionRegistry
    _main_loop: asyncio.AbstractEventLoop | None = field(default=None, repr=False)

    def bind_main_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._main_loop = loop

    def main_loop(self) -> asyncio.AbstractEventLoop | None:
        return self._main_loop

    def start(self) -> None:
        self.worker.start()

    def stop(self) -> None:
        self.worker.stop()


def build_managed_workspace_sync_runtime(
    *,
    document_store: DocumentStore,
    sync_service: ManagedWorkspaceSyncService,
    repository: ManagedWorkspaceRepository,
    existing_message_bus: MessageBus | None = None,
) -> ManagedWorkspaceSyncRuntime:
    """Prefer an injected MessageBus; otherwise use durable DocumentStoreTaskQueue."""
    runtime_holder: dict[str, ManagedWorkspaceSyncRuntime] = {}

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
        return runtime

    durable_queue = DocumentStoreTaskQueue(document_store)
    wiring_context = ToolWiringContext(message_bus=durable_queue)
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
    return runtime

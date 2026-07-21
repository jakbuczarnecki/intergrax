# © Artur Czarnecki. All rights reserved.

"""Worker handler for durable managed workspace synchronization."""

from __future__ import annotations

import asyncio
from typing import Callable, Optional, Protocol

from pydantic import BaseModel, Field

from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.tools.execution_models import ToolExecutionResult
from local_workspace_application.workspaces.models import WorkspaceOperation
from local_workspace_application.workspaces.sync_jobs import (
    LKW_MANAGED_WORKSPACE_SYNC_TASK_NAME,
    decode_managed_workspace_sync_job,
)

MainLoopProvider = Callable[[], asyncio.AbstractEventLoop | None]


class ManagedWorkspaceSyncRunner(Protocol):
    async def run_operation(self, *, tenant_id: str, operation_id: str) -> WorkspaceOperation: ...


class ManagedWorkspaceSyncWorkerOutput(BaseModel):
    operation_id: str
    status: str
    schema_version: str = "lkw.managed_workspace_sync_worker.v1"
    metadata: dict[str, object] = Field(default_factory=dict)


def make_managed_workspace_sync_worker_handler(
    sync_service: ManagedWorkspaceSyncRunner,
    *,
    main_loop_provider: MainLoopProvider | None = None,
):
    def handler(
        *,
        tenant_id: str,
        run_id: str,
        payload: bytes,
        idempotency_key: Optional[str] = None,
    ) -> ToolExecutionResult[ManagedWorkspaceSyncWorkerOutput]:
        _ = run_id
        _ = idempotency_key
        try:
            job = decode_managed_workspace_sync_job(payload)
            if job.tenant_id != tenant_id:
                return ToolExecutionResult.fail(
                    "managed_workspace_sync_tenant_mismatch",
                    "managed_workspace_sync_tenant_mismatch",
                )
            coro = sync_service.run_operation(
                tenant_id=job.tenant_id,
                operation_id=job.operation_id,
            )
            main_loop = main_loop_provider() if main_loop_provider is not None else None
            if main_loop is not None and main_loop.is_running():
                # Execute on the host loop so in-process integrations are shared.
                operation = asyncio.run_coroutine_threadsafe(coro, main_loop).result(timeout=600)
            else:
                operation = asyncio.run(coro)
            if not isinstance(operation, WorkspaceOperation):
                return ToolExecutionResult.fail(
                    "managed_workspace_sync_invalid_result",
                    "managed_workspace_sync_invalid_result",
                )
            return ToolExecutionResult.ok(
                ManagedWorkspaceSyncWorkerOutput(
                    operation_id=operation.operation_id,
                    status=operation.status.value,
                    metadata={
                        "workspace_id": operation.workspace_id,
                        "source_id": operation.source_id,
                        "documents_indexed": operation.documents_indexed,
                        "documents_unchanged": operation.documents_unchanged,
                        "error": operation.error,
                    },
                )
            )
        except Exception as exc:  # noqa: BLE001 - worker plane normalizes failures
            return ToolExecutionResult.fail(type(exc).__name__, str(exc))

    return handler


def register_managed_workspace_sync_worker_handler(
    registry: TaskExecutionRegistry,
    sync_service: ManagedWorkspaceSyncRunner,
    *,
    logical_task_name: str = LKW_MANAGED_WORKSPACE_SYNC_TASK_NAME,
    main_loop_provider: MainLoopProvider | None = None,
) -> None:
    registry.register(
        logical_task_name,
        make_managed_workspace_sync_worker_handler(
            sync_service,
            main_loop_provider=main_loop_provider,
        ),
    )

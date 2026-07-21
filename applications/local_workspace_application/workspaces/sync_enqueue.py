# © Artur Czarnecki. All rights reserved.

"""Enqueue managed workspace sync through the platform MessageBus contract."""

from __future__ import annotations

from intergrax.tools.providers.message_bus.contracts import (
    MessageBusEnqueueInput,
    MessageBusEnqueueOutput,
)
from intergrax.tools.providers.message_bus.service import message_bus_enqueue
from intergrax.tools.registry.wiring import ToolWiringContext
from local_workspace_application.workspaces.sync_jobs import (
    LKW_MANAGED_WORKSPACE_SYNC_TASK_NAME,
    ManagedWorkspaceSyncJob,
    managed_workspace_sync_idempotency_key,
    managed_workspace_sync_payload_base64,
)


def build_managed_workspace_sync_enqueue_input(
    job: ManagedWorkspaceSyncJob,
) -> MessageBusEnqueueInput:
    return MessageBusEnqueueInput(
        tenant_id=job.tenant_id,
        run_id=job.operation_id,
        task_name=LKW_MANAGED_WORKSPACE_SYNC_TASK_NAME,
        payload_base64=managed_workspace_sync_payload_base64(job),
        idempotency_key=managed_workspace_sync_idempotency_key(job),
    )


def enqueue_managed_workspace_sync(
    ctx: ToolWiringContext,
    job: ManagedWorkspaceSyncJob,
) -> MessageBusEnqueueOutput:
    return message_bus_enqueue(ctx, build_managed_workspace_sync_enqueue_input(job))

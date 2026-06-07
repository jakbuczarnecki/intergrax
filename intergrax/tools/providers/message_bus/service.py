# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import base64

from intergrax.integrations.contracts.message_bus import MessageBus
from intergrax.queueing.contracts.task_queue import TaskHandle, TaskRequest
from intergrax.tools.providers.message_bus.contracts import (
    MessageBusCancelInput,
    MessageBusCancelOutput,
    MessageBusEnqueueInput,
    MessageBusEnqueueOutput,
    MessageBusGetResultInput,
    MessageBusGetResultOutput,
    MessageBusGetStatusInput,
    MessageBusGetStatusOutput,
    MessageBusListTasksInput,
    MessageBusListTasksOutput,
    MessageBusPurgeCompletedInput,
    MessageBusPurgeCompletedOutput,
    MessageBusTaskSummaryOutput,
)
from intergrax.tools.registry.wiring import ToolWiringContext

MESSAGE_BUS_ENQUEUE_TOOL_ID = "message_bus.enqueue"
MESSAGE_BUS_GET_STATUS_TOOL_ID = "message_bus.get_status"
MESSAGE_BUS_GET_RESULT_TOOL_ID = "message_bus.get_result"
MESSAGE_BUS_LIST_TASKS_TOOL_ID = "message_bus.list_tasks"
MESSAGE_BUS_CANCEL_TOOL_ID = "message_bus.cancel"
MESSAGE_BUS_PURGE_COMPLETED_TOOL_ID = "message_bus.purge_completed"


def _require_bus(ctx: ToolWiringContext) -> MessageBus:
    bus = ctx.message_bus
    if bus is None:
        raise RuntimeError("message_bus_not_configured")
    return bus


def _handle(params_task_id: str, provider: str, tenant_id: str | None) -> TaskHandle:
    return TaskHandle(task_id=params_task_id.strip(), provider=provider.strip(), tenant_id=tenant_id)


def message_bus_enqueue(ctx: ToolWiringContext, params: MessageBusEnqueueInput) -> MessageBusEnqueueOutput:
    bus = _require_bus(ctx)
    payload = base64.b64decode(params.payload_base64)
    handle = bus.enqueue(
        TaskRequest(
            tenant_id=params.tenant_id.strip(),
            run_id=params.run_id.strip(),
            task_name=params.task_name.strip(),
            payload=payload,
            idempotency_key=params.idempotency_key,
        )
    )
    return MessageBusEnqueueOutput(
        task_id=handle.task_id,
        provider=handle.provider,
        tenant_id=handle.tenant_id,
    )


def message_bus_get_status(ctx: ToolWiringContext, params: MessageBusGetStatusInput) -> MessageBusGetStatusOutput:
    bus = _require_bus(ctx)
    status = bus.get_status(_handle(params.task_id, params.provider, params.tenant_id))
    return MessageBusGetStatusOutput(task_id=params.task_id.strip(), status=status)


def message_bus_get_result(ctx: ToolWiringContext, params: MessageBusGetResultInput) -> MessageBusGetResultOutput:
    bus = _require_bus(ctx)
    result = bus.get_result(_handle(params.task_id, params.provider, params.tenant_id))
    if result is None:
        return MessageBusGetResultOutput(task_id=params.task_id.strip(), completed=False)
    output_b64 = ""
    if result.output is not None:
        output_b64 = base64.b64encode(result.output).decode("ascii")
    return MessageBusGetResultOutput(
        task_id=params.task_id.strip(),
        completed=True,
        status=result.status,
        output_base64=output_b64,
        error_message=result.error_message or "",
        attempts=result.attempts,
    )


def message_bus_list_tasks(ctx: ToolWiringContext, params: MessageBusListTasksInput) -> MessageBusListTasksOutput:
    bus = _require_bus(ctx)
    rows = bus.list_tasks(
        params.tenant_id.strip(),
        limit=params.limit,
        status_filter=params.status_filter,
    )
    tasks = [
        MessageBusTaskSummaryOutput(
            task_id=row.task_id,
            tenant_id=row.tenant_id,
            task_name=row.task_name,
            status=row.status,
            provider=row.provider,
        )
        for row in rows
    ]
    return MessageBusListTasksOutput(tasks=tasks, total=len(tasks))


def message_bus_cancel(ctx: ToolWiringContext, params: MessageBusCancelInput) -> MessageBusCancelOutput:
    bus = _require_bus(ctx)
    cancelled = bus.cancel(_handle(params.task_id, params.provider, params.tenant_id))
    return MessageBusCancelOutput(task_id=params.task_id.strip(), cancelled=cancelled)


def message_bus_purge_completed(
    ctx: ToolWiringContext,
    params: MessageBusPurgeCompletedInput,
) -> MessageBusPurgeCompletedOutput:
    bus = _require_bus(ctx)
    purged = bus.purge_completed(
        params.tenant_id.strip(),
        older_than_seconds=params.older_than_seconds,
    )
    return MessageBusPurgeCompletedOutput(tenant_id=params.tenant_id.strip(), purged_count=purged)

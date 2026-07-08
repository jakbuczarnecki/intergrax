# © Artur Czarnecki. All rights reserved.

"""LKW background ingest enqueue helper (LKW.4C)."""

from __future__ import annotations

from intergrax.tools.providers.message_bus.contracts import MessageBusEnqueueInput, MessageBusEnqueueOutput
from intergrax.tools.providers.message_bus.service import message_bus_enqueue
from intergrax.tools.registry.wiring import ToolWiringContext

from local_workspace_application.background_ingest.contracts import (
    LKW_BACKGROUND_INGEST_TASK_NAME,
    LkwBackgroundIngestJob,
    background_ingest_idempotency_key,
    background_ingest_payload_base64,
)


def build_background_ingest_enqueue_input(
    job: LkwBackgroundIngestJob,
    *,
    run_id: str | None = None,
) -> MessageBusEnqueueInput:
    resolved_run_id = run_id or job.run_id or background_ingest_idempotency_key(job)
    return MessageBusEnqueueInput(
        tenant_id=job.tenant_id,
        run_id=resolved_run_id,
        task_name=LKW_BACKGROUND_INGEST_TASK_NAME,
        payload_base64=background_ingest_payload_base64(job),
        idempotency_key=background_ingest_idempotency_key(job),
    )


def enqueue_background_ingest_job(
    ctx: ToolWiringContext,
    job: LkwBackgroundIngestJob,
    *,
    run_id: str | None = None,
) -> MessageBusEnqueueOutput:
    params = build_background_ingest_enqueue_input(job, run_id=run_id)
    return message_bus_enqueue(ctx, params)

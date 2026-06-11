# © Artur Czarnecki. All rights reserved.

"""Drain pending compensation jobs through a declarative tool invoker."""

from __future__ import annotations

from intergrax.agents.persistence.compensation_enqueue import CompensationActionResult
from intergrax.agents.persistence.compensation_queue_store import CompensationQueueStore
from intergrax.agents.persistence.declarative_tool_executor import DeclarativeToolInvoker


async def drain_pending_compensation_jobs(
    store: CompensationQueueStore,
    *,
    tenant_id: str,
    invoker: DeclarativeToolInvoker,
    limit: int = 100,
) -> list[CompensationActionResult]:
    """Process pending compensation jobs for a tenant (worker entrypoint)."""
    results: list[CompensationActionResult] = []
    for job in store.list_pending(tenant_id, limit=limit):
        invoke_result = await invoker.invoke(
            tool_id=job.request.compensation_tool_id,
            args=job.request.args,
            idempotency_key=job.request.idempotency_key,
        )
        if invoke_result.status == "success":
            store.mark_completed(tenant_id, job.request.idempotency_key)
            results.append(
                CompensationActionResult(request=job.request, status="compensated"),
            )
            continue
        error = invoke_result.error or invoke_result.status
        store.mark_failed(tenant_id, job.request.idempotency_key, error)
        results.append(
            CompensationActionResult(
                request=job.request,
                status="failed",
                error=error,
            ),
        )
    return results

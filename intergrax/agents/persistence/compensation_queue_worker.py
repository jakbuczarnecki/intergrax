# © Artur Czarnecki. All rights reserved.

"""Drain pending compensation jobs through canonical execution admission."""

from __future__ import annotations

from uuid import uuid4

from intergrax.agents.persistence.compensation_enqueue import CompensationActionResult
from intergrax.agents.persistence.compensation_queue_store import CompensationQueueStore
from intergrax.agents.persistence.compensation_side_effect_input import (
    compensation_side_effect_input_from_job,
)
from intergrax.contracts.compensation_side_effect_execution import (
    CompensationSideEffectExecutionPort,
)

_DEFAULT_LEASE_SECONDS = 300


async def drain_pending_compensation_jobs(
    store: CompensationQueueStore,
    *,
    tenant_id: str,
    side_effect_execution: CompensationSideEffectExecutionPort,
    limit: int = 100,
    owner_id: str | None = None,
    lease_seconds: int = _DEFAULT_LEASE_SECONDS,
) -> list[CompensationActionResult]:
    """Process compensation jobs atomically claimed for a tenant (worker entrypoint)."""
    worker_owner = owner_id or f"comp-worker-{uuid4().hex}"
    results: list[CompensationActionResult] = []
    claims = store.claim_pending(
        tenant_id,
        worker_owner,
        lease_seconds,
        limit=limit,
    )
    for claim in claims:
        job = claim.job
        invoke_result = await side_effect_execution.execute(
            compensation_side_effect_input_from_job(job),
        )
        if invoke_result.status == "success":
            store.complete_claim(claim)
            results.append(
                CompensationActionResult(request=job.request, status="compensated"),
            )
            continue
        error = invoke_result.error or invoke_result.status
        store.fail_claim(claim, error, retryable=False)
        results.append(
            CompensationActionResult(
                request=job.request,
                status="failed",
                error=error,
            ),
        )
    return results

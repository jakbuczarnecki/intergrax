# © Artur Czarnecki. All rights reserved.

"""Map durable compensation jobs to admitted side-effect execution input."""

from __future__ import annotations

from intergrax.agents.persistence.compensation_queue_store import CompensationJob
from intergrax.contracts.compensation_side_effect_execution import CompensationSideEffectInput
from intergrax.contracts.execution_identity import mint_task_id, validate_run_id, validate_task_id


def compensation_side_effect_input_from_job(job: CompensationJob) -> CompensationSideEffectInput:
    suffix = job.job_id.removeprefix("cjob_")
    if len(suffix) == 32 and suffix.isalnum():
        task_id = validate_task_id(f"task_{suffix}")
    else:
        task_id = mint_task_id()
    run_id = validate_run_id(job.run_id)
    return CompensationSideEffectInput(
        tenant_id=job.tenant_id,
        run_id=str(run_id),
        agent_id=job.agent_id,
        step_index=job.step_index,
        task_id=task_id,
        compensation_tool_id=job.request.compensation_tool_id,
        args=dict(job.request.args),
        idempotency_key=job.request.idempotency_key,
        original_side_effect_id=job.request.original_side_effect_id,
    )

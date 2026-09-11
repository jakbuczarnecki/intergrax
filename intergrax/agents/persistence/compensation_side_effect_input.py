# © Artur Czarnecki. All rights reserved.

"""Map durable compensation jobs to admitted side-effect execution input."""

from __future__ import annotations

from intergrax.agents.persistence.compensation_queue_store import CompensationJob
from intergrax.contracts.compensation_side_effect_execution import CompensationSideEffectInput
from intergrax.contracts.execution_identity import validate_run_id, validate_task_id
from intergrax.knowledge.contracts.validation import JsonObject


def compensation_side_effect_input_from_job(job: CompensationJob) -> CompensationSideEffectInput:
    if job.schema_version != "compensation_job.v2":
        raise ValueError(
            f"unsupported compensation job schema_version={job.schema_version!r}; "
            "execution identity requires compensation_job.v2",
        )
    run_id = str(validate_run_id(job.run_id))
    task_id = str(validate_task_id(job.task_id))
    args: JsonObject = dict(job.request.args)
    return CompensationSideEffectInput(
        tenant_id=job.tenant_id,
        run_id=run_id,
        agent_id=job.agent_id,
        step_index=job.step_index,
        task_id=task_id,
        compensation_tool_id=job.request.compensation_tool_id,
        args=args,
        idempotency_key=job.request.idempotency_key,
        original_side_effect_id=job.request.original_side_effect_id,
    )

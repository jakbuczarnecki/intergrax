# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.agents.authoring.runtime_tool_helpers import exec_ctx_from_step, request_metadata
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.integrations.contracts.external_work import ExternalWorkIntegration
from intergrax.runtime.policy.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationBoundary,
    resolve_governed_execution_task,
)
from external_contractor_adapter.external_work_adapter import adapt_from_step_metadata

DOMAIN_STEP_ID = "external_contractor_adapter_step"


async def run_domain_job(
    step_ctx: AgentStepContext,
    *,
    external_work: ExternalWorkIntegration | None = None,
    authorization_boundary: MeaningfulSideEffectAuthorizationBoundary | None = None,
) -> dict[str, object]:
    """Map Intergrax intent through ExternalWorkIntegration (GEC-3…GEC-5).

    Sync boundary calls only — may surface a governed continuation blocker and
    compose with injected meaningful side-effect policy before mutations.
    Does not own HITL decisions, policy rules, or Nexus resume.
    """
    exec_ctx = exec_ctx_from_step(step_ctx)
    meta = request_metadata(exec_ctx, step_ctx)
    # ACP merges AgentRunRequest.metadata into step_ctx.metadata.
    merged: dict[str, object] = {**dict(step_ctx.metadata or {}), **meta}
    governed_task = None
    if exec_ctx is not None and exec_ctx.request is not None:
        request_meta = getattr(exec_ctx.request, "metadata", None)
        if isinstance(request_meta, dict):
            governed_task = resolve_governed_execution_task(request_meta)
    task_id = (step_ctx.task_id or "").strip() or (step_ctx.run_id or "").strip() or "unknown-task"
    run_id = (step_ctx.run_id or "").strip() or None
    result = adapt_from_step_metadata(
        external_work,
        task_id=task_id,
        run_id=run_id,
        message=step_ctx.message or "",
        metadata=merged,
        authorization_boundary=authorization_boundary,
        task=governed_task,
    )
    summary = result.to_domain_summary()
    answer = (
        f"external_contractor_adapter: {result.reason}"
        if result.used
        else f"external_contractor_adapter: {result.reason} (external_contractor.adapt)"
    )
    return {
        "summary": answer,
        "answer": answer,
        "run_id": step_ctx.run_id,
        "domain_summary": summary,
        "external_work": summary,
        "domain_step_id": DOMAIN_STEP_ID,
    }

# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from external_contractor_adapter.external_work_adapter import adapt_from_step_metadata
from external_contractor_adapter.sanctioned_operation_from_metadata import (
    resolve_sanctioned_operation_from_metadata,
)
from external_contractor_adapter.schemas.adapt_result import ExternalWorkAdapterResult
from intergrax.agents.authoring.runtime_tool_helpers import exec_ctx_from_step, request_metadata
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.sanctioned_external_work_operation import (
    SanctionedExternalWorkOperationPort,
)
from intergrax.integrations.contracts.external_work import ExternalWorkIntegration
from intergrax.runtime.governance.active_governed_execution_task import (
    current_governed_execution_task,
)
from intergrax.runtime.policy.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationBoundary,
)

DOMAIN_STEP_ID = "external_contractor_adapter_step"


def _fail_closed_domain_payload(reason: str) -> dict[str, object]:
    summary = ExternalWorkAdapterResult(
        used=False,
        reason=reason,
        metadata={"capability": "external_contractor.adapt"},
    ).to_domain_summary()
    answer = f"external_contractor_adapter: {reason}"
    return {
        "summary": answer,
        "answer": answer,
        "domain_summary": summary,
        "external_work": summary,
        "domain_step_id": DOMAIN_STEP_ID,
    }


async def run_domain_job(
    step_ctx: AgentStepContext,
    *,
    external_work: ExternalWorkIntegration | None = None,
    authorization_boundary: MeaningfulSideEffectAuthorizationBoundary | None = None,
    sanctioned_operation_port: SanctionedExternalWorkOperationPort | None = None,
    require_sanctioned_external_work_mutations: bool = False,
) -> dict[str, object]:
    """Map Intergrax intent through ExternalWorkIntegration (GEC-3…GEC-5).

    Sync boundary calls only — may surface a governed continuation blocker and
    compose with injected meaningful side-effect policy before mutations.
    Does not own HITL decisions, policy rules, or Nexus resume.
    """
    exec_ctx = exec_ctx_from_step(step_ctx)
    meta = request_metadata(exec_ctx, step_ctx)
    merged: dict[str, object] = {**dict(step_ctx.metadata or {}), **meta}
    governed_task = current_governed_execution_task()
    task_id = (step_ctx.task_id or "").strip() or (step_ctx.run_id or "").strip() or "unknown-task"
    run_id = (step_ctx.run_id or "").strip() or None

    if require_sanctioned_external_work_mutations:
        if sanctioned_operation_port is None:
            return _fail_closed_domain_payload(
                "sanctioned_external_work_operation_port_missing",
            )
        try:
            operation_request = resolve_sanctioned_operation_from_metadata(
                merged,
                task_id=task_id,
                run_id=run_id,
                message=step_ctx.message or "",
            )
        except ValueError as exc:
            return _fail_closed_domain_payload(str(exc))
        port_result = sanctioned_operation_port.execute(operation_request)
        summary = port_result.domain_summary.to_step_dict()
        reason = port_result.adapter_reason or port_result.reason
        answer = (
            f"external_contractor_adapter: {reason}"
            if port_result.adapter_used
            else f"external_contractor_adapter: {reason} (external_contractor.adapt)"
        )
        return {
            "summary": answer,
            "answer": answer,
            "run_id": step_ctx.run_id,
            "domain_summary": summary,
            "external_work": summary,
            "domain_step_id": DOMAIN_STEP_ID,
        }

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

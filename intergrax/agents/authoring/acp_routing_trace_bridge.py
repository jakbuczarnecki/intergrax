# © Artur Czarnecki. All rights reserved.

"""ACP Plane A routing trace bridge (M-LLM-X.13.2)."""

from __future__ import annotations

from datetime import datetime, timezone

from intergrax.llm_adapters.routing.contracts import RoutingEvaluation
from intergrax.runtime.events.trace_bridge import trace_event_to_runtime_event
from intergrax.contracts.execution_identity import (
    require_active_execution_identity,
    validate_run_id,
    validate_task_id,
)
from intergrax.runtime.kernel.step_kernel import StepKernelContext
from intergrax.runtime.nexus.tracing.adapters.llm_routing_attempt import (
    LLMRoutingRuleDiagV1,
    routing_evaluation_to_diag,
)
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceEvent, TraceLevel
from intergrax.runtime.task.task import Task


def record_acp_routing_rule_evaluation(
    kernel_ctx: StepKernelContext,
    evaluation: RoutingEvaluation,
) -> None:
    """Record routing evaluation on Plane B diagnostics and Plane A runtime events."""
    diag = routing_evaluation_to_diag(evaluation)
    kernel_ctx.routing_rule_evaluations.append(diag.to_dict())

    resolved_task_id = validate_task_id(kernel_ctx.task_id)
    resolved_run_id = validate_run_id(kernel_ctx.run_id)
    trace = TraceEvent(
        event_id=f"acp-routing-{len(kernel_ctx.events)}",
        run_id=resolved_run_id,
        seq=len(kernel_ctx.events),
        ts_utc=datetime.now(timezone.utc).isoformat(),
        level=TraceLevel.INFO,
        component=TraceComponent.ENGINE,
        step="llm_routing_rule",
        message="LLM routing rule evaluation recorded.",
        tags={
            "task_id": resolved_task_id,
            "agent_id": kernel_ctx.agent_id,
            "tenant_id": kernel_ctx.tenant_id,
        },
    )
    task = Task(
        tenant_id=kernel_ctx.tenant_id,
        user_id="",
        message="",
        task_id=resolved_task_id,
        agent_id=kernel_ctx.agent_id,
    )
    active_run_id, attempt_id = require_active_execution_identity()
    if validate_run_id(kernel_ctx.run_id) != active_run_id:
        raise RuntimeError("kernel run_id conflicts with active execution identity")
    event = trace_event_to_runtime_event(
        trace,
        task,
        run_id=active_run_id,
        attempt_id=attempt_id,
        payload_schema_id=LLMRoutingRuleDiagV1.schema_id(),
        payload_dict=diag.to_dict(),
    )
    kernel_ctx.events.append(event)

# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared UAEP helpers for pipeline-backed Tier-2 agents (§42.32)."""

from __future__ import annotations

from typing import Sequence

from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.nexus.engine.runtime import RuntimeEngine


def pipeline_agent_steps(
    *,
    step_id: str,
    step_name: str,
    trace_label: str = "",
    allowed_tools: Sequence[str] | None = None,
) -> list[AgentStep]:
    return [
        AgentStep(
            step_id=step_id,
            step_name=step_name,
            step_index=0,
            trace_label=trace_label or step_name,
            allowed_tools=list(allowed_tools or []),
        )
    ]


async def run_pipeline_step(
    step: AgentStep,
    ctx: RuntimeExecutionContext,
) -> StepOutput:
    """Run the agent pipeline via ``RuntimeEngine`` inside a UAEP step boundary."""
    request = ctx.request
    runtime_context = ctx.domain_context
    if request is None or runtime_context is None:
        raise RuntimeError("UAEP context missing request or domain_context.")

    runtime = RuntimeEngine(runtime_context)
    answer = await runtime.run(request)
    ctx.metadata["runtime_answer"] = answer

    summary = (answer.answer or "").strip()
    return StepOutput(
        step_id=step.step_id,
        summary=summary,
        data={"run_id": answer.run_id or ctx.run_id},
    )


def pipeline_step_complete(*, reason: str = "pipeline step finished") -> AgentDecision:
    return AgentDecision(type=AgentDecisionType.COMPLETE, reason=reason)

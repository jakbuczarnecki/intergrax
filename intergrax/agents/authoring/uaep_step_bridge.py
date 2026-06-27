# © Artur Czarnecki. All rights reserved.

"""UAEP run_step / decide_after_step → advance_step + HarnessKernel (ACP-STEP-3)."""

from __future__ import annotations

from typing import Any

from intergrax.agents.authoring.diagnostic_serialization import aggregate_step_diagnostics
from intergrax.agents.authoring.step_outcome import StepOutcome
from intergrax.agents.uaep_protocol import UAEPAgent, UAEPAgentWithDecide
from intergrax.contracts.acp_state import ACP_STATE_KEY, ACP_STATE_SCHEMA_VERSION
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.agent_run import AgentRunError
from intergrax.contracts.agent_run_enums import AgentRunErrorCode, TerminalReason
from intergrax.contracts.agent_run_trace import AgentRunTrace
from intergrax.contracts.agent_step import AgentStep, StepExecutionResult, StepOutput
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.contracts.step_execution import StepExecutionRecord
from intergrax.contracts.uaep_bridge_keys import UaepStateDeltaKey
from intergrax.runtime.kernel.step_kernel import HarnessKernel, StepKernelContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.policy.policy_engine import PolicyEngine


def initial_kernel_state_from_request(request: RuntimeRequest) -> dict[str, Any]:
    raw = request.metadata.get(ACP_STATE_KEY)
    if isinstance(raw, dict):
        return {ACP_STATE_KEY: dict(raw)}
    return {
        ACP_STATE_KEY: {
            "schema_version": ACP_STATE_SCHEMA_VERSION,
            "_version": 0,
        }
    }


def step_output_to_run_output(output: StepOutput) -> str | dict[str, Any]:
    if output.data:
        payload = dict(output.data)
        if output.summary:
            payload.setdefault("summary", output.summary)
        return payload
    return output.summary


def _state_delta_from_output(output: StepOutput | None) -> dict[str, Any]:
    if output is None:
        return {}
    return {
        UaepStateDeltaKey.LAST_STEP_ID: output.step_id,
        UaepStateDeltaKey.LAST_STEP_SUMMARY: output.summary,
    }


def agent_decision_to_step_outcome(
    decision: AgentDecision,
    output: StepOutput | None,
) -> StepOutcome:
    state_delta = _state_delta_from_output(output)
    run_output = step_output_to_run_output(output) if output is not None else None

    if decision.type == AgentDecisionType.CONTINUE:
        diagnostics: dict[str, Any] = {"uaep_decision": decision.type.value}
        next_step_id = decision.payload.get("next_step_id")
        if isinstance(next_step_id, str) and next_step_id:
            diagnostics["next_step_id"] = next_step_id
        return StepOutcome.continue_with(state_delta, diagnostics=diagnostics)

    if decision.type == AgentDecisionType.COMPLETE:
        return StepOutcome.complete(
            run_output or "",
            terminal_reason=TerminalReason.GOAL_MET,
            state_delta=state_delta,
            confidence=decision.confidence,
        )

    if decision.type == AgentDecisionType.REQUEST_HUMAN:
        return StepOutcome.pause_hitl(decision.reason or "human_required", state_delta=state_delta)

    if decision.type == AgentDecisionType.MODIFY_PLAN:
        diagnostics: dict[str, Any] = {"uaep_decision": decision.type.value}
        if decision.handoff is not None:
            diagnostics["handoff"] = decision.handoff.model_dump(mode="json")
        if decision.suggested_plan_delta is not None:
            diagnostics["suggested_plan_delta"] = decision.suggested_plan_delta.model_dump(
                mode="json"
            )
        return StepOutcome.replan(state_delta, diagnostics=diagnostics)

    if decision.type == AgentDecisionType.FAIL:
        return StepOutcome.fail(
            [
                AgentRunError(
                    code=AgentRunErrorCode.INTERNAL_ERROR,
                    message=decision.reason or AgentDecisionType.FAIL.value,
                )
            ],
            terminal_reason=TerminalReason.ERROR,
            state_delta=state_delta,
        )

    if decision.type == AgentDecisionType.CANCEL:
        return StepOutcome.fail(
            [
                AgentRunError(
                    code=AgentRunErrorCode.CANCELLED,
                    message=decision.reason or AgentDecisionType.CANCEL.value,
                )
            ],
            terminal_reason=TerminalReason.CANCELLED,
            state_delta=state_delta,
        )

    if decision.type == AgentDecisionType.RETRY:
        return StepOutcome.continue_with(
            state_delta,
            diagnostics={"uaep_decision": decision.type.value, "retry_reason": decision.reason},
        )

    if decision.type in {AgentDecisionType.INTERRUPT, AgentDecisionType.ESCALATE}:
        return StepOutcome.pause_hitl(
            decision.reason or decision.type.value,
            state_delta=state_delta,
        )

    return StepOutcome.continue_with(
        state_delta,
        diagnostics={"uaep_decision": decision.type.value, "reason": decision.reason},
    )


def build_uaep_step_context(
    step: AgentStep,
    exec_ctx: RuntimeExecutionContext,
    kernel_ctx: StepKernelContext,
) -> AgentStepContext:
    return AgentStepContext(
        step_index=step.step_index,
        run_id=exec_ctx.run_id,
        agent_id=exec_ctx.agent_id,
        contract_id=exec_ctx.contract.id if exec_ctx.contract else exec_ctx.agent_id,
        state_snapshot=dict(kernel_ctx.state_root),
        metadata={
            "task_id": exec_ctx.task_id,
            "graph_node_id": exec_ctx.node_id,
            "step_id": step.step_id,
            "uaep_exec_ctx": exec_ctx,
        },
    )


def decide_after_uaep_step(
    agent: UAEPAgent,
    step: AgentStep,
    output: StepOutput | None,
    ctx: RuntimeExecutionContext,
) -> AgentDecision:
    if isinstance(agent, UAEPAgentWithDecide):
        return agent.decide_after_step(step, output, ctx)
    return AgentDecision(type=AgentDecisionType.CONTINUE)


def kernel_policy_denied_decision(record: StepExecutionRecord) -> AgentDecision:
    code = record.error_code or AgentRunErrorCode.POLICY_DENIED
    if code == AgentRunErrorCode.POLICY_DENIED:
        return AgentDecision(type=AgentDecisionType.FAIL, reason=TerminalReason.POLICY_DENIED.value)
    return AgentDecision(type=AgentDecisionType.FAIL, reason=code.value)


async def execute_uaep_step_via_kernel(
    agent: UAEPAgent,
    step: AgentStep,
    exec_ctx: RuntimeExecutionContext,
    kernel_ctx: StepKernelContext,
) -> StepExecutionResult:
    """Run one UAEP step through HarnessKernel for policy, merge, and Plane B trace."""
    output = await agent.run_step(step, exec_ctx)
    decision = decide_after_uaep_step(agent, step, output, exec_ctx)
    outcome = agent_decision_to_step_outcome(decision, output)
    step_ctx = build_uaep_step_context(step, exec_ctx, kernel_ctx)
    record = await HarnessKernel.execute_step(outcome, step_ctx, kernel_ctx)

    if record.error_code is not None and not record.outcome_applied:
        decision = kernel_policy_denied_decision(record)

    exec_ctx.metadata["uaep_last_kernel_record"] = record.model_dump(mode="json")
    return StepExecutionResult(output=output, decision=decision)


def build_kernel_session(
    *,
    agent_id: str,
    run_id: str,
    task_id: str,
    tenant_id: str,
    max_steps: int | None,
    policy_engine: PolicyEngine,
    request: RuntimeRequest,
) -> StepKernelContext:
    return StepKernelContext(
        agent_id=agent_id,
        run_id=run_id,
        task_id=task_id,
        tenant_id=tenant_id,
        max_steps=max_steps,
        policy_engine=policy_engine,
        state_root=initial_kernel_state_from_request(request),
        run_trace=AgentRunTrace(run_id=run_id),
    )


def trace_summary_from_kernel(kernel_ctx: StepKernelContext) -> dict[str, object]:
    trace = kernel_ctx.run_trace
    return {
        "total_steps": trace.total_steps,
        "total_llm_tokens": trace.total_llm_tokens,
        "total_tool_calls": trace.total_tool_calls,
        "total_rag_calls": trace.total_rag_calls,
        "step_diagnostics": aggregate_step_diagnostics(trace),
        "bridge": "uaep",
    }

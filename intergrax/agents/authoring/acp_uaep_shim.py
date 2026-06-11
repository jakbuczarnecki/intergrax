# © Artur Czarnecki. All rights reserved.

"""UAEP run_step shim for migrated CognitiveAgent classes (ACP-MIG / Wave 8)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from intergrax.agents.authoring.state_merge import extract_acp_state_blob
from intergrax.contracts.acp_metadata_keys import AcpRunContextKey
from intergrax.contracts.acp_state import ACP_STATE_KEY
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.agent_run_enums import StepNextAction, TerminalReason
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.nexus.responses.response_schema import (
    RuntimeAnswer,
    RuntimeRequest,
    RuntimeStats,
)

if TYPE_CHECKING:
    from intergrax.agents.authoring.patterns.base import CognitiveAgent


def _run_input_from_request(exec_ctx: RuntimeExecutionContext) -> str | dict[str, Any]:
    request = exec_ctx.request
    if request is None:
        return ""
    if isinstance(request, RuntimeRequest):
        message = request.message
        if message.strip():
            return message
        metadata = request.metadata
        if metadata.get("message"):
            return str(metadata["message"])
        return message or ""
    metadata = request.metadata
    if metadata.get("message"):
        return str(metadata["message"])
    return ""


def build_step_context_from_uaep(
    agent: CognitiveAgent,
    step: AgentStep,
    exec_ctx: RuntimeExecutionContext,
) -> AgentStepContext:
    state_root: dict[str, Any] = {}
    raw_state = exec_ctx.metadata.get(ACP_STATE_KEY)
    if isinstance(raw_state, dict):
        state_root = {ACP_STATE_KEY: dict(raw_state)}
    return AgentStepContext(
        step_index=step.step_index,
        run_id=exec_ctx.run_id,
        agent_id=exec_ctx.agent_id,
        contract_id=exec_ctx.contract.id if exec_ctx.contract else exec_ctx.agent_id,
        state_snapshot=state_root,
        metadata={
            AcpRunContextKey.RUN_INPUT: _run_input_from_request(exec_ctx),
            "uaep_exec_ctx": exec_ctx,
            "task_id": exec_ctx.task_id,
        },
    )


def step_output_from_outcome(step: AgentStep, outcome: Any) -> StepOutput:
    if isinstance(outcome.output, dict):
        summary = str(outcome.output.get("summary") or outcome.output.get("answer") or "")
        data = {key: value for key, value in outcome.output.items() if key != "summary"}
    else:
        summary = str(outcome.output or "")
        data = {}
    return StepOutput(step_id=step.step_id, summary=summary, data=data)


def agent_decision_from_outcome(outcome: Any) -> AgentDecision:
    if outcome.next_action == StepNextAction.PAUSE_HITL:
        return AgentDecision(
            type=AgentDecisionType.REQUEST_HUMAN,
            reason=outcome.diagnostics.get("pause_reason", "human_required")
            if outcome.diagnostics
            else "human_required",
        )
    if outcome.next_action == StepNextAction.FAIL:
        return AgentDecision(
            type=AgentDecisionType.FAIL,
            reason=outcome.terminal_reason.value if outcome.terminal_reason else "failed",
        )
    if outcome.next_action == StepNextAction.REPLAN:
        return AgentDecision(
            type=AgentDecisionType.MODIFY_PLAN,
            reason=outcome.terminal_reason.value if outcome.terminal_reason else "replan",
        )
    if outcome.is_terminal or outcome.next_action == StepNextAction.CONTINUE:
        if outcome.is_terminal:
            return AgentDecision(
                type=AgentDecisionType.COMPLETE,
                reason=outcome.terminal_reason.value if outcome.terminal_reason else TerminalReason.GOAL_MET.value,
            )
    return AgentDecision(type=AgentDecisionType.CONTINUE, reason="continue")


async def execute_cognitive_step_via_acp(
    agent: CognitiveAgent,
    step: AgentStep,
    exec_ctx: RuntimeExecutionContext,
) -> StepOutput:
    step_ctx = build_step_context_from_uaep(agent, step, exec_ctx)
    outcome = await agent.on_next_step(step_ctx)
    exec_ctx.metadata[AcpRunContextKey.LAST_OUTCOME] = outcome.model_dump(mode="json")
    acp_blob = extract_acp_state_blob(step_ctx.state_snapshot)
    exec_ctx.metadata[ACP_STATE_KEY] = acp_blob
    step_output = step_output_from_outcome(step, outcome)
    exec_ctx.metadata["runtime_answer"] = RuntimeAnswer(
        run_id=exec_ctx.run_id,
        answer=step_output.summary,
        stats=RuntimeStats(total_tokens=0, duration_ms=0, extra={"cost": 0.0}),
    )
    return step_output


def decide_after_cognitive_step(
    exec_ctx: RuntimeExecutionContext,
    *,
    default_reason: str = "cognitive step finished",
) -> AgentDecision:
    raw = exec_ctx.metadata.get(AcpRunContextKey.LAST_OUTCOME)
    if raw is None:
        return AgentDecision(type=AgentDecisionType.COMPLETE, reason=default_reason)
    from intergrax.agents.authoring.step_outcome import StepOutcome

    outcome = StepOutcome.model_validate(raw)
    return agent_decision_from_outcome(outcome)

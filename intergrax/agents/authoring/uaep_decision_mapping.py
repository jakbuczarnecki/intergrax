# © Artur Czarnecki. All rights reserved.

"""Map legacy UAEP ``AgentDecision`` to typed ``StepOutcome`` (agent-owned neutral semantics)."""

from __future__ import annotations

from typing import Any

from intergrax.agents.authoring.step_outcome import StepOutcome
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.agent_run import AgentRunError
from intergrax.contracts.agent_run_enums import AgentRunErrorCode, TerminalReason
from intergrax.contracts.agent_step import StepOutput
from intergrax.contracts.uaep_bridge_keys import UaepStateDeltaKey


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
        return StepOutcome.pause_hitl(
            decision.reason or "human_required", state_delta=state_delta
        )

    if decision.type == AgentDecisionType.MODIFY_PLAN:
        diagnostics = {"uaep_decision": decision.type.value}
        if decision.handoff is not None:
            diagnostics["handoff"] = decision.handoff.model_dump(mode="json")
        if decision.suggested_plan_delta is not None:
            diagnostics["suggested_plan_delta"] = (
                decision.suggested_plan_delta.model_dump(mode="json")
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
            diagnostics={
                "uaep_decision": decision.type.value,
                "retry_reason": decision.reason,
            },
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

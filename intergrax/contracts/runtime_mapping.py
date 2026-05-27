# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Optional

from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.runtime.interrupts.handler import GovernanceResolution
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer


def runtime_answer_to_agent_result(
    answer: RuntimeAnswer,
    *,
    agent_id: str,
    valid: bool = True,
    validation_errors: list[str] | None = None,
    governance: Optional[GovernanceResolution] = None,
) -> AgentExecutionResult:
    """Map RuntimeAnswer to canonical AgentExecutionResult (§14)."""
    errors: list[str] = []
    status = AgentExecutionStatus.COMPLETED if valid else AgentExecutionStatus.FAILED

    if governance is not None:
        if governance.should_pause:
            status = AgentExecutionStatus.NEEDS_INPUT
            errors.append("awaiting human input")
        elif governance.should_fail:
            status = AgentExecutionStatus.FAILED
            if governance.agent_decision.reason:
                errors.append(governance.agent_decision.reason)

    if not valid and status == AgentExecutionStatus.COMPLETED:
        errors = list(validation_errors or ["validation failed"])
        status = AgentExecutionStatus.FAILED

    structured = dict(answer.route.extra if answer.route else {})
    if governance is not None:
        structured["governance"] = governance.model_dump()

    return AgentExecutionResult(
        agent_id=agent_id,
        run_id=answer.run_id or "",
        status=status,
        summary=answer.answer,
        structured_data=structured,
        used_tools=[t.tool_name for t in (answer.tool_calls or [])],
        errors=errors,
        agent_decision=governance.agent_decision if governance else None,
        human_request=governance.human_request if governance else None,
        execution_interrupt=governance.interrupt if governance else None,
        policy_rule_id=governance.policy_decision.policy_rule_id if governance else None,
    )

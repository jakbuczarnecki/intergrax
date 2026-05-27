# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer


def runtime_answer_to_agent_result(
    answer: RuntimeAnswer,
    *,
    agent_id: str,
    valid: bool = True,
    validation_errors: list[str] | None = None,
) -> AgentExecutionResult:
    """Map RuntimeAnswer to canonical AgentExecutionResult (§14)."""
    errors: list[str] = []
    if not valid:
        errors = list(validation_errors or ["validation failed"])
    return AgentExecutionResult(
        agent_id=agent_id,
        run_id=answer.run_id or "",
        status=AgentExecutionStatus.COMPLETED if valid else AgentExecutionStatus.FAILED,
        summary=answer.answer,
        structured_data=dict(answer.route.extra if answer.route else {}),
        used_tools=[t.tool_name for t in (answer.tool_calls or [])],
        errors=errors,
    )

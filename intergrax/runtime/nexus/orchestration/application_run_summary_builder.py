# © Artur Czarnecki. All rights reserved.

"""Build Plane A ApplicationRunSummary from Nexus task executions (ACP-OBS-2)."""

from __future__ import annotations

from typing import Any

from intergrax.contracts.acp_metadata_keys import AcpStructuredDataKey
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.agent_run_enums import AgentRunStatus
from intergrax.contracts.application_run_summary import AgentInvocationSummary, ApplicationRunSummary

_EXECUTION_TO_RUN_STATUS: dict[AgentExecutionStatus, AgentRunStatus] = {
    AgentExecutionStatus.COMPLETED: AgentRunStatus.SUCCEEDED,
    AgentExecutionStatus.FAILED: AgentRunStatus.FAILED,
    AgentExecutionStatus.PARTIAL: AgentRunStatus.SUCCEEDED,
    AgentExecutionStatus.NEEDS_INPUT: AgentRunStatus.PAUSED,
}


def _trace_summary_from_execution(execution: AgentExecutionResult) -> dict[str, Any]:
    raw = execution.structured_data.get(AcpStructuredDataKey.TRACE_SUMMARY)
    if isinstance(raw, dict):
        return raw
    return {}


def build_application_run_summary(
    *,
    task_id: str,
    graph_id: str,
    executions: list[AgentExecutionResult],
    terminal_status: AgentRunStatus | None = None,
) -> ApplicationRunSummary:
    invocations: list[AgentInvocationSummary] = []
    total_steps = 0
    total_llm_tokens = 0

    for execution in executions:
        trace_summary = _trace_summary_from_execution(execution)
        step_count = int(trace_summary.get("total_steps", 0))
        llm_tokens = int(trace_summary.get("total_llm_tokens", 0))
        total_steps += step_count
        total_llm_tokens += llm_tokens
        invocations.append(
            AgentInvocationSummary(
                agent_id=execution.agent_id,
                run_id=execution.run_id,
                status=_EXECUTION_TO_RUN_STATUS.get(
                    execution.status,
                    AgentRunStatus.FAILED,
                ),
                step_count=step_count,
                total_llm_tokens=llm_tokens,
                total_tool_calls=int(trace_summary.get("total_tool_calls", 0)),
                terminal_reason=trace_summary.get("terminal_reason"),
            )
        )

    resolved_terminal = terminal_status
    if resolved_terminal is None:
        if executions and executions[-1].status == AgentExecutionStatus.FAILED:
            resolved_terminal = AgentRunStatus.FAILED
        elif executions and executions[-1].status == AgentExecutionStatus.NEEDS_INPUT:
            resolved_terminal = AgentRunStatus.PAUSED
        else:
            resolved_terminal = AgentRunStatus.SUCCEEDED

    return ApplicationRunSummary(
        task_id=task_id,
        graph_id=graph_id,
        terminal_status=resolved_terminal,
        agent_invocations=invocations,
        total_agents=len(invocations),
        total_steps=total_steps,
        total_llm_tokens=total_llm_tokens,
    )

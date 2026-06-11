# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.contracts.acp_metadata_keys import AcpStructuredDataKey
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.agent_run_enums import AgentRunStatus
from intergrax.runtime.nexus.orchestration.application_run_summary_builder import (
    build_application_run_summary,
)


@pytest.mark.unit
@pytest.mark.gate
def test_build_application_run_summary_multi_agent() -> None:
    executions = [
        AgentExecutionResult(
            agent_id="a",
            run_id="run-a",
            status=AgentExecutionStatus.COMPLETED,
            structured_data={
                AcpStructuredDataKey.TRACE_SUMMARY: {
                    "total_steps": 2,
                    "total_llm_tokens": 30,
                    "total_tool_calls": 0,
                }
            },
        ),
        AgentExecutionResult(
            agent_id="b",
            run_id="run-b",
            status=AgentExecutionStatus.COMPLETED,
            structured_data={
                AcpStructuredDataKey.TRACE_SUMMARY: {
                    "total_steps": 1,
                    "total_llm_tokens": 10,
                    "total_tool_calls": 1,
                }
            },
        ),
    ]
    summary = build_application_run_summary(
        task_id="task-99",
        graph_id="graph-seq",
        executions=executions,
    )
    assert summary.total_agents == 2
    assert summary.total_steps == 3
    assert summary.total_llm_tokens == 40
    assert summary.terminal_status == AgentRunStatus.SUCCEEDED

# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from intergrax.contracts.acp_metadata_keys import AcpStructuredDataKey
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.agent_run_enums import AgentRunStatus
from intergrax.runtime.task.task import TaskResult, TaskState
from intergrax.runtime.task.task_metadata_keys import TaskResultMetadataKey
from local_workspace_application.host.lifecycle import LocalWorkspaceHostLifecycle
from local_workspace_application.host.task_executor import LocalWorkspaceTaskExecutor
from local_workspace_application.serving.fastapi_router import LocalWorkspaceRunService
from local_workspace_application.serving.schemas import LocalWorkspaceRunRequestV1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_task_attaches_lkw_evidence_metadata() -> None:
    execution = AgentExecutionResult(
        agent_id="local_search",
        run_id="run-api",
        status=AgentExecutionStatus.COMPLETED,
        summary="answer",
        structured_data={
            AcpStructuredDataKey.TRACE_SUMMARY: {
                "total_steps": 1,
                "step_diagnostics": {
                    "lkw.search_summary.v1": {
                        "num_results": 1,
                        "evidence_count": 1,
                        "source_refs": ["docs/a.md"],
                    }
                },
            }
        },
    )
    task_result = TaskResult(
        task_id="task-api",
        run_id="run-api",
        state=TaskState.COMPLETED,
        answer="answer",
        agent_id="local_search",
        execution_result=execution,
        metadata={
            TaskResultMetadataKey.APPLICATION_RUN_SUMMARY: {
                "schema_version": "application_run_summary.v1",
                "terminal_status": AgentRunStatus.SUCCEEDED.value,
            }
        },
    )
    lifecycle = LocalWorkspaceHostLifecycle()
    lifecycle.set_executor_available(True)
    lifecycle.transition_to_ready()
    executor = AsyncMock(spec=LocalWorkspaceTaskExecutor)
    executor.execute = AsyncMock(return_value=task_result)
    executor.nexus_loop = None
    service = LocalWorkspaceRunService(task_executor=executor, default_agent_id="local_search")

    response = await service.run_task(
        LocalWorkspaceRunRequestV1(
            message="find docs",
            capability="local.workspace.search",
        )
    )

    assert TaskResultMetadataKey.APPLICATION_RUN_SUMMARY in response.metadata
    evidence = response.metadata["lkw_evidence.v1"]
    assert evidence["schema_version"] == "lkw_evidence.v1"
    assert evidence["capability"] == "local.workspace.search"
    assert "lkw.search_summary.v1" in evidence["diagnostics"]
    assert "full_trace" not in response.metadata

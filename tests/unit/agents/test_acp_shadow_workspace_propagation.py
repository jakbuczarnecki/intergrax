# © Artur Czarnecki. All rights reserved.

"""LKW-PF2A — ACP session propagates shadow_workspace_id into structured_data."""

from __future__ import annotations

import pytest

from intergrax.agents.agent_engine import AgentEngine
from intergrax.agents.authoring.patterns.reflex import ReflexAgent
from intergrax.agents.authoring.runtime_tool_helpers import exec_ctx_from_step, invoke_catalog_tool
from intergrax.contracts.acp_metadata_keys import AcpMetadataKey, AcpStructuredDataKey
from intergrax.contracts.agent_contract_meta import AgentRiskLevel
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.agent_run import AgentRunRequest, RequestIdentity
from intergrax.contracts.agent_run_enums import AgentRunStatus, TerminalReason
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.orchestration.run_artifact_bundle_builder import build_run_artifact_bundle
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.sandbox.manager import SandboxSessionManager
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskIsolationOptions
from intergrax.runtime.workspace.exec_ctx_isolation import (
    RUNTIME_SHADOW_MANAGER_METADATA_KEY,
    isolation_structured_data_from_exec_ctx,
)
from intergrax.runtime.workspace.manager import ShadowWorkspaceManager
from intergrax.runtime.workspace.shadow_workspace import (
    SHADOW_WORKSPACE_FLAG,
    SHADOW_WORKSPACE_ID_KEY,
)
from intergrax.tools.providers.workspace.service import WORKSPACE_WRITE_FILE_TOOL_ID
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager


def _stub_build_context(_agent: ReflexAgent, _request: RuntimeRequest) -> RuntimeContext:
    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(),
        production_mode=False,
        enable_rag=False,
        enable_websearch=False,
    )
    return RuntimeContext.build(
        config=config,
        session_manager=build_in_memory_session_manager(),
    )


class _ShadowWriteReflexAgent(ReflexAgent):
    contract_id = "shadow_acp_reflex"
    capabilities = ("demo.shadow.write",)
    agent_name = "Shadow ACP Reflex"
    agent_description = "Writes one artifact via runtime-bound workspace tool"
    risk_level = AgentRiskLevel.LOW
    main_step_id = "shadow_write_step"

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        return _stub_build_context(self, request)

    async def act(self, step_ctx: AgentStepContext, reasoning) -> dict[str, object]:
        _ = reasoning
        exec_ctx = exec_ctx_from_step(step_ctx)
        assert exec_ctx is not None
        entry = await invoke_catalog_tool(
            exec_ctx,
            tool_name=WORKSPACE_WRITE_FILE_TOOL_ID,
            agent_id=step_ctx.agent_id,
            step_id=self.main_step_id,
            tool_input={
                "path": "synthesis-draft.md",
                "content": "# draft\n",
                "content_type": "text/markdown",
            },
        )
        return {"write_status": entry.get("status"), "artifact_path": entry.get("relative_path")}


@pytest.mark.unit
@pytest.mark.gate
def test_isolation_structured_data_from_exec_ctx_exports_shadow_workspace_id() -> None:
    exec_ctx = RuntimeExecutionContext(
        task_id="task-iso",
        run_id="run-iso",
        agent_id="agent-1",
        metadata={SHADOW_WORKSPACE_ID_KEY: "shadow-ws-test"},
    )

    structured = isolation_structured_data_from_exec_ctx(exec_ctx)

    assert structured[SHADOW_WORKSPACE_ID_KEY] == "shadow-ws-test"


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_acp_run_propagates_shadow_workspace_id(tmp_path) -> None:
    shadow_manager = ShadowWorkspaceManager(root=tmp_path / "shadow")
    agent = _ShadowWriteReflexAgent()
    request = AgentRunRequest(
        input="write draft",
        identity=RequestIdentity(tenant_id="tenant-a", user_id="user-1"),
        metadata={
            AcpMetadataKey.SESSION_ENABLED: True,
            SHADOW_WORKSPACE_FLAG: True,
            RUNTIME_SHADOW_MANAGER_METADATA_KEY: shadow_manager,
            "allowed_tools": [WORKSPACE_WRITE_FILE_TOOL_ID],
        },
    )

    result = await agent.run(request)

    assert result.status == AgentRunStatus.SUCCEEDED
    assert result.structured_data.get(SHADOW_WORKSPACE_ID_KEY)
    assert result.structured_data[AcpStructuredDataKey.TRACE_SUMMARY]["terminal_reason"] == (
        TerminalReason.GOAL_MET.value
    )


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_agent_engine_merges_acp_shadow_workspace_id_into_execution_result(tmp_path) -> None:
    shadow_manager = ShadowWorkspaceManager(root=tmp_path / "shadow")
    agent = _ShadowWriteReflexAgent()
    runtime_request = RuntimeRequest(
        agent_id=agent.contract_id,
        tenant_id="tenant-a",
        user_id="user-1",
        session_id="sess-1",
        message="write draft",
        metadata={
            AcpMetadataKey.SESSION_ENABLED: True,
            SHADOW_WORKSPACE_FLAG: True,
            RUNTIME_SHADOW_MANAGER_METADATA_KEY: shadow_manager,
            "allowed_tools": [WORKSPACE_WRITE_FILE_TOOL_ID],
        },
    )

    execution = await AgentEngine.run_agent_with_result(agent, runtime_request)

    assert execution.structured_data.get(SHADOW_WORKSPACE_ID_KEY)
    workspace_id = str(execution.structured_data[SHADOW_WORKSPACE_ID_KEY])
    assert shadow_manager.get(workspace_id) is not None


@pytest.mark.unit
@pytest.mark.gate
def test_run_artifact_bundle_resolves_workspace_from_execution_structured_data(tmp_path) -> None:
    shadow_manager = ShadowWorkspaceManager(root=tmp_path / "shadow")
    task = Task(
        tenant_id="tenant-a",
        user_id="user-1",
        message="bundle",
        options=TaskExecutionOptions(
            isolation=TaskIsolationOptions(shadow_workspace=True),
        ),
    )
    workspace = shadow_manager.open_or_create(tenant_id=task.tenant_id, task_id=task.task_id)
    workspace.write_text("synthesis-draft.md", "# draft\n")

    executions = [
        AgentExecutionResult(
            agent_id="local_synthesizer",
            run_id=task.task_id,
            status=AgentExecutionStatus.COMPLETED,
            structured_data={SHADOW_WORKSPACE_ID_KEY: workspace.workspace_id},
        )
    ]
    bundle = build_run_artifact_bundle(
        task=task,
        graph_id="graph-1",
        executions=executions,
        shadow_manager=shadow_manager,
        sandbox_manager=SandboxSessionManager(root=tmp_path / "sandbox"),
    )

    assert len(bundle.workspace) == 1
    assert bundle.workspace[0].workspace_id == workspace.workspace_id
    assert bundle.workspace[0].relative_path == "synthesis-draft.md"

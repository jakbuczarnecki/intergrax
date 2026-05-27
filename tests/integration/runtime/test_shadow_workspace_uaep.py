# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.agent_contract import Agent
from intergrax.agents.uaep import UAEPExecutor
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.runtime.workspace.manager import ShadowWorkspaceManager
from intergrax.runtime.workspace.shadow_workspace import (
    SHADOW_WORKSPACE_FLAG,
    SHADOW_WORKSPACE_ID_KEY,
)
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager

pytestmark = pytest.mark.gate


class _ShadowWriteAgent(Agent):
    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="shadow_writer",
            name="Shadow Writer",
            description="writes task message into shadow workspace",
            capabilities=["shadow.basic"],
            max_steps=1,
        )

    def can_handle(self, task_context: object) -> CapabilityMatchResult:
        capability = getattr(task_context, "capability", None)
        if capability in (None, "shadow.basic"):
            return CapabilityMatchResult(
                matched=True,
                agent_id="shadow_writer",
                matched_capabilities=["shadow.basic"],
                score=1.0,
            )
        return CapabilityMatchResult(matched=False, rationale="unsupported")

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        config = RuntimeConfig(
            llm_adapter=FakeLLMAdapter(fixed_text="ok"),
            enable_rag=False,
            production_mode=False,
            tenant_id=request.tenant_id,
        )
        return RuntimeContext.build(
            config=config,
            session_manager=build_in_memory_session_manager(),
        )

    def get_steps(self, context: RuntimeContext) -> list[AgentStep]:
        _ = context
        return [AgentStep(step_id="write", step_name="write", step_index=0)]

    async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
        workspace = ctx.metadata.get("shadow_workspace")
        message = (ctx.request.message if ctx.request else "") or ""
        if workspace is not None:
            workspace.write_text("output.txt", message)
        return StepOutput(step_id=step.step_id, summary=f"shadow: {message}")

    def decide_after_step(
        self,
        step: AgentStep,
        output: StepOutput | None,
        ctx: RuntimeExecutionContext,
    ) -> AgentDecision:
        _ = step, output, ctx
        return AgentDecision(type=AgentDecisionType.COMPLETE, reason="done")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_uaep_attaches_shadow_workspace(tmp_path):
    manager = ShadowWorkspaceManager(root=tmp_path)
    uaep = UAEPExecutor(shadow_manager=manager)
    agent = _ShadowWriteAgent()
    request = RuntimeRequest(
        tenant_id="t1",
        user_id="u1",
        session_id="s1",
        agent_id="shadow_writer",
        message="experiment artifact",
        metadata={
            SHADOW_WORKSPACE_FLAG: True,
            "task_id": "task_shadow_1",
        },
    )

    answer, validation, _context, _governance = await uaep.execute(agent, request)

    assert validation.valid is True
    assert answer.route is not None
    workspace_id = answer.route.extra.get(SHADOW_WORKSPACE_ID_KEY)
    assert workspace_id
    workspace = manager.get(str(workspace_id))
    assert workspace is not None
    assert workspace.read_text("output.txt") == "experiment artifact"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_nexus_loop_exposes_shadow_workspace_metadata(tmp_path):
    manager = ShadowWorkspaceManager(root=tmp_path)
    registry = AgentRegistry()
    registry.register(_ShadowWriteAgent())
    loop = NexusLoop(registry, shadow_manager=manager)

    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="via nexus",
        context=TaskContext(capability="shadow.basic"),
        metadata={SHADOW_WORKSPACE_FLAG: True},
    )

    result = await loop.handle_task(task)

    assert result.state == TaskState.COMPLETED
    assert result.metadata.get("shadow_workspace_id")
    workspace = manager.get(str(result.metadata["shadow_workspace_id"]))
    assert workspace is not None
    assert workspace.read_text("output.txt") == "via nexus"

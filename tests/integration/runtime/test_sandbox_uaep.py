# © Artur Czarnecki. All rights reserved.

from intergrax.utils import attribute_access
import pytest

from intergrax.agents.agent_contract import Agent
from intergrax.agents.uaep import UAEPExecutor
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.contracts.tool_request import ToolRequest, ToolResponseStatus
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.tools.registry.profile import ToolProfile
from intergrax.tools.registry.wiring import ToolWiringContext
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.applications._shared.policy_wiring import wire_policy_bundle
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    PolicyRulesProfile,
)
from testing_support.builder import (
    build_runtime_request_for_tests,
    canonical_governed_execution_scope,
)
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.sandbox.manager import SandboxSessionManager
from intergrax.runtime.sandbox.sandbox_runtime import (
    SANDBOX_FLAG,
    SANDBOX_SESSION_ID_KEY,
    SANDBOX_TOOL_NAME,
)
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from testing_support.builder import (
    FakeLLMAdapter,
    build_in_memory_session_manager,
)

pytestmark = pytest.mark.gate


class _SandboxToolAgent(Agent):
    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="sandbox_runner",
            name="Sandbox Runner",
            description="writes via sandbox.exec tool",
            capabilities=["sandbox.basic"],
            allowed_tools=[SANDBOX_TOOL_NAME],
            max_steps=1,
        )

    def can_handle(self, task_context: object) -> CapabilityMatchResult:
        capability = attribute_access.optional(task_context, "capability", None)
        if capability in (None, "sandbox.basic"):
            return CapabilityMatchResult(
                matched=True,
                agent_id="sandbox_runner",
                matched_capabilities=["sandbox.basic"],
                score=1.0,
            )
        return CapabilityMatchResult(matched=False, rationale="unsupported")

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        policy_env = ApplicationEnvironmentProfile.lab_defaults(
            profile_id="sandbox.uaep.integration",
        )
        policy_env.policy_rules = PolicyRulesProfile(
            inline_rules=[],
            policy_enforcement_mode="enforce",
        )
        wiring_ctx = ToolWiringContext(
            extras={"effective_environment_profile": policy_env},
        )
        config = RuntimeConfig(
            llm_adapter=FakeLLMAdapter(fixed_text="ok"),
            enable_rag=False,
            production_mode=False,
            tenant_id=request.tenant_id,
            tool_profile=ToolProfile(enabled=[SANDBOX_TOOL_NAME]),
            tool_wiring_context=wiring_ctx,
            policy_bundle=wire_policy_bundle(policy_env),
        )
        return RuntimeContext.build(
            config=config,
            session_manager=build_in_memory_session_manager(),
        )

    def get_steps(self, context: RuntimeContext) -> list[AgentStep]:
        _ = context
        return [AgentStep(step_id="sandbox_write", step_name="sandbox_write", step_index=0)]

    async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
        message = (ctx.request.message if ctx.request else "") or ""
        response = await ctx.invoke_tool(
            ToolRequest(
                tool_name=SANDBOX_TOOL_NAME,
                agent_id=ctx.agent_id,
                step_id=step.step_id,
                input={
                    "operation": "write_file",
                    "payload": {"path": "tool_output.txt", "content": message},
                },
            )
        )
        assert response.status == ToolResponseStatus.SUCCESS
        return StepOutput(step_id=step.step_id, summary=f"sandbox: {message}")

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
async def test_uaep_sandbox_tool_gateway(tmp_path):
    manager = SandboxSessionManager(root=tmp_path)
    uaep = UAEPExecutor(sandbox_manager=manager)
    agent = _SandboxToolAgent()
    with canonical_governed_execution_scope("sandbox-uaep"):
        request = build_runtime_request_for_tests(
            seed="sandbox-uaep",
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            agent_id="sandbox_runner",
            message="via sandbox tool",
            metadata={SANDBOX_FLAG: True},
        )
        answer, validation, _governance = await uaep.execute(agent, request)

    assert validation.valid is True
    assert answer.route is not None
    session_id = answer.route.extra.get(SANDBOX_SESSION_ID_KEY)
    assert session_id
    session = manager.get(str(session_id))
    assert session is not None
    read = session.execute("read_file", {"path": "tool_output.txt"})
    assert read.output["content"] == "via sandbox tool"
    assert answer.route.extra.get("sandbox_operation_count", 0) >= 1


@pytest.mark.asyncio
@pytest.mark.integration
async def test_nexus_loop_exposes_sandbox_session_metadata(tmp_path):
    manager = SandboxSessionManager(root=tmp_path)
    registry = AgentRegistry()
    registry.register(_SandboxToolAgent())
    loop = NexusLoop(registry, sandbox_manager=manager)

    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="via nexus sandbox",
        context=TaskContext(capability="sandbox.basic"),
        metadata={SANDBOX_FLAG: True},
    )
    seed = "nexus-sandbox-metadata"
    from intergrax.runtime.governance.active_execution_authority import (
        ParentExecutionAuthority,
        bind_active_execution_authority,
        reset_active_execution_authority,
    )

    with canonical_governed_execution_scope(seed) as run_id:
        authority_token = bind_active_execution_authority(
            ParentExecutionAuthority.unrestricted_root(),
        )
        try:
            result = await loop.handle_task(task, run_id=run_id)
        finally:
            reset_active_execution_authority(authority_token)

    assert result.state == TaskState.COMPLETED
    assert result.metadata.get("sandbox_session_id")
    session = manager.get(str(result.metadata["sandbox_session_id"]))
    assert session is not None
    read = session.execute("read_file", {"path": "tool_output.txt"})
    assert read.output["content"] == "via nexus sandbox"

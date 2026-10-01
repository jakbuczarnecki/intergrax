# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.authoring.acp_runtime_session_ports import AcpRuntimeSessionHooks
from intergrax.agents.authoring.acp_session_host import ACP_HOST_CONTEXT_KEY, ACPSessionHostContext
from intergrax.agents.authoring.base import IntergraxAgent
from intergrax.agents.authoring.shared_context_access import neutral_shared_context_access_for_run
from intergrax.agents.authoring.decorators import step
from intergrax.agents.authoring.step_outcome import StepOutcome
from intergrax.contracts.agent_contract_meta import AgentRiskLevel
from intergrax.contracts.agent_run import AgentRunRequest, RequestIdentity
from intergrax.contracts.agent_run_enums import AgentRunStatus, PrincipalType, TerminalReason
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.nexus.agents.acp_uaep_shim import (
    attach_acp_catalog_exec_ctx,
    close_acp_catalog_exec_ctx,
)
from testing_support.builder import (
    FakeLLMAdapter,
    build_in_memory_session_manager,
    canonical_governed_execution_scope,
    canonical_run_id_for_tests,
)
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest


def _stub_build_context(_agent: IntergraxAgent, _request: RuntimeRequest) -> RuntimeContext:
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


class _CounterAgent(IntergraxAgent):
    contract_id = "counter"
    capabilities = ("demo.counter",)
    agent_name = "Counter"
    agent_description = "Counts steps"
    risk_level = AgentRiskLevel.LOW
    max_steps = 5

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        return _stub_build_context(self, request)

    async def on_next_step(self, step_ctx: AgentStepContext) -> StepOutcome:
        if step_ctx.step_index >= 2:
            return StepOutcome.complete(
                output={"steps": step_ctx.step_index + 1},
                terminal_reason=TerminalReason.GOAL_MET,
            )
        return StepOutcome.continue_with(state_delta={"phase": f"p{step_ctx.step_index}"})


class _StepBridgeAgent(IntergraxAgent):
    contract_id = "bridge"
    capabilities = ("demo.bridge",)
    agent_name = "Bridge"
    agent_description = "Uses @step bridge"
    max_steps = 3

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        return _stub_build_context(self, request)

    @step("one")
    async def one(self, ctx: RuntimeExecutionContext) -> dict[str, str]:
        _ = ctx
        return {"summary": "one"}

    @step("two")
    async def two(self, ctx: RuntimeExecutionContext) -> dict[str, str]:
        _ = ctx
        return {"summary": "two"}


@pytest.mark.unit
@pytest.mark.gate
async def test_intergrax_agent_run_agent_run_request() -> None:
    agent = _CounterAgent()
    run_id = canonical_run_id_for_tests("direct-agent-run")
    request = AgentRunRequest(
        input="hello",
        identity=RequestIdentity(
            tenant_id="tenant-a",
            user_id="user-1",
            principal_type=PrincipalType.USER,
        ),
        metadata={"run_id": run_id},
    )
    with canonical_governed_execution_scope(
        "direct-agent-run",
        governance_tenant_id="tenant-a",
        governance_principal_id="user-1",
    ):
        result = await agent.run(request)
    assert result.status == AgentRunStatus.SUCCEEDED
    assert result.terminal_reason == TerminalReason.GOAL_MET
    assert result.output == {"steps": 3}
    assert result.run_id == run_id
    assert result.trace.steps


@pytest.mark.unit
@pytest.mark.gate
async def test_default_on_next_step_drives_authored_steps_with_exec_ctx() -> None:
    agent = _StepBridgeAgent()
    run_id = canonical_run_id_for_tests("step-bridge")
    host_ctx = ACPSessionHostContext(
        runtime_session_hooks=AcpRuntimeSessionHooks(
            attach_acp_catalog_exec_ctx=attach_acp_catalog_exec_ctx,
            close_acp_catalog_exec_ctx=close_acp_catalog_exec_ctx,
            resolve_shared_context_access=neutral_shared_context_access_for_run,
        ),
    )
    request = AgentRunRequest(
        input="go",
        identity=RequestIdentity(tenant_id="t", user_id="u"),
        metadata={
            "run_id": run_id,
            ACP_HOST_CONTEXT_KEY: host_ctx,
        },
    )
    with canonical_governed_execution_scope(
        "step-bridge",
        governance_tenant_id="t",
        governance_principal_id="u",
    ):
        result = await agent.run(request)
    assert result.status == AgentRunStatus.SUCCEEDED
    assert "summary" in str(result.output)

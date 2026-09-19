# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-1R2: Task Memory trusts tenant scope only from verified canonical identity."""

from __future__ import annotations

from dataclasses import replace

import pytest

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_run import AgentRunRequest, AgentRunResult
from intergrax.agents.uaep import UAEPExecutor
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.task_memory import InMemoryTaskMemoryStore
from intergrax.tools.registry.wiring import ToolWiringContext
from intergrax.memory.contracts.memory_control import (
    MemoryControlPlaneScope,
    MemoryControlRecallRequest,
    MemoryControlRecallResult,
    MemoryControlScopeRef,
)
from testing_support.memory_control_plane_test_stub import MemoryControlPlaneTestStub
from testing_support.builder import (
    FakeLLMAdapter,
    build_in_memory_session_manager,
    build_runtime_request_for_tests,
    canonical_execution_identity_scope,
)

pytestmark = pytest.mark.gate

_CANONICAL_TENANT = "tenant-a"


class _EmptyRecallMemoryPlane(MemoryControlPlaneTestStub):
    async def recall(
        self,
        identity: RequestIdentity,
        scope: MemoryControlScopeRef,
        request: MemoryControlRecallRequest,
    ) -> MemoryControlRecallResult:
        _ = identity, scope, request
        return MemoryControlRecallResult(
            scope=MemoryControlPlaneScope.USER,
            items=(),
            reason="no_hits",
        )


def _canonical_identity(*, tenant_id: str = _CANONICAL_TENANT) -> RequestIdentity:
    return RequestIdentity(
        tenant_id=tenant_id,
        user_id="verified-user",
        principal_type=PrincipalType.USER,
        auth_subject="verified-user",
    )


class _MemoryProbeAgent(Agent):
    def __init__(self, *, expect_memory: bool) -> None:
        self._expect_memory = expect_memory

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="probe",
            name="Probe",
            description="memory probe",
            capabilities=["probe"],
            max_steps=1,
        )

    async def run(self, request: AgentRunRequest) -> AgentRunResult:
        _ = request
        raise NotImplementedError("UAEP probe executes via UAEPExecutor")

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        wiring = ToolWiringContext()
        wiring.extras["memory_control_plane"] = _EmptyRecallMemoryPlane()
        config = RuntimeConfig(
            llm_adapter=FakeLLMAdapter(fixed_text="ok"),
            enable_rag=False,
            production_mode=False,
            tenant_id=request.tenant_id,
            tool_wiring_context=wiring,
        )
        return RuntimeContext.build(
            config=config,
            session_manager=build_in_memory_session_manager(),
        )

    def get_steps(self) -> list[AgentStep]:
        return [AgentStep(step_id="s1", step_name="s1", step_index=0)]

    async def run_step(
        self, step: AgentStep, ctx: RuntimeExecutionContext
    ) -> StepOutput:
        _ = step
        if self._expect_memory:
            assert ctx.memory_view is not None
            assert ctx.canonical_request_identity is not None
            assert ctx.canonical_request_identity.tenant_id == _CANONICAL_TENANT
            await ctx.memory_view.write("ns", "k", {"proof": True})
            return StepOutput(step_id=step.step_id, summary="trusted-memory")
        assert ctx.memory_view is None
        assert ctx.canonical_request_identity is None
        return StepOutput(step_id=step.step_id, summary="no-trusted-memory")

    def decide_after_step(
        self,
        step: AgentStep,
        output: StepOutput | None,
        ctx: RuntimeExecutionContext,
    ) -> AgentDecision:
        _ = step, output, ctx
        return AgentDecision(type=AgentDecisionType.COMPLETE, reason="done")


@pytest.mark.asyncio
async def test_execute_without_canonical_identity_leaves_memory_view_unavailable() -> None:
    request = replace(
        build_runtime_request_for_tests(
            tenant_id="attacker-tenant",
            agent_id="probe",
            seed="mem-ent-1r2-no-canonical",
        ),
        canonical_identity=None,
    )
    executor = UAEPExecutor(task_memory_store=InMemoryTaskMemoryStore())
    with canonical_execution_identity_scope(str(request.run_id)):
        answer, validation, _ = await executor.execute(
            _MemoryProbeAgent(expect_memory=False),
            request,
        )
    assert validation.valid
    assert answer.answer == "no-trusted-memory"


@pytest.mark.asyncio
async def test_execute_with_canonical_identity_enables_task_memory() -> None:
    request = replace(
        build_runtime_request_for_tests(
            tenant_id=_CANONICAL_TENANT,
            user_id="verified-user",
            agent_id="probe",
            seed="mem-ent-1r2-canonical",
        ),
        canonical_identity=_canonical_identity(),
    )
    store = InMemoryTaskMemoryStore()
    executor = UAEPExecutor(task_memory_store=store)
    with canonical_execution_identity_scope(str(request.run_id)):
        answer, validation, _ = await executor.execute(
            _MemoryProbeAgent(expect_memory=True),
            request,
        )
    assert validation.valid
    assert answer.answer == "trusted-memory"
    persisted = store.get(
        tenant_id=_CANONICAL_TENANT,
        task_id=str(request.task_id),
        namespace="ns",
        key="k",
    )
    assert persisted is not None
    assert persisted.value["proof"] is True

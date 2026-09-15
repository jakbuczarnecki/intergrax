# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-1R2: Memory trusts tenant scope only from verified canonical identity."""

from __future__ import annotations

from dataclasses import replace

import pytest

from intergrax.agents.uaep import UAEPExecutor
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.task_memory import InMemoryTaskMemoryStore
from testing_support.builder import (
    build_runtime_execution_context_for_tests,
    build_runtime_request_for_tests,
    canonical_execution_identity_scope,
)

pytestmark = pytest.mark.gate

_CANONICAL_TENANT = "tenant-a"
_CONFLICT_TENANT = "tenant-b"


def _canonical_identity(*, tenant_id: str = _CANONICAL_TENANT) -> RequestIdentity:
    return RequestIdentity(
        tenant_id=tenant_id,
        user_id="verified-user",
        principal_type=PrincipalType.USER,
        auth_subject="verified-user",
    )


def test_canonical_identity_resolves_for_execute() -> None:
    request = replace(
        build_runtime_request_for_tests(tenant_id=_CANONICAL_TENANT),
        canonical_identity=_canonical_identity(),
    )
    resolved = UAEPExecutor._canonical_request_identity_for_execute(request)
    assert resolved is not None
    assert resolved.tenant_id == _CANONICAL_TENANT


def test_metadata_tenant_conflict_rejected_at_execute_boundary() -> None:
    request = replace(
        build_runtime_request_for_tests(tenant_id=_CANONICAL_TENANT),
        canonical_identity=_canonical_identity(),
        metadata={"tenant_id": _CONFLICT_TENANT},
    )
    with pytest.raises(ValueError, match="metadata tenant_id conflicts"):
        UAEPExecutor._canonical_request_identity_for_execute(request)


def test_request_tenant_conflict_rejected_at_execute_boundary() -> None:
    request = replace(
        build_runtime_request_for_tests(tenant_id=_CONFLICT_TENANT),
        canonical_identity=_canonical_identity(tenant_id=_CANONICAL_TENANT),
    )
    with pytest.raises(ValueError, match="request tenant_id conflicts"):
        UAEPExecutor._canonical_request_identity_for_execute(request)


def test_missing_canonical_identity_does_not_synthesize_trusted_identity() -> None:
    request = replace(
        build_runtime_request_for_tests(tenant_id="attacker-tenant"),
        canonical_identity=None,
        metadata={"tenant_id": "tenant-x"},
    )
    assert UAEPExecutor._canonical_request_identity_for_execute(request) is None


def test_legacy_metadata_cannot_become_canonical_for_memory() -> None:
    request = replace(
        build_runtime_request_for_tests(tenant_id="attacker-tenant"),
        canonical_identity=None,
        metadata={"tenant_id": "tenant-x"},
    )
    exec_ctx = build_runtime_execution_context_for_tests(
        tenant_id="tenant-x",
        metadata=dict(request.metadata),
    )
    exec_ctx.canonical_request_identity = UAEPExecutor._canonical_request_identity_for_execute(
        request
    )
    executor = UAEPExecutor(task_memory_store=InMemoryTaskMemoryStore())
    executor._attach_memory_view(exec_ctx, request)
    assert exec_ctx.memory_view is None


@pytest.mark.asyncio
async def test_canonical_identity_enables_memory_view_attachment() -> None:
    request = replace(
        build_runtime_request_for_tests(
            tenant_id=_CANONICAL_TENANT,
            seed="mem-ent-1r2-canonical",
        ),
        canonical_identity=_canonical_identity(),
    )
    template_ctx = build_runtime_execution_context_for_tests(seed="mem-ent-1r2-canonical")
    exec_ctx = RuntimeExecutionContext(
        task_id=request.task_id,
        run_id=request.run_id,
        attempt_id=template_ctx.attempt_id,
        execution_id=template_ctx.execution_id,
        agent_id=request.agent_id,
        request=request,
        canonical_request_identity=UAEPExecutor._canonical_request_identity_for_execute(request),
    )
    store = InMemoryTaskMemoryStore()
    executor = UAEPExecutor(task_memory_store=store)
    executor._attach_memory_view(exec_ctx, request)
    assert exec_ctx.memory_view is not None
    await exec_ctx.memory_view.write("ns", "k", {"proof": True})
    persisted = store.get(
        tenant_id=_CANONICAL_TENANT,
        task_id=str(request.task_id),
        namespace="ns",
        key="k",
    )
    assert persisted is not None
    assert persisted.value["proof"] is True


@pytest.mark.asyncio
async def test_execute_without_canonical_identity_leaves_memory_view_unavailable() -> None:
    from intergrax.agents.agent_contract import Agent
    from intergrax.contracts.agent_contract_meta import AgentContract
    from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
    from intergrax.contracts.agent_step import AgentStep, StepOutput
    from intergrax.runtime.nexus.config import RuntimeConfig
    from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
    from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager

    class _ProbeAgent(Agent):
        def get_contract(self) -> AgentContract:
            return AgentContract(
                id="probe",
                name="Probe",
                description="memory probe",
                capabilities=["probe"],
                max_steps=1,
            )

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
            return [AgentStep(step_id="s1", step_name="s1", step_index=0)]

        async def run_step(
            self, step: AgentStep, ctx: RuntimeExecutionContext
        ) -> StepOutput:
            _ = step
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
        answer, validation, _ = await executor.execute(_ProbeAgent(), request)
    assert validation.valid
    assert answer.answer == "no-trusted-memory"

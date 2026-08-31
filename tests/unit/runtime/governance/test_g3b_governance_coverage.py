# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from intergrax.agents.agent_contract import Agent
from intergrax.agents.authoring.llm_router import StepLLMRouter
from intergrax.agents.uaep import UAEPExecutor
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.contracts.runtime_policy_context import PreModelPhase, PreModelPolicyContext
from intergrax.contracts.validation import ValidationResult
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.governance.post_run_governance_bridge import invoke_post_run_governance
from intergrax.runtime.interrupts.handler import ExecutionInterruptHandler, GovernanceResolution
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.policy.agent_decision_enforcement import (
    agent_decision_failure_from_resolution,
)
from intergrax.runtime.policy.policy_engine import PolicyEngine
from intergrax.runtime.policy.pre_model_policy_bridge import (
    PreModelPolicyBlockedError,
    wrap_policy_enforcing_llm_router,
)
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.task_trace import TaskTraceEmitter
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager


class _DenyContinuePolicyEngine(RuntimePolicyEngine):
    def evaluate_decision(self, decision, *, context=None):
        if decision.type is AgentDecisionType.CONTINUE:
            return PolicyDecision(
                action=PolicyAction.DENY,
                reason="test_deny_continue",
                policy_rule_id="test.deny_continue",
            )
        return super().evaluate_decision(decision, context=context)


class _RecordingGovernanceService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def evaluate(self, run_id: str, agent_id: str):
        self.calls.append((run_id, agent_id))
        return None


@pytest.mark.unit
def test_governance_resolution_should_block_execution_on_deny() -> None:
    resolution = GovernanceResolution(
        policy_decision=PolicyDecision(
            action=PolicyAction.DENY,
            reason="denied",
            policy_rule_id="test.deny",
        ),
        agent_decision=AgentDecision(type=AgentDecisionType.CONTINUE, reason="ok"),
    )
    assert resolution.should_block_execution is True
    assert resolution.should_pause is False


@pytest.mark.unit
def test_agent_decision_enforcement_maps_deny_to_fail() -> None:
    resolution = GovernanceResolution(
        policy_decision=PolicyDecision(
            action=PolicyAction.DENY,
            reason="denied",
            policy_rule_id="test.deny",
        ),
        agent_decision=AgentDecision(type=AgentDecisionType.CONTINUE, reason="ok"),
    )
    failed = agent_decision_failure_from_resolution(resolution)
    assert failed.type is AgentDecisionType.FAIL


@pytest.mark.unit
def test_interrupt_handler_deny_blocks_execution() -> None:
    handler = ExecutionInterruptHandler(_DenyContinuePolicyEngine())
    resolution = handler.resolve_decision(
        AgentDecision(type=AgentDecisionType.CONTINUE, reason="ok"),
        task_id="task_1",
        run_id="run_1",
        agent_id="agent_1",
        step_id="step_1",
    )
    assert resolution.should_block_execution is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pre_model_policy_blocks_provider_before_complete() -> None:
    called = {"value": False}

    class _Port:
        async def complete(self, prompt: str, *, model_id: str, provider: str):
            called["value"] = True
            return "ok", 1, 1

    inner = StepLLMRouter(
        allowed_models=("balanced",),
        default_model="balanced",
        llm_port=_Port(),
    )
    router = wrap_policy_enforcing_llm_router(
        inner,
        policy_engine=PolicyEngine(),
        tenant_id="tenant_1",
        agent_id="agent_1",
    )

    class _DenyAgentModelEngine(RuntimePolicyEngine):
        def evaluate_pre_llm(self, *, tenant_id, agent_id, message_count, context=None):
            ctx = context or PreModelPolicyContext()
            if ctx.phase is PreModelPhase.AGENT_STEP and ctx.model_id == "balanced":
                return PolicyDecision(
                    action=PolicyAction.DENY,
                    reason="agent_model_denied",
                    policy_rule_id="test.agent_model_denied",
                )
            return super().evaluate_pre_llm(
                tenant_id=tenant_id,
                agent_id=agent_id,
                message_count=message_count,
                context=context,
            )

    router._policy_engine = PolicyEngine(runtime=_DenyAgentModelEngine())  # noqa: SLF001

    with pytest.raises(PreModelPolicyBlockedError):
        await router.complete("hello")

    assert called["value"] is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pre_model_policy_allows_provider_on_allow() -> None:
    called = {"value": False}

    class _Port:
        async def complete(self, prompt: str, *, model_id: str, provider: str):
            called["value"] = True
            return "ok", 1, 1

    inner = StepLLMRouter(
        allowed_models=("balanced",),
        default_model="balanced",
        llm_port=_Port(),
    )
    router = wrap_policy_enforcing_llm_router(
        inner,
        policy_engine=PolicyEngine(),
        tenant_id="tenant_1",
        agent_id="agent_1",
    )
    result = await router.complete("hello")
    assert called["value"] is True
    assert result.text == "ok"


@pytest.mark.unit
def test_post_run_governance_bridge_invokes_service() -> None:
    service = _RecordingGovernanceService()
    invoke_post_run_governance(service, run_id="run_1", agent_id="agent_1")
    assert service.calls == [("run_1", "agent_1")]


class _UaepDenyBoundaryAgent(Agent):
    def __init__(self) -> None:
        self.protected_called = False

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="uaep-deny-boundary",
            name="UAEP Deny Boundary",
            description="two-step agent for policy deny boundary proof",
            capabilities=["stub.basic"],
            max_steps=3,
        )

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        config = RuntimeConfig(
            llm_adapter=FakeLLMAdapter(fixed_text="stub-ok"),
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
        return [
            AgentStep(step_id="s1", step_name="first", step_index=0),
            AgentStep(step_id="s2", step_name="second", step_index=1),
        ]

    async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
        _ = ctx
        if step.step_id == "s2":
            self.protected_called = True
        return StepOutput(step_id=step.step_id, summary=f"out:{step.step_name}")

    def decide_after_step(
        self,
        step: AgentStep,
        output: StepOutput | None,
        ctx: RuntimeExecutionContext,
    ) -> AgentDecision:
        _ = output, ctx
        if step.step_id == "s2":
            return AgentDecision(type=AgentDecisionType.COMPLETE, reason="done")
        return AgentDecision(type=AgentDecisionType.CONTINUE, reason="continue")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_nexus_finish_task_post_run_uses_active_run_id_not_task_id() -> None:
    service = _RecordingGovernanceService()
    loop = NexusLoop(AgentRegistry(), governance_service=service)
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    task_id = mint_task_id()
    assert run_id != task_id

    task = Task(
        task_id=task_id,
        tenant_id="tenant_1",
        user_id="user_1",
        agent_id="agent_exec_1",
        message="done",
    )
    trace_emitter = TaskTraceEmitter(run_id=run_id, attempt_id=attempt_id)
    token = bind_active_execution_identity(run_id=run_id, attempt_id=attempt_id)
    try:
        await loop._finish_task(  # noqa: SLF001
            task,
            trace_emitter,
            answer="ok",
            executions=[],
            validation=ValidationResult(valid=True),
            plan=None,
            retry_records=[],
            graph_id="graph_1",
        )
    finally:
        reset_active_execution_identity(token)

    assert len(service.calls) == 1
    received_run_id, received_agent_id = service.calls[0]
    assert received_run_id == run_id
    assert received_run_id != task.task_id
    assert received_agent_id == "agent_exec_1"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_uaep_agent_decision_deny_blocks_subsequent_protected_step() -> None:
    agent = _UaepDenyBoundaryAgent()
    bus = RuntimeEventBus()
    executor = UAEPExecutor(
        event_bus=bus,
        policy_engine=PolicyEngine(runtime=_DenyContinuePolicyEngine()),
    )
    run_id = mint_run_id()
    task_id = mint_task_id()
    attempt_id = mint_attempt_id()
    request = RuntimeRequest(
        tenant_id="t1",
        user_id="u1",
        session_id="s1",
        agent_id="uaep-deny-boundary",
        message="hi",
        task_id=task_id,
        run_id=run_id,
    )
    token = bind_active_execution_identity(run_id=run_id, attempt_id=attempt_id)
    try:
        _answer, _validation, governance = await executor.execute(agent, request)
    finally:
        reset_active_execution_identity(token)

    assert governance is not None
    assert governance.policy_decision.action is PolicyAction.DENY
    assert governance.should_block_execution is True
    assert agent.protected_called is False
    step_events = [
        event
        for event in bus.history
        if event.event_type is RuntimeEventType.STEP_STARTED
    ]
    assert len(step_events) == 1

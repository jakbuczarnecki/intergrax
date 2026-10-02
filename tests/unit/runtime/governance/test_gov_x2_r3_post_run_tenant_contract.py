# © Artur Czarnecki. All rights reserved.

"""GOV-X2-R3 post-run governance tenant contract regressions."""

from __future__ import annotations

import pytest

from intergrax.agents.harness_reference_agent import HarnessReferenceAgent
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.contracts.validation import ValidationResult
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.governance.execution_guard import ExecutionGuard, GovernanceEvaluation
from intergrax.runtime.governance.post_run_governance_bridge import (
    PostRunGovernanceService,
    invoke_post_run_governance,
)
from intergrax.runtime.governance.service import GovernanceService
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.uaep import UAEPExecutor
from intergrax.runtime.policy.policy_engine import PolicyEngine
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.replay.metrics import ExecutionMetrics
from intergrax.runtime.replay.models import ReconstructedRun
from intergrax.runtime.replay.policy import ExecutionPolicyEngine, PolicyDecisionType
from intergrax.runtime.replay.policy_config import ExecutionPolicyConfig
from intergrax.runtime.replay.regression import RegressionSignals
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.task_trace import TaskTraceEmitter
from intergrax.runtime.wiring.harness_governance import LabAllowGovernanceService
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager


class _RecordingGovernanceService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def evaluate(
        self,
        tenant_id: str,
        run_id: str,
        agent_id: str,
    ) -> GovernanceEvaluation | None:
        self.calls.append((tenant_id, run_id, agent_id))
        return None


class _RecordingReplay:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def inspect_run(self, tenant_id: str, run_id: str) -> ReconstructedRun:
        self.calls.append((tenant_id, run_id))
        return ReconstructedRun(
            run_id=run_id,
            steps=[],
            artifacts=[],
            tool_calls=[],
            llm_calls=[],
            final_answer=None,
        )


class _CrossTenantReplay:
    def inspect_run(self, tenant_id: str, run_id: str) -> ReconstructedRun:
        if tenant_id == "tenant_A" and run_id == "run_owned_by_B":
            raise ValueError("cross-tenant replay denied")
        return ReconstructedRun(
            run_id=run_id,
            steps=[],
            artifacts=[],
            tool_calls=[],
            llm_calls=[],
            final_answer=None,
        )


class _StaticMetricsEngine:
    def compute(self, reconstructed: ReconstructedRun) -> ExecutionMetrics:
        _ = reconstructed
        return ExecutionMetrics(
            step_count=0,
            total_llm_calls=0,
            total_tool_calls=0,
            total_artifacts=0,
            total_tokens=0,
            duration=None,
            tool_steps_ratio=0.0,
            llm_steps_ratio=0.0,
        )


class _EmptyHistoryEvaluator:
    def evaluate(
        self,
        current: ExecutionMetrics,
        previous_runs: list[ExecutionMetrics],
    ) -> RegressionSignals:
        _ = current, previous_runs
        return RegressionSignals()



class _EmptyMetricsStore:
    def get_recent(self, agent_id: str, limit: int) -> list[object]:
        _ = agent_id, limit
        return []

    def save(self, record: object) -> None:
        _ = record


def _execution_guard_with_replay(replay: _RecordingReplay | _CrossTenantReplay) -> ExecutionGuard:
    return ExecutionGuard(
        replay_service=replay,
        metrics_engine=_StaticMetricsEngine(),
        history_evaluator=_EmptyHistoryEvaluator(),
        policy_engine=ExecutionPolicyEngine(ExecutionPolicyConfig()),
        metrics_store=_EmptyMetricsStore(),
        actions=[],
    )


def _accept_post_run(service: PostRunGovernanceService) -> None:
    _ = service


@pytest.mark.unit
def test_gov_x2_r3_01_bridge_propagates_tenant_run_agent() -> None:
    service = _RecordingGovernanceService()
    invoke_post_run_governance(
        service,
        tenant_id="tenant_1",
        run_id="run_1",
        agent_id="agent_1",
    )
    assert service.calls == [("tenant_1", "run_1", "agent_1")]


@pytest.mark.unit
def test_gov_x2_r3_04_blank_tenant_skips_evaluation() -> None:
    service = _RecordingGovernanceService()
    assert (
        invoke_post_run_governance(
            service,
            tenant_id="",
            run_id="run_1",
            agent_id="agent_1",
        )
        is None
    )
    assert (
        invoke_post_run_governance(
            service,
            tenant_id="   ",
            run_id="run_1",
            agent_id="agent_1",
        )
        is None
    )
    assert service.calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_gov_x2_r3_02_nexus_finish_task_propagates_task_tenant() -> None:
    service = _RecordingGovernanceService()
    loop = NexusLoop(AgentRegistry(), governance_service=service)
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    task_id = mint_task_id()
    task = Task(
        task_id=task_id,
        tenant_id="tenant_A",
        user_id="user_1",
        agent_id="agent_exec_1",
        message="done",
    )
    trace_emitter = TaskTraceEmitter(run_id=run_id, attempt_id=attempt_id)
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=mint_execution_id(),
    )
    from testing_support.runtime_event_metric_scope_for_tests import (
        open_runtime_event_metric_scope_for_tests,
    )

    metric_scope = open_runtime_event_metric_scope_for_tests(
        task_id=task_id, run_id=run_id
    )
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
            runtime_event_metric_scope=metric_scope,
        )
    finally:
        metric_scope.close()
        reset_active_execution_identity(token)

    assert service.calls == [("tenant_A", run_id, "agent_exec_1")]


class _UaepPostRunTenantAgent(HarnessReferenceAgent):
    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="uaep-post-run-tenant",
            name="UAEP post-run tenant",
            description="single-step agent for post-run tenant propagation",
            capabilities=["stub.basic"],
            max_steps=1,
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

    def get_steps(self) -> list[AgentStep]:
        return [AgentStep(step_id="s1", step_name="only", step_index=0)]

    async def run_step(
        self, step: AgentStep, ctx: RuntimeExecutionContext
    ) -> StepOutput:
        _ = ctx
        return StepOutput(step_id=step.step_id, summary="out")

    def decide_after_step(
        self,
        step: AgentStep,
        output: StepOutput | None,
        ctx: RuntimeExecutionContext,
    ) -> AgentDecision:
        _ = step, output, ctx
        return AgentDecision(type=AgentDecisionType.COMPLETE, reason="done")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_gov_x2_r3_03_uaep_propagates_request_tenant() -> None:
    service = _RecordingGovernanceService()
    agent = _UaepPostRunTenantAgent()
    executor = UAEPExecutor(
        event_bus=RuntimeEventBus(),
        policy_engine=PolicyEngine(),
        governance_service=service,
    )
    run_id = mint_run_id()
    task_id = mint_task_id()
    attempt_id = mint_attempt_id()
    request = RuntimeRequest(
        tenant_id="tenant_B",
        user_id="u1",
        session_id="s1",
        agent_id="uaep-post-run-tenant",
        message="hi",
        task_id=task_id,
        run_id=run_id,
    )
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=mint_execution_id(),
    )
    try:
        await executor.execute(agent, request)
    finally:
        reset_active_execution_identity(token)

    assert len(service.calls) == 1
    tenant_id, received_run_id, agent_id = service.calls[0]
    assert tenant_id == "tenant_B"
    assert received_run_id == run_id
    assert agent_id == "uaep-post-run-tenant"


@pytest.mark.unit
def test_gov_x2_r3_05_governance_service_forwards_tenant_to_replay() -> None:
    replay = _RecordingReplay()
    guard = _execution_guard_with_replay(replay)
    service = GovernanceService(guard=guard)
    service.evaluate(tenant_id="tenant_A", run_id="run_X", agent_id="agent_Y")
    assert replay.calls == [("tenant_A", "run_X")]


@pytest.mark.unit
def test_gov_x2_r3_cross_tenant_replay_denial_preserved() -> None:
    guard = _execution_guard_with_replay(_CrossTenantReplay())
    service = GovernanceService(guard=guard)
    with pytest.raises(ValueError, match="cross-tenant replay denied"):
        service.evaluate(
            tenant_id="tenant_A",
            run_id="run_owned_by_B",
            agent_id="agent_1",
        )


@pytest.mark.unit
def test_gov_x2_r3_bridge_does_not_substitute_tenant() -> None:
    service = _RecordingGovernanceService()
    invoke_post_run_governance(
        service,
        tenant_id="tenant_A",
        run_id="run_X",
        agent_id="agent_1",
    )
    assert service.calls[0][0] == "tenant_A"
    assert service.calls[0][0] != "tenant_B"
    assert service.calls[0][0] != "global"
    assert service.calls[0][0] != "default"


@pytest.mark.unit
def test_gov_x2_r3_post_run_protocol_conformance() -> None:
    _accept_post_run(LabAllowGovernanceService())
    replay = _RecordingReplay()
    guard = _execution_guard_with_replay(replay)
    _accept_post_run(GovernanceService(guard=guard))
    _accept_post_run(_RecordingGovernanceService())

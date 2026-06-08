# © Artur Czarnecki. All rights reserved.

"""CRIT-V-3.4 / CRIT-V-3.5 graph critic hook tests."""

from __future__ import annotations

import pytest

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.validation import ValidationResult
from intergrax.runtime.critic.contracts import CriticScope, CriticVerdict, LayerVerdict, CriticLayer
from intergrax.runtime.critic.critic_wiring import (
    CriticGraphHooks,
    CriticHookConfig,
    build_critic_graph_hooks,
    critic_verdict_to_validation_result,
    validate_final_with_critic,
    validate_node_with_critic,
)
from intergrax.runtime.critic.critic_orchestrator import CriticOrchestrator
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.execution.execution_graph import ExecutionGraph, ExecutionNode, ExecutionNodeStatus
from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor
from intergrax.runtime.nexus.pipelines.contract import RuntimePipeline
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer, RuntimeRequest, RouteInfo
from intergrax.runtime.nexus.retry.retry_engine import RetryEngine, RetryPolicy
from intergrax.runtime.nexus.validation.validation_engine import NexusValidationEngine
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _StubPipeline(RuntimePipeline):
    async def _inner_run(self, state):
        answer = f"ok:{state.request.message}"
        state.raw_answer = answer
        state.runtime_answer = RuntimeAnswer(run_id=state.run_id, answer=answer, route=RouteInfo())
        return state.runtime_answer


class _StubAgent(Agent):
    def __init__(self, agent_id: str, capability: str) -> None:
        self._agent_id = agent_id
        self._capability = capability

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id=self._agent_id,
            name=self._agent_id,
            description="critic graph stub",
            capabilities=[self._capability],
        )

    def can_handle(self, task_context: TaskContext) -> CapabilityMatchResult:
        if task_context.capability == self._capability:
            return CapabilityMatchResult(
                matched=True,
                agent_id=self._agent_id,
                matched_capabilities=[self._capability],
                score=1.0,
            )
        return CapabilityMatchResult(matched=False)

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        config = RuntimeConfig(
            llm_adapter=FakeLLMAdapter(fixed_text="ok"),
            enable_rag=False,
            production_mode=False,
        )
        config.pipeline = _StubPipeline()
        return RuntimeContext.build(
            config=config,
            session_manager=build_in_memory_session_manager(),
        )


class _FailOnceValidation(NexusValidationEngine):
    def __init__(self, *, fail_agent: str) -> None:
        super().__init__()
        self._fail_agent = fail_agent
        self._failed: set[str] = set()

    def validate(self, execution, *, contract, capability=None, plan_criteria=None) -> ValidationResult:
        agent_id = contract.id
        if agent_id == self._fail_agent and agent_id not in self._failed:
            self._failed.add(agent_id)
            return ValidationResult(valid=False, errors=["simulated validation failure"])
        return super().validate(
            execution,
            contract=contract,
            capability=capability,
            plan_criteria=plan_criteria,
        )


def test_critic_verdict_to_validation_result_maps_failures() -> None:
    verdict = CriticVerdict(
        scope=CriticScope.NODE_PARTIAL,
        passed=False,
        layers=[
            LayerVerdict(
                layer=CriticLayer.L0_DETERMINISTIC,
                passed=False,
                score=0.0,
                errors=["empty summary"],
            ),
        ],
        failure_reasons=["empty summary"],
    )
    result = critic_verdict_to_validation_result(verdict)
    assert result.valid is False
    assert "empty summary" in result.errors


def test_validate_node_with_critic_delegates_to_orchestrator() -> None:
    validation = _FailOnceValidation(fail_agent="worker")
    hooks = build_critic_graph_hooks(
        config=CriticHookConfig(verify_node_partial=True),
        validation_engine=validation,
    )
    assert hooks is not None
    contract = AgentContract(id="worker", name="worker", description="x")
    execution = AgentExecutionResult(
        agent_id="worker",
        run_id="run-1",
        status=AgentExecutionStatus.COMPLETED,
        summary="ok",
    )
    first = validate_node_with_critic(
        execution,
        contract=contract,
        hooks=hooks,
        run_id="run-1",
        tenant_id="tenant-1",
    )
    assert first.valid is False


@pytest.mark.asyncio
async def test_graph_executor_critic_partial_l0_fail_triggers_retry() -> None:
    registry = AgentRegistry()
    registry.register(_StubAgent("agent_a", "cap.shared"))
    registry.register(_StubAgent("agent_b", "cap.shared"))

    validation = _FailOnceValidation(fail_agent="agent_a")
    hooks = build_critic_graph_hooks(
        config=CriticHookConfig(verify_node_partial=True, verify_graph_final=False),
        validation_engine=validation,
    )
    task = Task(tenant_id="t1", user_id="u1", message="retry", context=TaskContext(capability="cap.shared"))
    graph = ExecutionGraph(
        graph_id="critic_retry_graph",
        task_id=task.task_id,
        nodes=[ExecutionNode(node_id="n1", agent_id="agent_a", capability="cap.shared")],
    )
    executor = GraphExecutor(
        registry,
        validation_engine=validation,
        critic_graph_hooks=hooks,
        retry_engine=RetryEngine(registry, policy=RetryPolicy(max_retries=1)),
    )
    executions, retries, graph_out, _ = await executor.execute(graph, task)

    assert len(retries) == 1
    assert retries[0].alternate_agent_id == "agent_b"
    assert executions[-1].agent_id == "agent_b"
    assert graph_out.node_by_id("n1").status == ExecutionNodeStatus.COMPLETED


def test_validate_final_with_critic_uses_graph_final_scope() -> None:
    class _RecordingOrchestrator(CriticOrchestrator):
        last_scope = None

        def verify_final(self, request, *, contract=None):
            _RecordingOrchestrator.last_scope = request.scope
            return CriticVerdict(scope=request.scope, passed=True, layers=[])

    hooks = CriticGraphHooks(
        orchestrator=_RecordingOrchestrator(),
        config=CriticHookConfig(verify_graph_final=True),
    )
    contract = AgentContract(id="worker", name="worker", description="x")
    execution = AgentExecutionResult(
        agent_id="worker",
        run_id="run-final",
        status=AgentExecutionStatus.COMPLETED,
        summary="done",
    )
    result = validate_final_with_critic(
        execution,
        contract=contract,
        hooks=hooks,
        run_id="run-final",
        tenant_id="tenant-1",
    )
    assert result.valid is True
    assert _RecordingOrchestrator.last_scope is CriticScope.GRAPH_FINAL

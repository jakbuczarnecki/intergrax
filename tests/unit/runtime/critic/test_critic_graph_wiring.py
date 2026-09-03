# © Artur Czarnecki. All rights reserved.

"""CRIT-V-3.4 / CRIT-V-3.5 graph critic hook tests."""

from __future__ import annotations

import pytest

from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.execution_identity import bind_active_execution_identity, mint_attempt_id, mint_run_id
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
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
from intergrax.runtime.nexus.execution.execution_graph import ExecutionGraph, ExecutionNode, ExecutionNodeStatus
from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor
from intergrax.runtime.nexus.retry.retry_engine import RetryEngine, RetryPolicy
from intergrax.runtime.nexus.validation.validation_engine import NexusValidationEngine
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext
from testing_support.uaep_gate_stubs import UaepPipelineStubAgent

pytestmark = [pytest.mark.unit, pytest.mark.gate]


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
    run_id = mint_run_id()
    task = Task(tenant_id="tenant-1", user_id="u1", message="ok")
    first = validate_node_with_critic(
        execution,
        contract=contract,
        hooks=hooks,
        task_id=task.task_id,
        run_id=run_id,
        tenant_id="tenant-1",
    )
    assert first.valid is False


@pytest.mark.asyncio
@pytest.mark.skip(reason="GraphExecutor critic authority retired in DS-MIG-02")
async def test_graph_executor_critic_partial_l0_fail_triggers_retry() -> None:
    registry = AgentRegistry()
    registry.register(
        UaepPipelineStubAgent(
            agent_id="agent_a",
            capability="cap.shared",
            prefix="ok",
            answer_separator=":",
            description="critic graph stub",
        )
    )
    registry.register(
        UaepPipelineStubAgent(
            agent_id="agent_b",
            capability="cap.shared",
            prefix="ok",
            answer_separator=":",
            description="critic graph stub",
        )
    )

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
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    token = bind_active_execution_identity(run_id=run_id, attempt_id=attempt_id)
    try:
        executions, retries, graph_out, _ = await executor.execute(graph, task)
    finally:
        from intergrax.contracts.execution_identity import reset_active_execution_identity

        reset_active_execution_identity(token)

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
    run_id = mint_run_id()
    task = Task(tenant_id="tenant-1", user_id="u1", message="done")
    result = validate_final_with_critic(
        execution,
        contract=contract,
        hooks=hooks,
        task_id=task.task_id,
        run_id=run_id,
        tenant_id="tenant-1",
    )
    assert result.valid is True
    assert _RecordingOrchestrator.last_scope is CriticScope.GRAPH_FINAL

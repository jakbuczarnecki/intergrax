# © Artur Czarnecki. All rights reserved.

"""UE-11E — local retry recovery qualification."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from intergrax.agents.agent_engine import AgentEngine
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    peek_active_execution_id,
    peek_active_execution_identity,
    require_active_execution_id,
    require_active_execution_identity,
)
from intergrax.contracts.validation import ValidationResult
from intergrax.runtime.execution.active_execution_budget import (
    bind_root_execution_budget,
    peek_active_execution_budget,
    require_active_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.governance.active_execution_authority import peek_active_execution_authority
from intergrax.runtime.execution.boundary import ExecutionBoundary, ExecutionIdentityBinding
from intergrax.runtime.execution.budget.consumption import consume_llm_call
from intergrax.runtime.execution.budget.ledger import (
    InMemoryExecutionBudgetLedger,
    create_execution_budget_ledger,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.nexus.execution.execution_graph import (
    ExecutionGraph,
    ExecutionNode,
    ExecutionNodeStatus,
)
from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor
from intergrax.runtime.nexus.retry.retry_engine import RetryEngine, RetryPolicy
from intergrax.runtime.nexus.validation.validation_engine import NexusValidationEngine
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext
from testing_support.uaep_gate_stubs import UaepPipelineStubAgent

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@dataclass(frozen=True, slots=True)
class _RetryIdentityObservation:
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId


class _BudgetObservingAgentEngine(AgentEngine):
    """Records identity and governed consumption during each agent invocation."""

    def __init__(self, registry: AgentRegistry) -> None:
        super().__init__(registry)
        self.identity_observations: list[_RetryIdentityObservation] = []
        self.invocation_count: int = 0

    async def run_with_result(self, request):
        run_id, attempt_id = require_active_execution_identity()
        execution_id = require_active_execution_id()
        self.invocation_count += 1
        consume_llm_call()
        assert require_active_execution_budget().execution_id == execution_id
        self.identity_observations.append(
            _RetryIdentityObservation(
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
            )
        )
        return await super().run_with_result(request)


class _FailOnceValidation(NexusValidationEngine):
    def __init__(self, *, fail_agent: str) -> None:
        super().__init__()
        self._fail_agent = fail_agent
        self._failed: set[str] = set()

    def validate(
        self,
        execution: AgentExecutionResult,
        *,
        contract,
        capability=None,
        plan_criteria=None,
    ) -> ValidationResult:
        agent_id = contract.id
        if agent_id == self._fail_agent and agent_id not in self._failed:
            self._failed.add(agent_id)
            return ValidationResult(valid=False, errors=["controlled retriable failure"])
        return super().validate(
            execution,
            contract=contract,
            capability=capability,
            plan_criteria=plan_criteria,
        )


@dataclass
class _GraphRunBundle:
    executor: GraphExecutor
    graph: ExecutionGraph
    task: Task
    ledger: InMemoryExecutionBudgetLedger


class _GraphOrchestrationDelegate:
    __slots__ = ("_bundle",)

    def __init__(self, bundle: _GraphRunBundle) -> None:
        self._bundle = bundle

    async def execute(self, _request: object) -> tuple[object, ...]:
        budget_token = bind_root_execution_budget(
            execution_id=require_active_execution_id(),
            ledger=self._bundle.ledger,
        )
        try:
            return await self._bundle.executor.execute(
                self._bundle.graph,
                self._bundle.task,
            )
        finally:
            reset_active_execution_budget(budget_token)


async def test_ue_11e_local_retry_preserves_identity_and_budget() -> None:
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root_execution_id = mint_execution_id()
    registry = AgentRegistry()
    registry.register(
        UaepPipelineStubAgent(
            agent_id="agent_a",
            capability="cap.shared",
            prefix="agent_a",
            answer_separator=":",
        ),
    )
    registry.register(
        UaepPipelineStubAgent(
            agent_id="agent_b",
            capability="cap.shared",
            prefix="agent_b",
            answer_separator=":",
        ),
    )
    engine = _BudgetObservingAgentEngine(registry)
    executor = GraphExecutor(
        registry,
        engine=engine,
        validation_engine=_FailOnceValidation(fail_agent="agent_a"),
        retry_engine=RetryEngine(registry, policy=RetryPolicy(max_retries=1)),
    )
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="ue-11e retry",
        context=TaskContext(capability="cap.shared"),
    )
    graph = ExecutionGraph(
        graph_id="retry-root",
        task_id=task.task_id,
        nodes=[ExecutionNode(node_id="n1", agent_id="agent_a", capability="cap.shared")],
    )
    ledger = create_execution_budget_ledger(RunBudget(max_llm_calls=5))
    bundle = _GraphRunBundle(
        executor=executor,
        graph=graph,
        task=task,
        ledger=ledger,
    )
    boundary = ExecutionBoundary(
        _GraphOrchestrationDelegate(bundle),
        identity=ExecutionIdentityBinding(
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=root_execution_id,
        ),
        authority=ParentExecutionAuthority.scoped(("capability:read",)),
    )

    executions, retries, graph_out, _ = await boundary.execute(None)

    assert len(retries) == 1
    assert engine.invocation_count == 2
    assert len(engine.identity_observations) == 2
    first, second = engine.identity_observations
    assert first.run_id == run_id == second.run_id
    assert first.attempt_id == attempt_id == second.attempt_id
    assert first.execution_id == second.execution_id
    assert first.execution_id != root_execution_id
    assert graph_out.node_by_id("n1").status == ExecutionNodeStatus.COMPLETED
    assert executions[-1].status == AgentExecutionStatus.COMPLETED
    assert peek_active_execution_identity() is None
    assert peek_active_execution_id() is None
    assert peek_active_execution_authority() is None
    assert peek_active_execution_budget() is None
    assert ledger.snapshot_root_available().max_llm_calls == 3

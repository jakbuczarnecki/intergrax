# © Artur Czarnecki. All rights reserved.

"""P0-B evaluator-loop revision boundedness regression tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skip(reason="GraphExecutor critic authority retired in DS-MIG-02"),
]

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_run_id,
    reset_active_execution_identity,
)
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.contracts.validation import ValidationResult
from intergrax.runtime.critic.critic_wiring import (
    CriticHookConfig,
    build_critic_graph_hooks,
)
from intergrax.runtime.critic.evaluator_loop_metadata import (
    current_evaluator_loop_iteration,
    tag_node_evaluator_loop,
)
from intergrax.runtime.critic.evaluator_loop_spec import EvaluatorLoopSpec
from intergrax.runtime.critic.trace import CriticTraceEmitter
from intergrax.runtime.critic.trace_steps import CRITIC_STEP_EVALUATOR_LOOP
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.execution.execution_graph import (
    ExecutionGraph,
    ExecutionNode,
    ExecutionNodeStatus,
)
from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.retry.retry_engine import RetryEngine, RetryPolicy
from intergrax.runtime.nexus.tracing.in_memory_trace_store import InMemoryRunTraceStore
from intergrax.runtime.nexus.validation.validation_engine import NexusValidationEngine
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@dataclass
class _ExecutionCounts:
    worker: int = 0
    revise: int = 0
    evaluator_loop_entries: int = 0


class _CountingStubAgent(Agent):
    def __init__(self, agent_id: str, capability: str, counts: _ExecutionCounts, *, role: str) -> None:
        self._agent_id = agent_id
        self._capability = capability
        self._counts = counts
        self._role = role

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id=self._agent_id,
            name=self._agent_id,
            description="evaluator boundedness stub",
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
        return RuntimeContext.build(
            config=config,
            session_manager=build_in_memory_session_manager(),
        )

    def get_steps(self, context: RuntimeContext) -> list[AgentStep]:
        _ = context
        return [
            AgentStep(
                step_id=f"{self._agent_id}_step",
                step_name=f"{self._agent_id}_step",
                step_index=0,
                trace_label=self._capability,
            )
        ]

    async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
        _ = step
        if self._role == "worker":
            self._counts.worker += 1
            feedback = (ctx.request.metadata or {}).get("critic_feedback") if ctx.request else None
            answer = "revised:ok" if feedback else "draft:needs work"
        else:
            self._counts.revise += 1
            answer = "revise:prepared"
        return StepOutput(step_id=step.step_id, summary=answer, data={"answer": answer})

    def decide_after_step(
        self,
        step: AgentStep,
        output: StepOutput | None,
        ctx: RuntimeExecutionContext,
    ) -> AgentDecision:
        _ = step, output, ctx
        return AgentDecision(type=AgentDecisionType.COMPLETE, reason=f"{self._agent_id} done")


class _FailUntilRevisedValidation(NexusValidationEngine):
    def validate(self, execution, *, contract, capability=None, plan_criteria=None) -> ValidationResult:
        summary = execution.summary or ""
        if summary.startswith("revised:"):
            return ValidationResult(valid=True)
        return ValidationResult(valid=False, errors=["needs revision"])


class _AlwaysFailValidation(NexusValidationEngine):
    def validate(self, execution, *, contract, capability=None, plan_criteria=None) -> ValidationResult:
        _ = execution, contract, capability, plan_criteria
        return ValidationResult(valid=False, errors=["always fails"])


def _attach_revise_node(graph: ExecutionGraph, revise: ExecutionNode) -> None:
    """Expose a revise target for evaluator routing without scheduling it in batches."""

    def extended_node_by_id(node_id: str) -> ExecutionNode:
        if node_id == revise.node_id:
            return revise
        for node in graph.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(f"ExecutionNode not found: {node_id}")

    object.__setattr__(graph, "node_by_id", extended_node_by_id)


async def _execute_graph(
    executor: GraphExecutor,
    graph: ExecutionGraph,
    task: Task,
    **kwargs: Any,
):
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    token = bind_active_execution_identity(run_id=run_id, attempt_id=attempt_id)
    try:
        return await executor.execute(graph, task, **kwargs)
    finally:
        reset_active_execution_identity(token)


def _build_executor(
    validation: NexusValidationEngine,
    *,
    registry: AgentRegistry | None = None,
    retry_engine: RetryEngine | None = None,
) -> GraphExecutor:
    hooks = build_critic_graph_hooks(
        config=CriticHookConfig(verify_node_partial=True, verify_graph_final=False),
        validation_engine=validation,
    )
    assert hooks is not None
    resolved_registry = registry or AgentRegistry()
    resolved_retry = retry_engine or RetryEngine(
        resolved_registry,
        policy=RetryPolicy(max_retries=0, retry_alternate_agent=False),
    )
    return GraphExecutor(
        resolved_registry,
        validation_engine=validation,
        critic_graph_hooks=hooks,
        retry_engine=resolved_retry,
    )


def _worker_node(
    *,
    node_id: str = "worker-1",
    agent_id: str = "worker",
    capability: str = "cap.loop",
    max_iterations: int = 2,
    revise_node_id: str = "worker-1",
    escalate_on_exhaustion: bool = False,
) -> ExecutionNode:
    worker = ExecutionNode(node_id=node_id, agent_id=agent_id, capability=capability)
    tag_node_evaluator_loop(
        worker,
        EvaluatorLoopSpec(
            max_iterations=max_iterations,
            revise_node_id=revise_node_id,
            escalate_on_exhaustion=escalate_on_exhaustion,
        ),
    )
    return worker


def _register_loop_agents(counts: _ExecutionCounts) -> AgentRegistry:
    registry = AgentRegistry()
    registry.register(_CountingStubAgent("worker", "cap.loop", counts, role="worker"))
    registry.register(_CountingStubAgent("revise", "cap.revise", counts, role="revise"))
    return registry


@pytest.mark.asyncio
async def test_one_revision_succeeds() -> None:
    """TEST A — one critic fail, one revision, then pass."""
    counts = _ExecutionCounts()
    validation = _FailUntilRevisedValidation()
    registry = _register_loop_agents(counts)
    executor = _build_executor(validation, registry=registry)

    worker = _worker_node()
    task = Task(tenant_id="t1", user_id="u1", message="loop", context=TaskContext(capability="cap.loop"))
    graph = ExecutionGraph(graph_id="bounded_loop", task_id=task.task_id, nodes=[worker])

    with patch.object(
        GraphExecutor,
        "_maybe_run_evaluator_loop",
        wraps=executor._maybe_run_evaluator_loop,
    ) as loop_spy:
        executions, _, graph_out, _ = await _execute_graph(executor, graph, task)

    assert graph_out.node_by_id("worker-1").status == ExecutionNodeStatus.COMPLETED
    assert counts.worker == 2
    assert current_evaluator_loop_iteration(worker) == 1
    assert loop_spy.call_count == 1
    assert any("revised:" in (e.summary or "") for e in executions)


@pytest.mark.asyncio
async def test_repeated_critic_failure_respects_global_bound() -> None:
    """TEST B — budget exhaustion without RecursionError."""
    counts = _ExecutionCounts()
    validation = _AlwaysFailValidation()
    registry = _register_loop_agents(counts)
    executor = _build_executor(validation, registry=registry)

    worker = _worker_node(max_iterations=2, escalate_on_exhaustion=False)
    task = Task(tenant_id="t1", user_id="u1", message="loop", context=TaskContext(capability="cap.loop"))
    graph = ExecutionGraph(graph_id="bounded_fail", task_id=task.task_id, nodes=[worker])

    _, _, graph_out, _ = await _execute_graph(executor, graph, task)

    assert graph_out.node_by_id("worker-1").status == ExecutionNodeStatus.FAILED
    assert counts.worker == 2
    assert current_evaluator_loop_iteration(worker) == 1


@pytest.mark.asyncio
async def test_nested_revise_target_cannot_reset_budget() -> None:
    """TEST C — revise node with evaluator spec does not spawn fresh lifecycle."""
    counts = _ExecutionCounts()
    validation = _AlwaysFailValidation()
    registry = _register_loop_agents(counts)
    executor = _build_executor(validation, registry=registry)

    worker = _worker_node(max_iterations=2, revise_node_id="revise-1", escalate_on_exhaustion=False)
    revise = ExecutionNode(node_id="revise-1", agent_id="revise", capability="cap.revise")
    tag_node_evaluator_loop(
        revise,
        EvaluatorLoopSpec(max_iterations=4, revise_node_id="revise-1", escalate_on_exhaustion=False),
    )

    task = Task(tenant_id="t1", user_id="u1", message="loop", context=TaskContext(capability="cap.loop"))
    graph = ExecutionGraph(graph_id="nested_revise", task_id=task.task_id, nodes=[worker])
    _attach_revise_node(graph, revise)

    with patch.object(
        GraphExecutor,
        "_maybe_run_evaluator_loop",
        wraps=executor._maybe_run_evaluator_loop,
    ) as loop_spy:
        _, _, graph_out, _ = await _execute_graph(executor, graph, task)

    assert graph_out.node_by_id("worker-1").status == ExecutionNodeStatus.FAILED
    assert loop_spy.call_count == 1
    assert counts.worker == 2
    assert counts.revise == 1
    assert current_evaluator_loop_iteration(worker) == 1


@pytest.mark.asyncio
async def test_no_duplicate_execution_per_self_revise() -> None:
    """TEST D — self-revise causes exactly one worker rerun, not recursive + retry."""
    counts = _ExecutionCounts()
    validation = _FailUntilRevisedValidation()
    registry = _register_loop_agents(counts)
    executor = _build_executor(validation, registry=registry)

    worker = _worker_node()
    task = Task(tenant_id="t1", user_id="u1", message="loop", context=TaskContext(capability="cap.loop"))
    graph = ExecutionGraph(graph_id="no_dup", task_id=task.task_id, nodes=[worker])

    await _execute_graph(executor, graph, task)

    assert counts.worker == 2
    assert counts.revise == 0


@pytest.mark.asyncio
async def test_separate_revise_node_single_worker_retry() -> None:
    """TEST D (separate revise) — one revise execution + one worker retry per REVISE."""
    counts = _ExecutionCounts()
    validation = _FailUntilRevisedValidation()
    registry = _register_loop_agents(counts)
    executor = _build_executor(validation, registry=registry)

    worker = _worker_node(revise_node_id="revise-1")
    revise = ExecutionNode(node_id="revise-1", agent_id="revise", capability="cap.revise")
    task = Task(tenant_id="t1", user_id="u1", message="loop", context=TaskContext(capability="cap.loop"))
    graph = ExecutionGraph(graph_id="separate_revise", task_id=task.task_id, nodes=[worker])
    _attach_revise_node(graph, revise)

    await _execute_graph(executor, graph, task)

    assert counts.worker == 2
    assert counts.revise == 1


@pytest.mark.asyncio
async def test_critic_feedback_propagated_to_revision() -> None:
    """TEST E — revision request receives critic failure reasons."""
    captured_feedback: list[list[str]] = []

    class _FeedbackCapturingAgent(_CountingStubAgent):
        async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
            feedback = (ctx.request.metadata or {}).get("critic_feedback") if ctx.request else None
            if isinstance(feedback, list):
                captured_feedback.append(list(feedback))
            return await super().run_step(step, ctx)

    counts = _ExecutionCounts()
    validation = _FailUntilRevisedValidation()
    executor = _build_executor(validation)
    registry = AgentRegistry()
    registry.register(_FeedbackCapturingAgent("worker", "cap.loop", counts, role="worker"))
    executor = _build_executor(validation, registry=registry)

    worker = _worker_node()
    task = Task(tenant_id="t1", user_id="u1", message="loop", context=TaskContext(capability="cap.loop"))
    graph = ExecutionGraph(graph_id="feedback", task_id=task.task_id, nodes=[worker])

    await _execute_graph(executor, graph, task)

    assert worker.metadata.get("critic_feedback") == ["needs revision"]
    assert captured_feedback
    assert captured_feedback[-1] == ["needs revision"]


@pytest.mark.asyncio
async def test_prior_outputs_preserved_on_revision() -> None:
    """TEST F — worker prior execution is recorded before revision retry."""
    counts = _ExecutionCounts()
    validation = _FailUntilRevisedValidation()
    registry = _register_loop_agents(counts)
    executor = _build_executor(validation, registry=registry)

    worker = _worker_node(revise_node_id="revise-1")
    revise = ExecutionNode(node_id="revise-1", agent_id="revise", capability="cap.revise")
    task = Task(tenant_id="t1", user_id="u1", message="loop", context=TaskContext(capability="cap.loop"))
    graph = ExecutionGraph(graph_id="prior_outputs", task_id=task.task_id, nodes=[worker])
    _attach_revise_node(graph, revise)

    executions, _, graph_out, _ = await _execute_graph(executor, graph, task)

    assert graph_out.node_by_id("worker-1").status == ExecutionNodeStatus.COMPLETED
    assert counts.worker == 2
    assert counts.revise == 1
    assert any((e.summary or "").startswith("revised:") for e in executions)


@pytest.mark.asyncio
async def test_operational_retry_does_not_reset_evaluator_iteration() -> None:
    """TEST G — retry engine retries do not reset evaluator revision counter."""
    counts = _ExecutionCounts()
    validation = _AlwaysFailValidation()
    registry = _register_loop_agents(counts)
    retry_engine = RetryEngine(registry, policy=RetryPolicy(max_retries=0, retry_alternate_agent=False))
    executor = _build_executor(validation, registry=registry, retry_engine=retry_engine)

    worker = _worker_node(max_iterations=3, escalate_on_exhaustion=False)
    task = Task(tenant_id="t1", user_id="u1", message="loop", context=TaskContext(capability="cap.loop"))
    graph = ExecutionGraph(graph_id="retry_sep", task_id=task.task_id, nodes=[worker])

    await _execute_graph(executor, graph, task)

    assert current_evaluator_loop_iteration(worker) == 2
    assert counts.worker == 3


@pytest.mark.asyncio
async def test_recursion_regression_finite_evaluator_path() -> None:
    """Regression — previously nested _execute_node + retry could recurse unboundedly."""
    counts = _ExecutionCounts()
    validation = _AlwaysFailValidation()
    registry = _register_loop_agents(counts)
    executor = _build_executor(validation, registry=registry)

    worker = _worker_node(max_iterations=3, revise_node_id="revise-1", escalate_on_exhaustion=False)
    revise = ExecutionNode(node_id="revise-1", agent_id="revise", capability="cap.revise")
    tag_node_evaluator_loop(
        revise,
        EvaluatorLoopSpec(max_iterations=3, revise_node_id="revise-1"),
    )
    task = Task(tenant_id="t1", user_id="u1", message="loop", context=TaskContext(capability="cap.loop"))
    graph = ExecutionGraph(graph_id="recursion", task_id=task.task_id, nodes=[worker])
    _attach_revise_node(graph, revise)

    with patch.object(
        GraphExecutor,
        "_maybe_run_evaluator_loop",
        wraps=executor._maybe_run_evaluator_loop,
    ) as loop_spy:
        _, _, graph_out, _ = await _execute_graph(executor, graph, task)

    assert graph_out.node_by_id("worker-1").status == ExecutionNodeStatus.FAILED
    assert loop_spy.call_count == 1
    assert counts.worker == 3
    assert counts.revise == 2


@pytest.mark.asyncio
async def test_evaluator_loop_trace_iterations_monotonic() -> None:
    """Trace exposes one lifecycle with monotonic iteration values."""
    store = InMemoryRunTraceStore()
    trace = CriticTraceEmitter(run_id="run-bounded", trace_writer=store)
    counts = _ExecutionCounts()
    validation = _FailUntilRevisedValidation()
    registry = _register_loop_agents(counts)
    executor = _build_executor(validation, registry=registry)

    worker = _worker_node()
    task = Task(tenant_id="t1", user_id="u1", message="loop", context=TaskContext(capability="cap.loop"))
    graph = ExecutionGraph(graph_id="trace_loop", task_id=task.task_id, nodes=[worker])

    await _execute_graph(executor, graph, task, critic_trace_emitter=trace)

    loop_events = [
        event
        for event in store._events_by_run.get("run-bounded", [])
        if event.step == CRITIC_STEP_EVALUATOR_LOOP
    ]
    iterations = [event.tags.get("iteration") for event in loop_events]
    assert 0 in iterations
    assert iterations.count(0) == 1
    assert iterations == sorted(iterations)

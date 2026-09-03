# © Artur Czarnecki. All rights reserved.

"""CRIT-V-4.2 evaluator-loop graph integration tests."""

from __future__ import annotations

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
from intergrax.runtime.critic.evaluator_loop_metadata import tag_node_evaluator_loop
from intergrax.runtime.critic.evaluator_loop_spec import EvaluatorLoopSpec
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.execution.execution_graph import ExecutionGraph, ExecutionNode, ExecutionNodeStatus
from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.validation.validation_engine import NexusValidationEngine
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _StubAgent(Agent):
    def __init__(self, agent_id: str, capability: str) -> None:
        self._agent_id = agent_id
        self._capability = capability

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id=self._agent_id,
            name=self._agent_id,
            description="evaluator loop stub",
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
        feedback = (ctx.request.metadata or {}).get("critic_feedback") if ctx.request else None
        answer = "revised:ok" if feedback else "draft:needs work"
        return StepOutput(step_id=step.step_id, summary=answer, data={"answer": answer})

    def decide_after_step(
        self,
        step: AgentStep,
        output: StepOutput | None,
        ctx: RuntimeExecutionContext,
    ) -> AgentDecision:
        _ = step, output, ctx
        return AgentDecision(type=AgentDecisionType.COMPLETE, reason=f"{self._agent_id} evaluator loop stub finished")


class _FailUntilRevisedValidation(NexusValidationEngine):
    def validate(self, execution, *, contract, capability=None, plan_criteria=None) -> ValidationResult:
        summary = execution.summary or ""
        if summary.startswith("revised:"):
            return ValidationResult(valid=True)
        return ValidationResult(valid=False, errors=["needs revision"])


@pytest.mark.asyncio
async def test_graph_executor_evaluator_loop_two_iterations() -> None:
    registry = AgentRegistry()
    registry.register(_StubAgent("worker", "cap.loop"))

    validation = _FailUntilRevisedValidation()
    hooks = build_critic_graph_hooks(
        config=CriticHookConfig(verify_node_partial=True, verify_graph_final=False),
        validation_engine=validation,
    )
    assert hooks is not None

    worker = ExecutionNode(node_id="worker-1", agent_id="worker", capability="cap.loop")
    tag_node_evaluator_loop(
        worker,
        EvaluatorLoopSpec(max_iterations=2, revise_node_id="worker-1", escalate_on_exhaustion=False),
    )

    task = Task(tenant_id="t1", user_id="u1", message="loop", context=TaskContext(capability="cap.loop"))
    graph = ExecutionGraph(
        graph_id="evaluator_loop_graph",
        task_id=task.task_id,
        nodes=[worker],
    )
    executor = GraphExecutor(registry, validation_engine=validation, critic_graph_hooks=hooks)
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    token = bind_active_execution_identity(run_id=run_id, attempt_id=attempt_id)
    try:
        executions, _, graph_out, _ = await executor.execute(graph, task)
    finally:
        reset_active_execution_identity(token)

    assert graph_out.node_by_id("worker-1").status == ExecutionNodeStatus.COMPLETED
    assert any("revised:" in (e.summary or "") for e in executions)

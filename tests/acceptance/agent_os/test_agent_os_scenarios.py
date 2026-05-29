# © Artur Czarnecki. All rights reserved.

"""
Agent OS acceptance scenarios (Phase L.5).

Each test maps to a mandatory runtime acceptance criterion.
"""

from __future__ import annotations

import pytest

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType, HumanRequest
from intergrax.contracts.agent_execution_result import AgentExecutionStatus
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.contracts.tool_request import ToolRequest
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.long_running.runtime_checkpoint import (
    RUNTIME_CHECKPOINT_KEY,
    UAEP_STEP_CURSOR_KEY,
    runtime_checkpoint_from_metadata,
)
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.execution.execution_graph import ExecutionGraph, ExecutionNode, ExecutionNodeStatus
from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.pipelines.contract import RuntimePipeline
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer, RuntimeRequest
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.nexus.retry.retry_engine import RetryEngine, RetryPolicy
from intergrax.runtime.nexus.validation.validation_engine import NexusValidationEngine
from intergrax.runtime.sandbox.manager import SandboxSessionManager
from intergrax.runtime.sandbox.sandbox_runtime import SANDBOX_FLAG, SANDBOX_TOOL_NAME
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskLongRunningOptions
from intergrax.runtime.nexus.context.shared_task_context import load_shared_task_context_from_metadata
from intergrax.runtime.workspace.manager import ShadowWorkspaceManager
from intergrax.runtime.workspace.shadow_workspace import SHADOW_WORKSPACE_FLAG

from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager

pytestmark = [pytest.mark.integration, pytest.mark.agent_os, pytest.mark.gate]


class _AnswerPipeline(RuntimePipeline):
    def __init__(self, prefix: str) -> None:
        self._prefix = prefix

    async def _inner_run(self, state: RuntimeState) -> RuntimeAnswer:
        answer = f"{self._prefix}: {state.request.message}"
        state.runtime_answer = RuntimeAnswer(run_id=state.run_id, answer=answer)
        return state.runtime_answer


class _GraphStubAgent(Agent):
    run_log: list[str] = []

    def __init__(self, *, agent_id: str, capability: str, prefix: str) -> None:
        self._agent_id = agent_id
        self._capability = capability
        self._prefix = prefix

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id=self._agent_id,
            name=self._agent_id,
            description="acceptance graph stub",
            capabilities=[self._capability],
        )

    def can_handle(self, task_context: object) -> CapabilityMatchResult:
        capability = getattr(task_context, "capability", None)
        if capability == self._capability:
            return CapabilityMatchResult(
                matched=True,
                agent_id=self._agent_id,
                matched_capabilities=[self._capability],
                score=1.0,
            )
        return CapabilityMatchResult(matched=False)

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        config = RuntimeConfig(
            llm_adapter=FakeLLMAdapter(fixed_text=f"{self._prefix}: {request.message}"),
            enable_rag=False,
            production_mode=False,
            tenant_id=request.tenant_id,
        )
        config.pipeline = _AnswerPipeline(self._prefix)
        return RuntimeContext.build(
            config=config,
            session_manager=build_in_memory_session_manager(),
        )


class _HitlAcceptanceAgent(Agent):
    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="hitl_acceptance",
            name="HITL Acceptance",
            description="acceptance HITL agent",
            capabilities=["acceptance.hitl"],
            max_steps=1,
        )

    def can_handle(self, task_context: object) -> CapabilityMatchResult:
        capability = getattr(task_context, "capability", None)
        if capability in (None, "acceptance.hitl"):
            return CapabilityMatchResult(
                matched=True,
                agent_id="hitl_acceptance",
                matched_capabilities=["acceptance.hitl"],
                score=1.0,
            )
        return CapabilityMatchResult(matched=False)

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
        return [AgentStep(step_id="review", step_name="review", step_index=0)]

    async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
        _ = ctx
        return StepOutput(step_id=step.step_id, summary="needs approval")

    def decide_after_step(
        self,
        step: AgentStep,
        output: StepOutput | None,
        ctx: RuntimeExecutionContext,
    ) -> AgentDecision:
        _ = step, output
        if ctx.request and ctx.request.metadata.get("human_approved"):
            return AgentDecision(type=AgentDecisionType.COMPLETE, reason="approved")
        return AgentDecision(
            type=AgentDecisionType.REQUEST_HUMAN,
            reason="approval required",
            human_request=HumanRequest(request_id="hr_acceptance", prompt="Approve?", options=["approve"]),
        )


class _MidStepAcceptanceAgent(Agent):
    """UAEP agent that pauses mid-step (phase 1 done, phase 2 after HITL resume)."""

    phase1_runs: int = 0

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="mid_step_acceptance",
            name="Mid-Step Acceptance",
            description="acceptance mid-step UAEP resume",
            capabilities=["acceptance.mid_step"],
            max_steps=1,
        )

    def can_handle(self, task_context: object) -> CapabilityMatchResult:
        capability = getattr(task_context, "capability", None)
        if capability in (None, "acceptance.mid_step"):
            return CapabilityMatchResult(
                matched=True,
                agent_id="mid_step_acceptance",
                matched_capabilities=["acceptance.mid_step"],
                score=1.0,
            )
        return CapabilityMatchResult(matched=False)

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
        return [AgentStep(step_id="process", step_name="process", step_index=0)]

    async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
        cursor = ctx.metadata.get(UAEP_STEP_CURSOR_KEY)
        if cursor and cursor.get("phase1_done"):
            return StepOutput(step_id=step.step_id, summary="mid-step complete")
        _MidStepAcceptanceAgent.phase1_runs += 1
        ctx.metadata[UAEP_STEP_CURSOR_KEY] = {"phase1_done": True}
        return StepOutput(step_id=step.step_id, summary="phase1 partial")

    def decide_after_step(
        self,
        step: AgentStep,
        output: StepOutput | None,
        ctx: RuntimeExecutionContext,
    ) -> AgentDecision:
        _ = step, output
        if ctx.request and ctx.request.metadata.get("human_approved"):
            return AgentDecision(type=AgentDecisionType.COMPLETE, reason="approved")
        if ctx.metadata.get(UAEP_STEP_CURSOR_KEY, {}).get("phase1_done"):
            return AgentDecision(
                type=AgentDecisionType.REQUEST_HUMAN,
                reason="mid-step approval",
                human_request=HumanRequest(
                    request_id="hr_mid_step",
                    prompt="Continue phase 2?",
                    options=["approve"],
                ),
            )
        return AgentDecision(type=AgentDecisionType.CONTINUE, reason="continue")


class _RetryPrimaryAgent(Agent):
    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="retry_primary",
            name="Retry Primary",
            description="acceptance retry primary",
            capabilities=["acceptance.retry"],
            max_steps=1,
        )

    def can_handle(self, task_context: object) -> CapabilityMatchResult:
        capability = getattr(task_context, "capability", None)
        if capability in (None, "acceptance.retry"):
            return CapabilityMatchResult(
                matched=True,
                agent_id="retry_primary",
                matched_capabilities=["acceptance.retry"],
                score=1.0,
            )
        return CapabilityMatchResult(matched=False)

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
        return [AgentStep(step_id="run", step_name="run", step_index=0)]

    async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
        _ = ctx
        return StepOutput(step_id=step.step_id, summary="")

    def decide_after_step(
        self,
        step: AgentStep,
        output: StepOutput | None,
        ctx: RuntimeExecutionContext,
    ) -> AgentDecision:
        _ = step, output, ctx
        return AgentDecision(type=AgentDecisionType.COMPLETE, reason="done")


class _RetryAlternateAgent(Agent):
    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="retry_alternate",
            name="Retry Alternate",
            description="acceptance retry alternate",
            capabilities=["acceptance.retry"],
            max_steps=1,
        )

    def can_handle(self, task_context: object) -> CapabilityMatchResult:
        capability = getattr(task_context, "capability", None)
        if capability in (None, "acceptance.retry"):
            return CapabilityMatchResult(
                matched=True,
                agent_id="retry_alternate",
                matched_capabilities=["acceptance.retry"],
                score=1.0,
            )
        return CapabilityMatchResult(matched=False)

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
        return [AgentStep(step_id="run", step_name="run", step_index=0)]

    async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
        _ = ctx
        return StepOutput(step_id=step.step_id, summary="recovered")

    def decide_after_step(
        self,
        step: AgentStep,
        output: StepOutput | None,
        ctx: RuntimeExecutionContext,
    ) -> AgentDecision:
        _ = step, output, ctx
        return AgentDecision(type=AgentDecisionType.COMPLETE, reason="done")


class _MemoryProducerAgent(Agent):
    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="memory_a",
            name="Memory A",
            description="acceptance memory producer",
            capabilities=["acceptance.memory_a"],
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
        return [AgentStep(step_id="write", step_name="write", step_index=0)]

    async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
        _ = step
        return StepOutput(step_id=step.step_id, summary="producer summary")

    def decide_after_step(
        self,
        step: AgentStep,
        output: StepOutput | None,
        ctx: RuntimeExecutionContext,
    ) -> AgentDecision:
        _ = step, output, ctx
        return AgentDecision(type=AgentDecisionType.COMPLETE, reason="done")


class _MemoryConsumerAgent(Agent):
    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="memory_b",
            name="Memory B",
            description="acceptance memory consumer",
            capabilities=["acceptance.memory_b"],
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
        return [AgentStep(step_id="read", step_name="read", step_index=0)]

    async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
        _ = step
        shared = load_shared_task_context_from_metadata(ctx.metadata)
        assert shared is not None
        assert "n1" in shared.structured_outputs
        summary = shared.structured_outputs["n1"]["summary"]
        return StepOutput(step_id=step.step_id, summary=f"consumer:{summary}")

    def decide_after_step(
        self,
        step: AgentStep,
        output: StepOutput | None,
        ctx: RuntimeExecutionContext,
    ) -> AgentDecision:
        _ = step, output, ctx
        return AgentDecision(type=AgentDecisionType.COMPLETE, reason="done")


class _SandboxAcceptanceAgent(Agent):
    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="sandbox_acceptance",
            name="Sandbox Acceptance",
            description="acceptance sandbox agent",
            capabilities=["acceptance.sandbox"],
            allowed_tools=[SANDBOX_TOOL_NAME],
            max_steps=1,
        )

    def can_handle(self, task_context: object) -> CapabilityMatchResult:
        capability = getattr(task_context, "capability", None)
        if capability in (None, "acceptance.sandbox"):
            return CapabilityMatchResult(
                matched=True,
                agent_id="sandbox_acceptance",
                matched_capabilities=["acceptance.sandbox"],
                score=1.0,
            )
        return CapabilityMatchResult(matched=False)

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
        return [AgentStep(step_id="sandbox", step_name="sandbox", step_index=0)]

    async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
        message = (ctx.request.message if ctx.request else "") or ""
        response = await ctx.invoke_tool(
            ToolRequest(
                tool_name=SANDBOX_TOOL_NAME,
                agent_id=ctx.agent_id,
                step_id=step.step_id,
                input={
                    "operation": "write_file",
                    "payload": {"path": "acceptance.txt", "content": message},
                },
            )
        )
        return StepOutput(step_id=step.step_id, summary=str(response.output))

    def decide_after_step(
        self,
        step: AgentStep,
        output: StepOutput | None,
        ctx: RuntimeExecutionContext,
    ) -> AgentDecision:
        _ = step, output, ctx
        return AgentDecision(type=AgentDecisionType.COMPLETE, reason="done")


class _ShadowAcceptanceAgent(Agent):
    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="shadow_acceptance",
            name="Shadow Acceptance",
            description="acceptance shadow agent",
            capabilities=["acceptance.shadow"],
            max_steps=1,
        )

    def can_handle(self, task_context: object) -> CapabilityMatchResult:
        capability = getattr(task_context, "capability", None)
        if capability in (None, "acceptance.shadow"):
            return CapabilityMatchResult(
                matched=True,
                agent_id="shadow_acceptance",
                matched_capabilities=["acceptance.shadow"],
                score=1.0,
            )
        return CapabilityMatchResult(matched=False)

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
        return [AgentStep(step_id="shadow", step_name="shadow", step_index=0)]

    async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
        workspace = ctx.metadata.get("shadow_workspace")
        message = (ctx.request.message if ctx.request else "") or ""
        if workspace is not None:
            workspace.write_text("artifact.txt", message)
        return StepOutput(step_id=step.step_id, summary=f"shadow:{message}")

    def decide_after_step(
        self,
        step: AgentStep,
        output: StepOutput | None,
        ctx: RuntimeExecutionContext,
    ) -> AgentDecision:
        _ = step, output, ctx
        return AgentDecision(type=AgentDecisionType.COMPLETE, reason="done")


@pytest.mark.asyncio
async def test_acceptance_01_single_agent_execution(echo_loop: NexusLoop):
    """Task → Agent → Result."""
    result = await echo_loop.handle_task(
        Task(
            tenant_id="t1",
            user_id="u1",
            message="acceptance single",
            context=TaskContext(capability="echo.basic"),
        )
    )
    assert result.state == TaskState.COMPLETED
    assert "acceptance single" in result.answer
    assert result.agent_id == "echo"


@pytest.mark.asyncio
async def test_acceptance_02_sequential_multi_agent():
    """Agent A → Agent B → Agent C."""
    _GraphStubAgent.run_log = []
    registry = AgentRegistry()
    registry.register(_GraphStubAgent(agent_id="a", capability="cap.a", prefix="A"))
    registry.register(_GraphStubAgent(agent_id="b", capability="cap.b", prefix="B"))
    registry.register(_GraphStubAgent(agent_id="c", capability="cap.c", prefix="C"))
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="seq",
        context=TaskContext(capability="cap.a"),
    )
    graph = ExecutionGraph(
        graph_id="acceptance_seq",
        task_id=task.task_id,
        nodes=[
            ExecutionNode(node_id="n1", agent_id="a", capability="cap.a"),
            ExecutionNode(node_id="n2", agent_id="b", capability="cap.b", depends_on=["n1"]),
            ExecutionNode(node_id="n3", agent_id="c", capability="cap.c", depends_on=["n2"]),
        ],
    )
    executions, _, final_graph, _ = await GraphExecutor(registry).execute(graph, task)
    assert len(executions) == 3
    assert executions[0].summary == "A: seq"
    assert "B: seq" in executions[1].summary
    assert "C: seq" in executions[2].summary
    assert all(node.status == ExecutionNodeStatus.COMPLETED for node in final_graph.nodes)


@pytest.mark.asyncio
async def test_acceptance_03_parallel_multi_agent():
    """Agent A, B, C in parallel batch."""
    registry = AgentRegistry()
    registry.register(_GraphStubAgent(agent_id="pa", capability="cap.pa", prefix="PA"))
    registry.register(_GraphStubAgent(agent_id="pb", capability="cap.pb", prefix="PB"))
    registry.register(_GraphStubAgent(agent_id="pc", capability="cap.pc", prefix="PC"))
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="parallel",
        context=TaskContext(capability="cap.pa"),
    )
    graph = ExecutionGraph(
        graph_id="acceptance_parallel",
        task_id=task.task_id,
        nodes=[
            ExecutionNode(node_id="n1", agent_id="pa", capability="cap.pa"),
            ExecutionNode(node_id="n2", agent_id="pb", capability="cap.pb"),
            ExecutionNode(node_id="n3", agent_id="pc", capability="cap.pc"),
        ],
    )
    batches = graph.batches()
    assert len(batches) == 1
    assert len(batches[0]) == 3

    executions, _, final_graph, _ = await GraphExecutor(registry).execute(graph, task)
    assert len(executions) == 3
    summaries = {execution.summary for execution in executions}
    assert summaries == {"PA: parallel", "PB: parallel", "PC: parallel"}
    assert all(node.status == ExecutionNodeStatus.COMPLETED for node in final_graph.nodes)


@pytest.mark.asyncio
async def test_acceptance_04_human_approval_flow(tmp_path):
    """Task → Approval → Resume."""
    registry = AgentRegistry()
    registry.register(_HitlAcceptanceAgent())
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "hitl.db")
    loop = NexusLoop(registry, checkpoint_store=store)
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="approve me",
        context=TaskContext(capability="acceptance.hitl"),
        options=TaskExecutionOptions(long_running=TaskLongRunningOptions(enabled=True)),
    )
    paused = await loop.handle_task(task)
    assert paused.state == TaskState.WAITING_FOR_HUMAN

    resumed = await loop.handle_task(
        Task(
            tenant_id="t1",
            user_id="u1",
            message="approve me",
            context=TaskContext(capability="acceptance.hitl"),
            task_id=task.task_id,
            metadata={"human_response": "approve"},
        )
    )
    assert resumed.state == TaskState.COMPLETED


@pytest.mark.asyncio
async def test_acceptance_05_checkpoint_recovery(tmp_path):
    """Task → Checkpoint → Resume."""
    registry = AgentRegistry()
    registry.register(_HitlAcceptanceAgent())
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "ckpt.db")
    loop = NexusLoop(registry, checkpoint_store=store)
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="checkpoint",
        context=TaskContext(capability="acceptance.hitl"),
        options=TaskExecutionOptions(long_running=TaskLongRunningOptions(enabled=True)),
    )
    paused = await loop.handle_task(task)
    checkpoints = store.list_for_task(task.task_id, tenant_id="t1")
    assert checkpoints

    resumed = await loop.handle_task(
        Task(
            tenant_id="t1",
            user_id="u1",
            message="checkpoint",
            context=TaskContext(capability="acceptance.hitl"),
            task_id=task.task_id,
            metadata={"human_response": "approve"},
        )
    )
    assert paused.task_id == resumed.task_id
    assert resumed.state == TaskState.COMPLETED


@pytest.mark.asyncio
async def test_acceptance_05b_mid_step_uaep_resume(tmp_path):
    """Mid-step UAEP cursor → HITL → resume without re-running phase 1."""
    _MidStepAcceptanceAgent.phase1_runs = 0
    registry = AgentRegistry()
    registry.register(_MidStepAcceptanceAgent())
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "mid_step.db")
    loop = NexusLoop(registry, checkpoint_store=store)
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="mid-step",
        context=TaskContext(capability="acceptance.mid_step"),
        options=TaskExecutionOptions(long_running=TaskLongRunningOptions(enabled=True)),
    )
    paused = await loop.handle_task(task)
    assert paused.state == TaskState.WAITING_FOR_HUMAN
    assert _MidStepAcceptanceAgent.phase1_runs == 1

    checkpoints = store.list_for_task(task.task_id, tenant_id="t1")
    assert checkpoints
    runtime = checkpoints[-1].runtime
    assert runtime is not None
    assert runtime.uaep_step_cursor == {"phase1_done": True}
    assert runtime.uaep_step_completed is False

    resumed = await loop.handle_task(
        Task(
            tenant_id="t1",
            user_id="u1",
            message="mid-step",
            context=TaskContext(capability="acceptance.mid_step"),
            task_id=task.task_id,
            metadata={"human_response": "approve"},
        )
    )
    assert resumed.state == TaskState.COMPLETED
    assert "mid-step complete" in resumed.answer
    assert _MidStepAcceptanceAgent.phase1_runs == 1

    restored_ckpt = runtime_checkpoint_from_metadata(resumed.metadata)
    assert restored_ckpt is None or restored_ckpt.uaep_step_completed or not restored_ckpt.uaep_step_cursor


@pytest.mark.asyncio
async def test_acceptance_06_retry_flow():
    """Failure → Retry → Recovery."""
    registry = AgentRegistry()
    registry.register(_RetryPrimaryAgent())
    registry.register(_RetryAlternateAgent())
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="retry",
        context=TaskContext(capability="acceptance.retry"),
    )
    graph = ExecutionGraph(
        graph_id="acceptance_retry",
        task_id=task.task_id,
        nodes=[ExecutionNode(node_id="n1", agent_id="retry_primary", capability="acceptance.retry")],
    )
    executor = GraphExecutor(
        registry,
        retry_engine=RetryEngine(registry, policy=RetryPolicy(max_retries=2)),
        validation_engine=NexusValidationEngine(),
    )
    executions, retries, final_graph, _ = await executor.execute(graph, task)
    assert executions[-1].status == AgentExecutionStatus.COMPLETED
    assert executions[-1].agent_id == "retry_alternate"
    assert "recovered" in executions[-1].summary
    assert retries
    assert final_graph.node_by_id("n1").status == ExecutionNodeStatus.COMPLETED


@pytest.mark.asyncio
async def test_acceptance_07_partial_results(tmp_path):
    """Long running task → progress events → partial results."""
    registry = AgentRegistry()
    registry.register(_HitlAcceptanceAgent())
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "progress.db")
    loop = NexusLoop(registry, checkpoint_store=store)
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="progress",
        context=TaskContext(capability="acceptance.hitl"),
        options=TaskExecutionOptions(long_running=TaskLongRunningOptions(enabled=True)),
    )
    await loop.handle_task(task)
    checkpoints = store.list_for_task(task.task_id, tenant_id="t1")
    assert checkpoints
    progress_types = {
        event.event_type
        for event in loop.event_bus.history
        if event.task_id == task.task_id
    }
    assert RuntimeEventType.TASK_PROGRESS in progress_types or RuntimeEventType.HUMAN_APPROVAL_REQUESTED in progress_types


@pytest.mark.asyncio
async def test_acceptance_08_memory_handoff():
    """Agent A → Shared Context → Agent B."""
    registry = AgentRegistry()
    registry.register(_MemoryProducerAgent())
    registry.register(_MemoryConsumerAgent())
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="memory",
        context=TaskContext(capability="acceptance.memory_a"),
    )
    graph = ExecutionGraph(
        graph_id="acceptance_memory",
        task_id=task.task_id,
        nodes=[
            ExecutionNode(node_id="n1", agent_id="memory_a", capability="acceptance.memory_a"),
            ExecutionNode(
                node_id="n2",
                agent_id="memory_b",
                capability="acceptance.memory_b",
                depends_on=["n1"],
            ),
        ],
    )
    executions, _, final_graph, _ = await GraphExecutor(registry).execute(graph, task)
    assert executions[0].summary == "producer summary"
    assert executions[1].summary == "consumer:producer summary"
    shared = load_shared_task_context_from_metadata(task.metadata)
    assert shared is not None
    assert final_graph.node_by_id("n2").status == ExecutionNodeStatus.COMPLETED


@pytest.mark.asyncio
async def test_acceptance_09_sandbox_tool_execution(tmp_path):
    """Agent → ToolRuntime → Sandbox."""
    registry = AgentRegistry()
    registry.register(_SandboxAcceptanceAgent())
    sandbox_manager = SandboxSessionManager(root=tmp_path)
    loop = NexusLoop(registry, sandbox_manager=sandbox_manager)
    result = await loop.handle_task(
        Task(
            tenant_id="t1",
            user_id="u1",
            message="sandbox-content",
            context=TaskContext(capability="acceptance.sandbox"),
            metadata={SANDBOX_FLAG: True},
        )
    )
    assert result.state == TaskState.COMPLETED
    assert result.metadata.get("sandbox_session_id")


@pytest.mark.asyncio
async def test_acceptance_10_shadow_workspace(tmp_path):
    """Agent → Shadow Workspace → Artifacts."""
    registry = AgentRegistry()
    registry.register(_ShadowAcceptanceAgent())
    shadow_manager = ShadowWorkspaceManager(root=tmp_path)
    loop = NexusLoop(registry, shadow_manager=shadow_manager)
    result = await loop.handle_task(
        Task(
            tenant_id="t1",
            user_id="u1",
            message="artifact-content",
            context=TaskContext(capability="acceptance.shadow"),
            metadata={SHADOW_WORKSPACE_FLAG: True},
        )
    )
    assert result.state == TaskState.COMPLETED
    assert result.metadata.get("shadow_workspace_id")
    workspace = shadow_manager.get(str(result.metadata["shadow_workspace_id"]))
    assert workspace is not None
    assert workspace.read_text("artifact.txt") == "artifact-content"

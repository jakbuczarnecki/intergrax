# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical multi-agent fan-out orchestration composition (NPSC-5B/R2)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Generic, TypeVar

from intergrax.agent_distribution.bounded_multi_agent_fanout import (
    FanOutItem,
    FanOutItemFailure,
    FanOutItemId,
    FanOutItemOutcome,
    FanOutItemStatus,
    FanOutRequest,
    FanOutResult,
)
from intergrax.agent_distribution.multi_agent_coordination import (
    CoordinationCleanupError,
    CoordinationError,
    CoordinationFailureCode,
    MultiAgentCoordinationService,
)
from intergrax.agents.agent_contract import Agent
from intergrax.agents.agent_engine import AgentEngine
from intergrax.agents.authoring.stub_llm import PrefixStubLLMAdapter
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.execution_identity import (
    mint_task_id,
    require_active_execution_identity,
)
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.execution.budget.ledger import ExecutionBudgetLedger
from intergrax.runtime.execution.execution_work_port import (
    ExecutionWorkPort,
    child_execution_work_port,
)
from intergrax.runtime.execution.request import ExecutionCapability, ExecutionRequest
from intergrax.runtime.execution.strategy_router import StrategyExecutionRouter
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.execution.execution_graph import (
    ExecutionGraph,
    ExecutionNode,
)
from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager
from intergrax.runtime.task.task import Task, TaskContext

FAN_OUT_SLOT_AGENT_ID = "npsc_fanout_coordination_slot"
FAN_OUT_SLOT_CAPABILITY = "npsc_fanout_slot_execute"
FAN_OUT_MERGE_AGENT_ID = "npsc_fanout_merge"
FAN_OUT_MERGE_CAPABILITY = "npsc_fanout_merge_collect"
FAN_OUT_SLOT_NODE_PREFIX = "fanout-slot-"
_ORCHESTRATION_CAPABILITIES = frozenset({ExecutionCapability.ORCHESTRATION})

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class MultiAgentFanOutOrchestrationInput(Generic[RequestT]):
    """Typed orchestration payload forwarded through ExecutionWorkPort."""

    fan_out: FanOutRequest[RequestT]
    principal: RequestIdentity


async def coordinate_fan_out_item(
    coordination: MultiAgentCoordinationService[RequestT, ResultT],
    item: FanOutItem[RequestT],
    *,
    principal: RequestIdentity,
) -> FanOutItemOutcome[ResultT]:
    try:
        coordination_result = await coordination.coordinate(
            item.request,
            delegation=item.delegation,
            principal=principal,
        )
    except CoordinationCleanupError as exc:
        return FanOutItemOutcome(
            item_id=item.item_id,
            status=FanOutItemStatus.FAILURE,
            failure=FanOutItemFailure(
                failure_code=CoordinationFailureCode.LEASE_RELEASE_FAILED,
                message=str(exc),
                partial_result=exc.result,
            ),
        )
    except CoordinationError as exc:
        return FanOutItemOutcome(
            item_id=item.item_id,
            status=FanOutItemStatus.FAILURE,
            failure=FanOutItemFailure(
                failure_code=exc.failure_code,
                message=str(exc),
            ),
        )
    return FanOutItemOutcome(
        item_id=item.item_id,
        status=FanOutItemStatus.SUCCESS,
        result=coordination_result,
    )


@dataclass(slots=True)
class _FanOutSlotExecutionBinding(Generic[RequestT, ResultT]):
    request: FanOutRequest[RequestT]
    coordination: MultiAgentCoordinationService[RequestT, ResultT]
    principal: RequestIdentity
    outcomes: dict[FanOutItemId, FanOutItemOutcome[ResultT]]
    lock: asyncio.Lock


def _fan_out_item_id_from_graph_node_id(node_id: object) -> FanOutItemId:
    if type(node_id) is not str or not node_id.startswith(FAN_OUT_SLOT_NODE_PREFIX):
        raise ValueError("fan-out slot graph node_id must use fanout-slot- prefix")
    return FanOutItemId(node_id[len(FAN_OUT_SLOT_NODE_PREFIX) :])


def _fan_out_item_by_id(
    binding: _FanOutSlotExecutionBinding[RequestT, ResultT],
    item_id: FanOutItemId,
) -> FanOutItem[RequestT]:
    for item in binding.request.items:
        if item.item_id == item_id:
            return item
    raise KeyError(f"unknown fan-out item_id: {item_id}")


class _FanOutCoordinationSlotAgent(Agent):
    __slots__ = ("_binding",)

    def __init__(
        self,
        binding: _FanOutSlotExecutionBinding[RequestT, ResultT],
    ) -> None:
        self._binding = binding

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id=FAN_OUT_SLOT_AGENT_ID,
            name=FAN_OUT_SLOT_AGENT_ID,
            description="NPSC fan-out coordination slot executor",
            capabilities=[FAN_OUT_SLOT_CAPABILITY],
        )

    def can_handle(self, task_context: TaskContext) -> CapabilityMatchResult:
        if task_context.capability == FAN_OUT_SLOT_CAPABILITY:
            return CapabilityMatchResult(
                matched=True,
                agent_id=FAN_OUT_SLOT_AGENT_ID,
                matched_capabilities=[FAN_OUT_SLOT_CAPABILITY],
                score=1.0,
            )
        return CapabilityMatchResult(matched=False)

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        config = RuntimeConfig(
            llm_adapter=PrefixStubLLMAdapter(prefix="fan-out-slot"),
            enable_rag=False,
            production_mode=False,
            tenant_id=request.tenant_id,
        )
        return RuntimeContext.build(
            config=config,
            session_manager=SessionManager(storage=InMemorySessionStorage()),
        )

    def get_steps(self, context: RuntimeContext) -> list[AgentStep]:
        del context
        return [
            AgentStep(
                step_id=f"{FAN_OUT_SLOT_AGENT_ID}_step",
                step_name=f"{FAN_OUT_SLOT_AGENT_ID}_step",
                step_index=0,
                trace_label=FAN_OUT_SLOT_CAPABILITY,
            ),
        ]

    async def run_step(
        self,
        step: AgentStep,
        ctx: RuntimeExecutionContext,
    ) -> StepOutput:
        del step
        if ctx.request is None:
            raise ValueError("fan-out slot execution requires runtime request")
        item_id = _fan_out_item_id_from_graph_node_id(
            ctx.request.metadata.get("graph_node_id"),
        )
        item = _fan_out_item_by_id(self._binding, item_id)
        outcome = await coordinate_fan_out_item(
            self._binding.coordination,
            item,
            principal=self._binding.principal,
        )
        async with self._binding.lock:
            self._binding.outcomes[item_id] = outcome
        return StepOutput(
            step_id=f"{FAN_OUT_SLOT_AGENT_ID}_step",
            summary=str(item_id),
            data={"item_id": str(item_id), "status": outcome.status.value},
        )

    def decide_after_step(
        self,
        step: AgentStep,
        output: StepOutput | None,
        ctx: RuntimeExecutionContext,
    ) -> AgentDecision:
        del step, output, ctx
        return AgentDecision(
            type=AgentDecisionType.COMPLETE,
            reason="fan-out slot complete",
        )


class _FanOutMergeAgent(Agent):
    __slots__ = ("_binding",)

    def __init__(
        self,
        binding: _FanOutSlotExecutionBinding[RequestT, ResultT],
    ) -> None:
        self._binding = binding

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id=FAN_OUT_MERGE_AGENT_ID,
            name=FAN_OUT_MERGE_AGENT_ID,
            description="NPSC fan-out deterministic fan-in merge node",
            capabilities=[FAN_OUT_MERGE_CAPABILITY],
        )

    def can_handle(self, task_context: TaskContext) -> CapabilityMatchResult:
        if task_context.capability == FAN_OUT_MERGE_CAPABILITY:
            return CapabilityMatchResult(
                matched=True,
                agent_id=FAN_OUT_MERGE_AGENT_ID,
                matched_capabilities=[FAN_OUT_MERGE_CAPABILITY],
                score=1.0,
            )
        return CapabilityMatchResult(matched=False)

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        config = RuntimeConfig(
            llm_adapter=PrefixStubLLMAdapter(prefix="fan-out-merge"),
            enable_rag=False,
            production_mode=False,
            tenant_id=request.tenant_id,
        )
        return RuntimeContext.build(
            config=config,
            session_manager=SessionManager(storage=InMemorySessionStorage()),
        )

    def get_steps(self, context: RuntimeContext) -> list[AgentStep]:
        del context
        return [
            AgentStep(
                step_id=f"{FAN_OUT_MERGE_AGENT_ID}_step",
                step_name=f"{FAN_OUT_MERGE_AGENT_ID}_step",
                step_index=0,
                trace_label=FAN_OUT_MERGE_CAPABILITY,
            ),
        ]

    async def run_step(
        self,
        step: AgentStep,
        ctx: RuntimeExecutionContext,
    ) -> StepOutput:
        del step, ctx
        expected = {item.item_id for item in self._binding.request.items}
        observed = set(self._binding.outcomes.keys())
        missing = sorted(str(item_id) for item_id in expected - observed)
        if missing:
            raise RuntimeError(
                f"fan-out merge missing slot outcomes: {', '.join(missing)}",
            )
        return StepOutput(
            step_id=f"{FAN_OUT_MERGE_AGENT_ID}_step",
            summary="fan-out merge complete",
            data={"item_count": len(expected)},
        )

    def decide_after_step(
        self,
        step: AgentStep,
        output: StepOutput | None,
        ctx: RuntimeExecutionContext,
    ) -> AgentDecision:
        del step, output, ctx
        return AgentDecision(
            type=AgentDecisionType.COMPLETE,
            reason="fan-out merge complete",
        )


def build_fan_out_execution_graph(
    request: FanOutRequest[object],
    *,
    task_id: str,
) -> ExecutionGraph:
    item_node_ids: list[str] = []
    nodes: list[ExecutionNode] = []
    for item in request.items:
        node_id = f"{FAN_OUT_SLOT_NODE_PREFIX}{item.item_id}"
        item_node_ids.append(node_id)
        nodes.append(
            ExecutionNode(
                node_id=node_id,
                agent_id=FAN_OUT_SLOT_AGENT_ID,
                capability=FAN_OUT_SLOT_CAPABILITY,
                description=f"fan-out slot {item.item_id}",
            ),
        )
    nodes.append(
        ExecutionNode(
            node_id="fanout-merge",
            agent_id=FAN_OUT_MERGE_AGENT_ID,
            capability=FAN_OUT_MERGE_CAPABILITY,
            description="fan-out deterministic fan-in",
            depends_on=item_node_ids,
        ),
    )
    return ExecutionGraph(
        graph_id=f"fanout-{request.fan_out_id}",
        task_id=task_id,
        nodes=nodes,
    )


def _project_orchestration_outcomes(
    request: FanOutRequest[RequestT],
    outcomes_by_id: dict[FanOutItemId, FanOutItemOutcome[ResultT]],
) -> tuple[FanOutItemOutcome[ResultT], ...]:
    ordered: list[FanOutItemOutcome[ResultT]] = []
    for item in request.items:
        outcome = outcomes_by_id.get(item.item_id)
        if outcome is None:
            raise RuntimeError(
                f"missing orchestration outcome for item_id: {item.item_id}",
            )
        ordered.append(outcome)
    return tuple(ordered)


class FanOutTopologyOrchestrator(Generic[RequestT, ResultT]):
    """Nexus GraphExecutor-backed bounded fan-out orchestration."""

    __slots__ = ("_coordination",)

    def __init__(
        self,
        *,
        coordination: MultiAgentCoordinationService[RequestT, ResultT],
    ) -> None:
        self._coordination = coordination

    async def execute(
        self,
        request: FanOutRequest[RequestT],
        *,
        principal: RequestIdentity,
    ) -> FanOutResult[ResultT]:
        binding = _FanOutSlotExecutionBinding(
            request=request,
            coordination=self._coordination,
            principal=principal,
            outcomes={},
            lock=asyncio.Lock(),
        )
        registry = AgentRegistry()
        registry.register(_FanOutCoordinationSlotAgent(binding))
        registry.register(_FanOutMergeAgent(binding))
        graph_executor = GraphExecutor(
            registry,
            engine=AgentEngine(registry),
            max_parallel_nodes=request.max_concurrency,
        )
        task_id = mint_task_id()
        graph = build_fan_out_execution_graph(request, task_id=str(task_id))
        task = Task(
            task_id=str(task_id),
            tenant_id="npsc-fanout",
            user_id="npsc-fanout-user",
            message=f"fan-out {request.fan_out_id}",
            context=TaskContext(capability=FAN_OUT_SLOT_CAPABILITY),
        )
        await graph_executor.execute(graph, task)
        outcomes = _project_orchestration_outcomes(request, binding.outcomes)
        return FanOutResult(
            fan_out_id=request.fan_out_id,
            items=outcomes,
        )


class FanOutOrchestrationRouterDelegate(Generic[RequestT, ResultT]):
    """Routes canonical ORCHESTRATION child work to fan-out topology orchestration."""

    __slots__ = ("_orchestrator",)

    def __init__(
        self,
        orchestrator: FanOutTopologyOrchestrator[RequestT, ResultT],
    ) -> None:
        self._orchestrator = orchestrator

    async def execute(
        self,
        request: ExecutionRequest[
            MultiAgentFanOutOrchestrationInput[RequestT],
            FanOutResult[ResultT],
        ],
    ) -> FanOutResult[ResultT]:
        return await self._orchestrator.execute(
            request.input.fan_out,
            principal=request.input.principal,
        )


class FanOutOrchestrationWorkPort(Generic[RequestT, ResultT]):
    """Composition adapter: FanOutOrchestrationPort via canonical ExecutionWorkPort."""

    __slots__ = ("_work_port",)

    def __init__(
        self,
        work_port: ExecutionWorkPort[
            MultiAgentFanOutOrchestrationInput[RequestT],
            FanOutResult[ResultT],
            FanOutResult[ResultT],
        ],
    ) -> None:
        self._work_port = work_port

    async def orchestrate_fan_out(
        self,
        request: FanOutRequest[RequestT],
        *,
        principal: RequestIdentity,
    ) -> tuple[FanOutItemOutcome[ResultT], ...]:
        result = await self._work_port.execute(
            ExecutionRequest(
                input=MultiAgentFanOutOrchestrationInput(
                    fan_out=request,
                    principal=principal,
                ),
                output_type=FanOutResult[ResultT],
                capabilities=_ORCHESTRATION_CAPABILITIES,
            ),
        )
        return result.items


def build_fan_out_orchestration_work_port(
    coordination: MultiAgentCoordinationService[RequestT, ResultT],
    *,
    ledger: ExecutionBudgetLedger | None = None,
) -> FanOutOrchestrationWorkPort[RequestT, ResultT]:
    orchestrator = FanOutTopologyOrchestrator(coordination=coordination)
    router = StrategyExecutionRouter[
        MultiAgentFanOutOrchestrationInput[RequestT],
        FanOutResult[ResultT],
        FanOutResult[ResultT],
    ](
        orchestration_executor=FanOutOrchestrationRouterDelegate(orchestrator),
    )
    work_port = child_execution_work_port(router, ledger=ledger)
    return FanOutOrchestrationWorkPort(work_port)


__all__ = [
    "FAN_OUT_MERGE_AGENT_ID",
    "FAN_OUT_SLOT_AGENT_ID",
    "FAN_OUT_SLOT_NODE_PREFIX",
    "FanOutOrchestrationRouterDelegate",
    "FanOutOrchestrationWorkPort",
    "FanOutTopologyOrchestrator",
    "MultiAgentFanOutOrchestrationInput",
    "build_fan_out_execution_graph",
    "build_fan_out_orchestration_work_port",
    "coordinate_fan_out_item",
]

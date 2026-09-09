# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical orchestration topology submission through the Nexus host."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

from intergrax.contracts.execution_identity import require_active_execution_identity
from intergrax.contracts.orchestration_topology import (
    OrchestrationResult,
    OrchestrationSchedulingPolicy,
    OrchestrationSlotContinuationError,
    OrchestrationSlotContinuationExecutor,
    OrchestrationSlotContinuationRequest,
    OrchestrationSlotExecutor,
    OrchestrationSlotId,
    OrchestrationSlotOutcome,
    OrchestrationSlotStatus,
    OrchestrationTopology,
    OrchestrationTopologyContinuationPort,
    OrchestrationTopologyExecutionId,
    OrchestrationTopologySubmissionPort,
    build_orchestration_result,
    mint_orchestration_topology_execution_id,
    validate_orchestration_scheduling_policy,
    validate_orchestration_topology,
)
from intergrax.runtime.governance.active_governed_execution_task import (
    peek_governed_execution_task,
)
from intergrax.runtime.nexus.execution.execution_graph import ExecutionGraph
from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor
from intergrax.runtime.nexus.execution.orchestration_node_execution import (
    bind_orchestration_node_execution,
    bind_orchestration_slot_continuation_execution,
)
from intergrax.runtime.nexus.execution.orchestration_topology_graph import (
    orchestration_topology_to_execution_graph,
)
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.task.task import Task, TaskContext

PayloadT = TypeVar("PayloadT")
ResultT = TypeVar("ResultT")


def _resolve_orchestration_host_task() -> Task:
    host_task = peek_governed_execution_task()
    if host_task is not None:
        return host_task
    raise RuntimeError(
        "orchestration topology submission requires an active governed host task"
    )


@dataclass
class _OrchestrationExecutionRecord(Generic[PayloadT, ResultT]):
    topology: OrchestrationTopology[PayloadT]
    scheduling_policy: OrchestrationSchedulingPolicy
    graph: ExecutionGraph
    host_task: Task
    payloads: dict[OrchestrationSlotId, PayloadT]
    outcomes: dict[OrchestrationSlotId, OrchestrationSlotOutcome[ResultT]]
    continuable_slots: set[OrchestrationSlotId] = field(default_factory=set)
    continuation_results: dict[tuple[str, str], OrchestrationSlotOutcome[ResultT]] = field(
        default_factory=dict,
    )


@dataclass
class _OrchestrationExecutionRegistry:
    records: dict[str, _OrchestrationExecutionRecord[object, object]] = field(
        default_factory=dict,
    )


@dataclass(frozen=True, slots=True)
class CanonicalOrchestrationTopologySubmissionPort(
    Generic[PayloadT, ResultT],
):
    """Submit consumer-defined topology to the canonical Nexus graph scheduler."""

    _graph_executor: GraphExecutor
    _registry: _OrchestrationExecutionRegistry = field(
        default_factory=_OrchestrationExecutionRegistry,
    )

    async def submit(
        self,
        topology: OrchestrationTopology[PayloadT],
        scheduling_policy: OrchestrationSchedulingPolicy,
        slot_executor: OrchestrationSlotExecutor[PayloadT, ResultT],
    ) -> OrchestrationResult[ResultT]:
        validate_orchestration_topology(topology)
        validate_orchestration_scheduling_policy(scheduling_policy)
        require_active_execution_identity()

        host_task = _resolve_orchestration_host_task()
        graph = orchestration_topology_to_execution_graph(
            topology,
            graph_id=f"orchestration-topology:{host_task.task_id}",
            task_id=host_task.task_id,
        )
        payloads = {slot.slot_id: slot.payload for slot in topology.slots}
        node_execution = bind_orchestration_node_execution(
            payloads=payloads,
            slot_executor=slot_executor,
        )
        result = await self._graph_executor.execute_orchestration_topology(
            graph,
            host_task,
            topology=topology,
            node_execution=node_execution,
            scheduling_policy=scheduling_policy,
        )
        execution_id = mint_orchestration_topology_execution_id(
            host_task_id=host_task.task_id,
            topology=topology,
        )
        self._registry.records[str(execution_id)] = _OrchestrationExecutionRecord(
            topology=topology,
            scheduling_policy=scheduling_policy,
            graph=graph,
            host_task=host_task,
            payloads=payloads,
            outcomes={outcome.slot_id: outcome for outcome in result.outcomes},
        )
        return result

    def register_governed_continuation_slots(
        self,
        execution_id: OrchestrationTopologyExecutionId,
        slot_ids: tuple[OrchestrationSlotId, ...],
    ) -> None:
        record = self._registry.records.get(str(execution_id))
        if record is None:
            raise OrchestrationSlotContinuationError(
                "unknown orchestration topology execution",
                code="wrong_topology",
            )
        for slot_id in slot_ids:
            if slot_id not in record.payloads:
                raise OrchestrationSlotContinuationError(
                    "slot_id not present in topology execution",
                    code="wrong_slot",
                )
            record.continuable_slots.add(slot_id)

    async def continue_slot(
        self,
        request: OrchestrationSlotContinuationRequest,
        *,
        slot_continuation_executor: OrchestrationSlotContinuationExecutor[PayloadT, ResultT],
    ) -> OrchestrationResult[ResultT]:
        record = self._registry.records.get(str(request.execution_id))
        if record is None:
            raise OrchestrationSlotContinuationError(
                "unknown orchestration topology execution",
                code="wrong_topology",
            )
        if request.slot_id not in record.payloads:
            raise OrchestrationSlotContinuationError(
                "slot_id not present in topology execution",
                code="wrong_slot",
            )

        cache_key = (str(request.slot_id), request.correlation_id)
        cached = record.continuation_results.get(cache_key)
        if cached is not None:
            updated = dict(record.outcomes)
            updated[request.slot_id] = cached
            record.outcomes = updated
            return build_orchestration_result(
                record.topology,
                outcomes_by_slot=updated,
            )

        prior = record.outcomes.get(request.slot_id)
        if prior is None:
            raise OrchestrationSlotContinuationError(
                "slot has no prior orchestration outcome",
                code="wrong_slot",
            )
        if request.slot_id not in record.continuable_slots:
            if prior.status is OrchestrationSlotStatus.SUCCESS:
                raise OrchestrationSlotContinuationError(
                    "terminal successful slot cannot be continued",
                    code="terminal_slot",
                )
            if prior.status is OrchestrationSlotStatus.SKIPPED:
                raise OrchestrationSlotContinuationError(
                    "skipped slot cannot be continued",
                    code="terminal_slot",
                )
            if prior.status is not OrchestrationSlotStatus.FAILURE:
                raise OrchestrationSlotContinuationError(
                    "slot is not eligible for governed continuation",
                    code="slot_not_continuable",
                )

        node_execution = bind_orchestration_slot_continuation_execution(
            payloads=record.payloads,
            slot_continuation_executor=slot_continuation_executor,
        )
        resumed = await self._graph_executor.continue_orchestration_topology_slot(
            record.graph,
            record.host_task,
            slot_id=request.slot_id,
            topology=record.topology,
            node_execution=node_execution,
            scheduling_policy=record.scheduling_policy,
        )
        record.continuation_results[cache_key] = resumed
        updated = dict(record.outcomes)
        updated[request.slot_id] = resumed
        record.outcomes = updated
        record.continuable_slots.discard(request.slot_id)
        return build_orchestration_result(
            record.topology,
            outcomes_by_slot=updated,
        )


def build_orchestration_topology_submission_port(
    nexus_loop: NexusLoop,
) -> OrchestrationTopologySubmissionPort[object, object]:
    """Composition-root factory wiring topology submission to the canonical Nexus host."""
    return CanonicalOrchestrationTopologySubmissionPort(
        _graph_executor=nexus_loop.graph_executor,
    )


def build_orchestration_topology_continuation_port(
    nexus_loop: NexusLoop,
) -> OrchestrationTopologyContinuationPort[object, object]:
    """Composition-root factory wiring exact slot continuation to the canonical Nexus host."""
    return CanonicalOrchestrationTopologySubmissionPort(
        _graph_executor=nexus_loop.graph_executor,
    )


def resolve_orchestration_topology_execution_id(
    *,
    host_task: Task,
    topology: OrchestrationTopology[PayloadT],
) -> OrchestrationTopologyExecutionId:
    """Resolve execution identity for a topology submitted under the active execution context."""
    return mint_orchestration_topology_execution_id(
        host_task_id=host_task.task_id,
        topology=topology,
    )


def build_orchestration_topology_host_task(
    *,
    tenant_id: str,
    user_id: str,
    task_id: str,
) -> Task:
    """Minimal governed host task carrier for topology submission proofs."""
    return Task(
        tenant_id=tenant_id,
        user_id=user_id,
        task_id=task_id,
        message="orchestration-topology-submission",
        context=TaskContext(capability="orchestration.topology"),
    )

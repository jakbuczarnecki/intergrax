# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical orchestration topology submission through the Nexus host."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from intergrax.contracts.execution_identity import require_active_execution_identity
from intergrax.contracts.orchestration_topology import (
    OrchestrationResult,
    OrchestrationSchedulingPolicy,
    OrchestrationSlotExecutor,
    OrchestrationTopology,
    OrchestrationTopologySubmissionPort,
    validate_orchestration_scheduling_policy,
    validate_orchestration_topology,
)
from intergrax.runtime.governance.active_governed_execution_task import (
    peek_governed_execution_task,
)
from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor
from intergrax.runtime.nexus.execution.orchestration_node_execution import (
    bind_orchestration_node_execution,
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


@dataclass(frozen=True, slots=True)
class CanonicalOrchestrationTopologySubmissionPort(
    Generic[PayloadT, ResultT],
):
    """Submit consumer-defined topology to the canonical Nexus graph scheduler."""

    _graph_executor: GraphExecutor

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
        return await self._graph_executor.execute_orchestration_topology(
            graph,
            host_task,
            topology=topology,
            node_execution=node_execution,
            scheduling_policy=scheduling_policy,
        )


def build_orchestration_topology_submission_port(
    nexus_loop: NexusLoop,
) -> OrchestrationTopologySubmissionPort[object, object]:
    """Composition-root factory wiring topology submission to the canonical Nexus host."""
    return CanonicalOrchestrationTopologySubmissionPort(
        _graph_executor=nexus_loop.graph_executor,
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

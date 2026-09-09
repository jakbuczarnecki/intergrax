# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Convert canonical orchestration topology contracts into Nexus execution graphs."""

from __future__ import annotations

from typing import TypeVar

from intergrax.contracts.execution_identity import TaskId, validate_task_id
from intergrax.contracts.orchestration_topology import (
    OrchestrationSlotId,
    OrchestrationTopology,
    validate_orchestration_topology,
)
from intergrax.runtime.nexus.execution.execution_graph import ExecutionGraph, ExecutionNode

PayloadT = TypeVar("PayloadT")


def orchestration_topology_to_execution_graph(
    topology: OrchestrationTopology[PayloadT],
    *,
    graph_id: str,
    task_id: TaskId | str,
) -> ExecutionGraph:
    """Materialize a typed orchestration topology into a canonical execution graph."""
    validate_orchestration_topology(topology)
    resolved_task_id = validate_task_id(task_id)
    if not graph_id or not graph_id.strip():
        raise ValueError("graph_id must be non-empty")

    nodes: list[ExecutionNode] = []
    for slot in topology.slots:
        slot_id = OrchestrationSlotId(slot.slot_id)
        nodes.append(
            ExecutionNode(
                node_id=str(slot_id),
                orchestration_slot_id=slot_id,
                depends_on=[str(dependency) for dependency in slot.depends_on],
            )
        )
    return ExecutionGraph(
        graph_id=graph_id.strip(),
        task_id=resolved_task_id,
        nodes=nodes,
    )

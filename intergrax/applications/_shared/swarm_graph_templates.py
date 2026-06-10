# © Artur Czarnecki. All rights reserved.

"""Swarm and peer-to-peer coordination graph templates (AUDIT-IDEAL-9.2)."""

from __future__ import annotations

from intergrax.applications.contracts.graph_spec import ApplicationGraphSpec, GraphEdge, GraphNode
from intergrax.runtime.architecture.multi_agent_coordination import CoordinationPattern


def swarm_exploration_graph_template(
    *,
    worker_agent_ids: tuple[str, ...],
    aggregator_agent_id: str,
) -> ApplicationGraphSpec:
    """Parallel swarm workers converge on a single aggregator node."""
    nodes = [GraphNode(agent_id=agent_id) for agent_id in worker_agent_ids]
    nodes.append(GraphNode(agent_id=aggregator_agent_id))
    edges = [
        GraphEdge(source_agent_id=worker_id, target_agent_id=aggregator_agent_id)
        for worker_id in worker_agent_ids
    ]
    return ApplicationGraphSpec(
        nodes=nodes,
        edges=edges,
        retry_on_error=1,
        trigger_capabilities=[f"{CoordinationPattern.SWARM.value}.pipeline"],
    )


def peer_to_peer_graph_template(
    *,
    agent_ids: tuple[str, ...],
) -> ApplicationGraphSpec:
    """Ring-style peer coordination template for negotiation flows."""
    if len(agent_ids) < 2:
        raise ValueError("peer_to_peer template requires at least two agents")
    nodes = [GraphNode(agent_id=agent_id) for agent_id in agent_ids]
    edges = [
        GraphEdge(source_agent_id=agent_ids[index], target_agent_id=agent_ids[(index + 1) % len(agent_ids)])
        for index in range(len(agent_ids))
    ]
    return ApplicationGraphSpec(
        nodes=nodes,
        edges=edges,
        trigger_capabilities=[f"{CoordinationPattern.PEER_TO_PEER.value}.pipeline"],
    )

# © Artur Czarnecki. All rights reserved.

"""Declarative Nexus graphs for dispute_sim_application (CFG-06 / CFG-07)."""

from __future__ import annotations

from intergrax.applications.contracts.graph_spec import (
    ApplicationGraphSpec,
    GraphEdge,
    GraphEdgeKind,
    GraphNode,
)

# User scenario: corporate evidence (analyst / RAG) → legal web + correspondence (scenario).
DISPUTE_SIM_CORRESPONDENCE_GRAPH = ApplicationGraphSpec(
    nodes=[
        GraphNode(agent_id="dispute_analyst"),
        GraphNode(agent_id="dispute_scenario"),
    ],
    edges=[
        GraphEdge(
            source_agent_id="dispute_analyst",
            target_agent_id="dispute_scenario",
            kind=GraphEdgeKind.DEPENDS_ON,
        ),
    ],
    trigger_capabilities=["dispute.pipeline", "dispute.correspondence"],
)

# Full dispute lifecycle (intake → analyze → strategy → scenario).
DISPUTE_SIM_FULL_PIPELINE_GRAPH = ApplicationGraphSpec(
    nodes=[
        GraphNode(agent_id="dispute_intake"),
        GraphNode(agent_id="dispute_analyst"),
        GraphNode(agent_id="dispute_strategist"),
        GraphNode(agent_id="dispute_scenario"),
    ],
    edges=[
        GraphEdge(
            source_agent_id="dispute_intake",
            target_agent_id="dispute_analyst",
            kind=GraphEdgeKind.DEPENDS_ON,
        ),
        GraphEdge(
            source_agent_id="dispute_analyst",
            target_agent_id="dispute_strategist",
            kind=GraphEdgeKind.DEPENDS_ON,
        ),
        GraphEdge(
            source_agent_id="dispute_strategist",
            target_agent_id="dispute_scenario",
            kind=GraphEdgeKind.DEPENDS_ON,
        ),
    ],
    trigger_capabilities=["dispute.full_pipeline"],
)

DEFAULT_DISPUTE_SIM_GRAPH = DISPUTE_SIM_CORRESPONDENCE_GRAPH

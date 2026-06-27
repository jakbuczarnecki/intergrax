from __future__ import annotations

from intergrax.runtime.architecture.capability_graph import build_catalog_capability_graph
from intergrax.runtime.architecture.capability_graph_applications import application_capability_node_id


def test_application_agent_edges_are_scoped_per_manifest() -> None:
    graph = build_catalog_capability_graph()
    edge_keys = {
        (edge.source_node_id, edge.target_node_id)
        for edge in graph.edges
        if edge.source_node_id.startswith("application:")
    }

    legal_node = application_capability_node_id("legal")
    lab_node = application_capability_node_id("lab")

    assert (legal_node, "agent:legal") in edge_keys
    assert (lab_node, "agent:echo") in edge_keys
    assert (legal_node, "agent:echo") not in edge_keys
    assert (lab_node, "agent:legal") not in edge_keys


def test_research_agents_link_to_research_application_not_lab() -> None:
    graph = build_catalog_capability_graph()
    edge_keys = {
        (edge.source_node_id, edge.target_node_id)
        for edge in graph.edges
        if edge.source_node_id.startswith("application:")
    }
    research_node = application_capability_node_id("research")
    lab_node = application_capability_node_id("lab")

    assert (research_node, "agent:research") in edge_keys
    assert (research_node, "agent:research-summary") in edge_keys
    assert (lab_node, "agent:research") not in edge_keys

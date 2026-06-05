from __future__ import annotations

from intergrax.runtime.architecture.capability_graph import build_catalog_capability_graph
from intergrax.runtime.architecture.capability_graph_applications import (
    application_capability_node_id,
    catalog_application_manifests,
)


def test_application_agent_edges_are_scoped_per_manifest() -> None:
    graph = build_catalog_capability_graph()
    edge_keys = {
        (edge.source_node_id, edge.target_node_id)
        for edge in graph.edges
        if edge.source_node_id.startswith("application:")
    }

    legal_manifest = next(m for m in catalog_application_manifests() if m.app_id == "legal")
    legal_node = application_capability_node_id(legal_manifest)
    lab_manifest = next(m for m in catalog_application_manifests() if m.app_id == "lab")
    lab_node = application_capability_node_id(lab_manifest)

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
    research_manifest = next(m for m in catalog_application_manifests() if m.app_id == "research")
    research_node = application_capability_node_id(research_manifest)
    lab_manifest = next(m for m in catalog_application_manifests() if m.app_id == "lab")
    lab_node = application_capability_node_id(lab_manifest)

    assert (research_node, "agent:research") in edge_keys
    assert (research_node, "agent:research-summary") in edge_keys
    assert (lab_node, "agent:research") not in edge_keys

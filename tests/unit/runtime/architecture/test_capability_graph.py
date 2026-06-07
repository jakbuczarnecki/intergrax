from __future__ import annotations

import pytest

from intergrax.runtime.architecture.capability_graph import (
    CapabilityEdge,
    CapabilityEdgeType,
    CapabilityGraph,
    CapabilityNode,
    CapabilityNodeType,
    build_catalog_capability_graph,
)


def test_capability_graph_rejects_duplicate_node_id() -> None:
    with pytest.raises(ValueError, match="duplicate node_id"):
        CapabilityGraph(
            nodes=[
                CapabilityNode(node_id="tool:one", node_type=CapabilityNodeType.TOOL),
                CapabilityNode(node_id="tool:one", node_type=CapabilityNodeType.TOOL),
            ],
            edges=[],
        )


def test_capability_graph_rejects_missing_edge_target() -> None:
    with pytest.raises(ValueError, match="Edge target is not present"):
        CapabilityGraph(
            nodes=[
                CapabilityNode(node_id="tool:one", node_type=CapabilityNodeType.TOOL),
            ],
            edges=[
                CapabilityEdge(
                    source_node_id="tool:one",
                    target_node_id="integration:missing",
                    edge_type=CapabilityEdgeType.DEPENDS_ON,
                )
            ],
        )


def test_capability_graph_rejects_invalid_relation() -> None:
    with pytest.raises(ValueError, match="Invalid edge relation"):
        CapabilityGraph(
            nodes=[
                CapabilityNode(node_id="tool:one", node_type=CapabilityNodeType.TOOL),
                CapabilityNode(node_id="agent:one", node_type=CapabilityNodeType.AGENT),
            ],
            edges=[
                CapabilityEdge(
                    source_node_id="tool:one",
                    target_node_id="agent:one",
                    edge_type=CapabilityEdgeType.DEPENDS_ON,
                )
            ],
        )


def test_build_catalog_capability_graph_includes_modality_compatibility_edges() -> None:
    graph = build_catalog_capability_graph()
    edge_keys = {(edge.source_node_id, edge.target_node_id) for edge in graph.edges}
    assert ("tool:vision.detect", "tool:rag.retrieve") in edge_keys
    assert ("tool:rag.retrieve", "tool:vision.detect") in edge_keys
    assert ("tool:storage.get", "tool:knowledge.search") in edge_keys
    assert ("tool:workspace.write_file", "tool:memory.write") in edge_keys


def test_build_catalog_capability_graph_returns_typed_graph() -> None:
    graph = build_catalog_capability_graph()
    assert graph.nodes
    assert graph.edges
    assert any(node.node_type == CapabilityNodeType.INTEGRATION for node in graph.nodes)
    assert any(node.node_type == CapabilityNodeType.TOOL for node in graph.nodes)
    assert any(node.node_type == CapabilityNodeType.SKILL for node in graph.nodes)

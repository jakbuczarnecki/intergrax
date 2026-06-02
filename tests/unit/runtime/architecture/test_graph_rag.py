from __future__ import annotations

import pytest

from intergrax.runtime.architecture.graph_rag import (
    GraphRagArchitectureContract,
    GraphRagEdge,
    GraphRagEdgeType,
    GraphRagNode,
    GraphRagNodeType,
)


def test_graph_rag_contract_validates_edge_endpoints() -> None:
    contract = GraphRagArchitectureContract(
        graph_id="g1",
        nodes=[GraphRagNode(node_id="n1", node_type=GraphRagNodeType.DOCUMENT, label="Doc")],
        edges=[
            GraphRagEdge(
                source_node_id="n1",
                target_node_id="n1",
                edge_type=GraphRagEdgeType.RELATED_TO,
            )
        ],
    )
    assert contract.graph_id == "g1"


def test_graph_rag_contract_rejects_unknown_edge_node() -> None:
    with pytest.raises(ValueError, match="Unknown target node"):
        GraphRagArchitectureContract(
            graph_id="g1",
            nodes=[GraphRagNode(node_id="n1", node_type=GraphRagNodeType.DOCUMENT, label="Doc")],
            edges=[
                GraphRagEdge(
                    source_node_id="n1",
                    target_node_id="missing",
                    edge_type=GraphRagEdgeType.RELATED_TO,
                )
            ],
        )

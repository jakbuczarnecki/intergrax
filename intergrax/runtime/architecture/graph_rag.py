# © Artur Czarnecki. All rights reserved.

"""Graph-RAG architecture contracts (Phase V-KG.1)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class GraphRagNodeType(str, Enum):
    DOCUMENT = "document"
    ENTITY = "entity"
    CONCEPT = "concept"
    AGENT = "agent"
    TOOL = "tool"


class GraphRagEdgeType(str, Enum):
    REFERENCES = "references"
    DERIVED_FROM = "derived_from"
    RELATED_TO = "related_to"
    EXECUTED_BY = "executed_by"


class GraphRagNode(BaseModel):
    node_id: str
    node_type: GraphRagNodeType
    label: str


class GraphRagEdge(BaseModel):
    source_node_id: str
    target_node_id: str
    edge_type: GraphRagEdgeType


class GraphRagArchitectureContract(BaseModel):
    schema_version: str = "1.0.0"
    graph_id: str
    nodes: list[GraphRagNode] = Field(default_factory=list)
    edges: list[GraphRagEdge] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_edge_endpoints(self) -> "GraphRagArchitectureContract":
        node_ids = {node.node_id for node in self.nodes}
        for edge in self.edges:
            if edge.source_node_id not in node_ids:
                raise ValueError(f"Unknown source node: {edge.source_node_id}")
            if edge.target_node_id not in node_ids:
                raise ValueError(f"Unknown target node: {edge.target_node_id}")
        return self

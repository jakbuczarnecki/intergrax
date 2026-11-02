# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from testing_support.change_safety.discovery.closed_world import ClosedWorldInventory
from testing_support.change_safety.discovery.extractors import GraphEdge, GraphNode
from testing_support.change_safety.graph.digest import semantic_graph_digest


@dataclass(frozen=True)
class SemanticGraph:
    schema_version: str
    source_commit_sha: str
    ruleset_version: str
    inventory_digest: str
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]

    @cached_property
    def digest(self) -> str:
        return semantic_graph_digest(
            schema_version=self.schema_version,
            inventory_digest=self.inventory_digest,
            nodes=self.nodes,
            edges=self.edges,
        )

    def rel_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for edge in self.edges:
            counts[edge.rel] = counts.get(edge.rel, 0) + 1
        return dict(sorted(counts.items()))


def build_semantic_graph(inventory: ClosedWorldInventory) -> SemanticGraph:
    return SemanticGraph(
        schema_version="evol_semantic_graph.v1",
        source_commit_sha=inventory.source_commit_sha,
        ruleset_version=inventory.ruleset_version,
        inventory_digest=inventory.inventory_digest,
        nodes=inventory.nodes,
        edges=inventory.edges,
    )

# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

from testing_support.change_safety.contracts.canonical_json import stable_digest_hex
from testing_support.change_safety.discovery.extractors import (
    GraphEdge,
    GraphNode,
    build_import_edges,
    build_inventory_nodes,
)


@dataclass(frozen=True)
class ClosedWorldInventory:
    schema_version: str
    ruleset_version: str
    source_commit_sha: str
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]

    @property
    def inventory_digest(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "ruleset_version": self.ruleset_version,
            "source_commit_sha": self.source_commit_sha,
            "nodes": [n.__dict__ for n in self.nodes],
            "edges": [e.__dict__ for e in self.edges],
        }
        return stable_digest_hex(payload)


def build_closed_world_inventory(
    *, source_commit_sha: str, ruleset_version: str = "ruleset.v1"
) -> ClosedWorldInventory:
    return ClosedWorldInventory(
        schema_version="evol_closed_world_inventory.v1",
        ruleset_version=ruleset_version,
        source_commit_sha=source_commit_sha,
        nodes=build_inventory_nodes(),
        edges=build_import_edges(),
    )

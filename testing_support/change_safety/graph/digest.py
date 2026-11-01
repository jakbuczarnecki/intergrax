# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from testing_support.change_safety.contracts.canonical_json import stable_digest_hex
from testing_support.change_safety.discovery.extractors import GraphEdge, GraphNode


def semantic_graph_digest(
    *,
    schema_version: str,
    inventory_digest: str,
    nodes: tuple[GraphNode, ...],
    edges: tuple[GraphEdge, ...],
) -> str:
    payload = {
        "schema_version": schema_version,
        "inventory_digest": inventory_digest,
        "nodes": [n.__dict__ for n in nodes],
        "edges": [e.__dict__ for e in edges],
    }
    return stable_digest_hex(payload)

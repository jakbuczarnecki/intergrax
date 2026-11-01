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
    """Canonical semantic graph digest: node/edge content (incl. REL) bound to the inventory digest (XG-03)."""
    payload = {
        "schema_version": schema_version,
        "inventory_digest": inventory_digest,
        "nodes": [[n.node_id, n.category] for n in nodes],
        "edges": [[e.source_id, e.target_id, e.rel] for e in edges],
    }
    return stable_digest_hex(payload)

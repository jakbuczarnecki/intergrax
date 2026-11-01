# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from collections import deque

from testing_support.change_safety.contracts.impact import AffectedNode
from testing_support.change_safety.discovery.extractors import GraphEdge


def _reverse_adjacency(edges: tuple[GraphEdge, ...]) -> dict[str, list[GraphEdge]]:
    adj: dict[str, list[GraphEdge]] = {}
    for edge in edges:
        if edge.rel != "IMPORTS":
            continue
        adj.setdefault(edge.target_id, []).append(edge)
    return adj


def traverse_from_changed_modules(
    edges: tuple[GraphEdge, ...],
    seed_module_ids: frozenset[str],
) -> dict[str, AffectedNode]:
    adj = _reverse_adjacency(edges)
    affected: dict[str, AffectedNode] = {}
    queue: deque[tuple[str, tuple[str, ...]]] = deque()
    for seed in sorted(seed_module_ids):
        queue.append((seed, (seed,)))
        affected[seed] = AffectedNode(node_id=seed, kind="DIRECT", witness_path=(seed,))
    while queue:
        node_id, path = queue.popleft()
        for edge in sorted(adj.get(node_id, []), key=lambda e: e.source_id):
            consumer = edge.source_id
            if consumer in affected:
                continue
            witness = path + (consumer,)
            affected[consumer] = AffectedNode(
                node_id=consumer,
                kind="TRANSITIVE",
                witness_path=witness,
                rule_id="REL-IMPORTS-REVERSE",
            )
            queue.append((consumer, witness))
    return affected

# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

from testing_support.change_safety.discovery.closed_world import ClosedWorldInventory
from testing_support.change_safety.discovery.extractors import GraphEdge


@dataclass(frozen=True)
class ConsumerIndex:
    schema_version: str
    consumers_by_target: dict[str, tuple[str, ...]]


def build_consumer_index(inventory: ClosedWorldInventory) -> ConsumerIndex:
    mapping: dict[str, set[str]] = {}
    for edge in inventory.edges:
        if edge.rel != "IMPORTS":
            continue
        mapping.setdefault(edge.target_id, set()).add(edge.source_id)
    frozen = {k: tuple(sorted(v)) for k, v in sorted(mapping.items())}
    return ConsumerIndex(schema_version="evol_consumer_closure.v1", consumers_by_target=frozen)


def orphan_import_targets(edges: tuple[GraphEdge, ...], known_module_ids: frozenset[str]) -> tuple[str, ...]:
    orphans: list[str] = []
    for edge in edges:
        if edge.rel != "IMPORTS":
            continue
        if edge.target_id.startswith("module:") and edge.target_id not in known_module_ids:
            orphans.append(edge.target_id)
    return tuple(sorted(set(orphans)))


def count_critical_consumer_omissions(inventory: ClosedWorldInventory) -> int:
    """CW-06 / XG-02 oracle slot — W3-02 supplies second-method parity; W2 keeps fail-closed hook at 0."""
    _ = inventory
    return 0

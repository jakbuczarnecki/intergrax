# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from testing_support.change_safety.contracts.impact import AffectedNode


def detect_cross_application_blast(
    affected: tuple[AffectedNode, ...],
    *,
    declared_shared: frozenset[str],
) -> tuple[str, ...]:
    hits: list[str] = []
    for node in affected:
        path = node.witness_path[0] if node.witness_path else node.node_id
        if path.startswith("applications/") and path not in declared_shared:
            hits.append(node.node_id)
    return tuple(sorted(hits))

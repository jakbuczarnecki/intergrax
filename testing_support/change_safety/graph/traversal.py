# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from collections import deque

from testing_support.change_safety.contracts.impact import AffectedNode
from testing_support.change_safety.discovery.closed_world import ClosedWorldInventory


def traverse_from_changed_modules(
    inventory: ClosedWorldInventory,
    seed_module_ids: frozenset[str],
) -> dict[str, AffectedNode]:
    """Reverse traversal over all consumption RELs, then forward TESTED_BY to suites."""
    adj = inventory.consumers_reverse
    affected: dict[str, AffectedNode] = {}
    queue: deque[tuple[str, tuple[str, ...]]] = deque()
    for seed in sorted(seed_module_ids):
        queue.append((seed, (seed,)))
        affected[seed] = AffectedNode(node_id=seed, kind="DIRECT", witness_path=(seed,))
    while queue:
        node_id, path = queue.popleft()
        for consumer, rel in adj.get(node_id, ()):
            if consumer in affected:
                continue
            witness = path + (consumer,)
            affected[consumer] = AffectedNode(
                node_id=consumer,
                kind="TRANSITIVE",
                witness_path=witness,
                rule_id=f"REL-{rel}-REVERSE",
            )
            queue.append((consumer, witness))
    for node_id in sorted(list(affected)):
        node = affected[node_id]
        for suite in inventory.tested_by.get(node_id, ()):
            if suite not in affected:
                affected[suite] = AffectedNode(
                    node_id=suite,
                    kind="TRANSITIVE",
                    witness_path=node.witness_path + (suite,),
                    rule_id="REL-TESTED_BY",
                )
    return affected

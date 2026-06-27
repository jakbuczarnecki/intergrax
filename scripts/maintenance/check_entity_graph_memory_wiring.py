#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-15.3 — entity graph memory wiring gate."""

from __future__ import annotations

import sys

from intergrax.applications._shared.entity_graph_wiring import resolve_entity_graph_memory_store
from intergrax.applications._shared.memory_wiring import resolve_memory_platform_wiring
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.memory.entity_graph_memory import EntityGraphMemoryStore, EntityNode


def main() -> int:
    env = ApplicationEnvironmentProfile.product_defaults()
    if not env.memory_profile.enable_entity_graph_memory:
        print("product_defaults must enable entity graph memory", file=sys.stderr)
        return 1
    store = resolve_entity_graph_memory_store(env)
    if store is None:
        print("entity graph store must resolve for product profile", file=sys.stderr)
        return 1
    wiring = resolve_memory_platform_wiring(env)
    if wiring.entity_graph_store is None:
        print("memory platform wiring must include entity graph store", file=sys.stderr)
        return 1

    store.upsert_node(EntityNode(entity_id="e1", label="Acme", entity_type="org"))
    if not isinstance(store, EntityGraphMemoryStore):
        print("unexpected entity graph store type", file=sys.stderr)
        return 1

    print("OK: entity graph memory wiring")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

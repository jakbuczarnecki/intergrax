# © Artur Czarnecki. All rights reserved.

"""Entity graph memory wiring (AUDIT-IDEAL-15.3)."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.memory.entity_graph_memory import EntityGraphMemoryStore


def resolve_entity_graph_memory_store(
    env: ApplicationEnvironmentProfile,
) -> EntityGraphMemoryStore | None:
    """Return entity graph store when memory profile enables entity graph memory."""
    if not env.memory_profile.enable_entity_graph_memory:
        return None
    return EntityGraphMemoryStore()

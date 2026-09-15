# © Artur Czarnecki. All rights reserved.

"""Entity graph memory wiring (MEM-ENT-7)."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.memory.contracts.entity_temporal_memory import EntityTemporalMemoryStore
from intergrax.memory.entity_graph_memory import EntityGraphMemoryStore
from intergrax.memory.stores.in_memory_entity_temporal_memory_store import (
    InMemoryEntityTemporalMemoryStore,
)


def resolve_entity_temporal_memory_store(
    env: ApplicationEnvironmentProfile,
) -> EntityTemporalMemoryStore | None:
    """Return entity/temporal store when memory profile enables entity graph memory."""
    if not env.memory_profile.enable_entity_graph_memory:
        return None
    return InMemoryEntityTemporalMemoryStore()


def resolve_entity_graph_memory_store(
    env: ApplicationEnvironmentProfile,
) -> EntityGraphMemoryStore | None:
    """Legacy facade for composition roots that still expect ``EntityGraphMemoryStore``."""
    backend = resolve_entity_temporal_memory_store(env)
    if backend is None:
        return None
    return EntityGraphMemoryStore(backend=backend)

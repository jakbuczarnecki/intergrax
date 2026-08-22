# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Reference IdempotencyStore providers selected by persistence topology."""

from __future__ import annotations

from pathlib import Path

from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.contracts.persistence_topology import PersistenceTopology
from intergrax.integrations.providers.relational_store.sqlite.bundle import (
    create_sqlite_idempotency_store,
)
from intergrax.integrations.providers.relational_store.sqlite.paths import (
    resolve_idempotency_db_path,
)
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore


def resolve_reference_idempotency_store(
    required_topology: PersistenceTopology,
    *,
    db_path: Path | None = None,
) -> IdempotencyStore | None:
    """
    Materialize a reference IdempotencyStore for ``required_topology``.

    PROCESS_LOCAL → in-process reference provider.
    DURABLE_SINGLE_HOST → durable single-host reference provider (SQLite).
    SHARED_MULTI_HOST → None; caller must inject a qualifying shared provider.
    """
    if db_path is not None:
        resolved_path = resolve_idempotency_db_path(db_path)
        store = create_sqlite_idempotency_store(db_path=resolved_path)
        assert isinstance(store, IdempotencyStore)
        return store

    if required_topology is PersistenceTopology.DURABLE_SINGLE_HOST:
        resolved_path = resolve_idempotency_db_path(None)
        store = create_sqlite_idempotency_store(db_path=resolved_path)
        assert isinstance(store, IdempotencyStore)
        return store

    if required_topology is PersistenceTopology.PROCESS_LOCAL:
        return InMemoryIdempotencyStore()

    return None

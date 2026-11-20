# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""RuntimeEvent store opener and persistence resolution (§42.24)."""

from __future__ import annotations

import os
from pathlib import Path

from intergrax.contracts.execution_evidence.persistence_port import EvidencePersistencePort
from intergrax.integrations.providers.relational_store.sqlite.paths import (
    DEFAULT_RUNTIME_EVENTS_DB,
    ENV_RUNTIME_EVENTS_DB,
    resolve_runtime_events_db_path,
)
from intergrax.runtime.events.persistence_contract import (
    NullRuntimeEventPersistence,
    RuntimeEventPersistence,
)
from intergrax.runtime.events.stores.sqlite_runtime_event_store import (
    SQLiteRuntimeEventStore,
)
from intergrax.runtime.events.stores.validating_runtime_event_store import (
    ValidatingRuntimeEventPersistence,
)

__all__ = [
    "DEFAULT_RUNTIME_EVENTS_DB",
    "ENV_RUNTIME_EVENTS_DB",
    "resolve_runtime_events_db_path",
    "open_runtime_event_store",
    "resolve_runtime_event_persistence",
    "RuntimeEventPersistenceNotConfiguredError",
]


class RuntimeEventPersistenceNotConfiguredError(RuntimeError):
    """PRODUCT/STRICT composition has neither RuntimeEvent persistence, a path nor the explicit acknowledgement."""


def open_runtime_event_store(db_path: Path | None = None) -> SQLiteRuntimeEventStore:
    """Open SQLite runtime event store via runtime persistence composition."""
    from intergrax.runtime.persistence.sqlite_composition import (
        create_sqlite_runtime_event_store,
    )

    if db_path is not None:
        return create_sqlite_runtime_event_store(db_path=db_path)  # type: ignore[return-value]
    return create_sqlite_runtime_event_store()  # type: ignore[return-value]


def resolve_runtime_event_persistence(
    *,
    db_path: Path | None = None,
    implementation: EvidencePersistencePort | RuntimeEventPersistence | None = None,
    production_mode: bool = False,
    disabled: bool = False,
) -> EvidencePersistencePort | RuntimeEventPersistence | None:
    """
    Resolve RuntimeEvent persistence for composition roots (NexusLoop, debug API, tests).

    Priority: explicit ``implementation`` > SQLite at ``db_path``/env path.
    Pass ``implementation=InMemoryRuntimeEventStore()`` in unit tests.
    Pass ``implementation=None`` and omit path to disable persistence.

    With ``production_mode=True`` the ambient ``ENV_RUNTIME_EVENTS_DB`` is never read: without an
    ``implementation`` or ``db_path`` the call raises
    :class:`RuntimeEventPersistenceNotConfiguredError` unless the caller states ``disabled=True``
    (typed acknowledgement, result ``None``; a bus that already persists is such a case).
    """
    if implementation is not None:
        if isinstance(
            implementation,
            (ValidatingRuntimeEventPersistence, NullRuntimeEventPersistence),
        ):
            return implementation
        if isinstance(implementation, RuntimeEventPersistence):
            return ValidatingRuntimeEventPersistence(implementation)
        return implementation
    if production_mode:
        if db_path is not None:
            return ValidatingRuntimeEventPersistence(open_runtime_event_store(db_path))
        if disabled:
            return None
        raise RuntimeEventPersistenceNotConfiguredError(
            "production_mode requires RuntimeEvent persistence or a db_path, or an explicit disable "
            "acknowledgement; the ambient environment variable is not read.",
        )
    if db_path is None and not os.environ.get(ENV_RUNTIME_EVENTS_DB, "").strip():
        return None
    return ValidatingRuntimeEventPersistence(open_runtime_event_store(db_path))

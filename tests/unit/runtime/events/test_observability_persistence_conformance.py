# © Artur Czarnecki. All rights reserved.

"""OBS-BUS-5 persistence conformance — shared contract across backends."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.providers.document_store.cassandra.runtime_events import (
    runtime_event_persistence_from_document_store,
)
from intergrax.integrations.providers.observability_backend.elasticsearch.runtime_events import (
    runtime_event_persistence_for_elasticsearch_lab,
)
from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.events.stores.document_backed_runtime_event_store import (
    DocumentBackedRuntimeEventStore,
)
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.events.stores.sqlite_runtime_event_store import SQLiteRuntimeEventStore
from intergrax.runtime.observability.persistence_conformance import (
    assert_runtime_event_persistence_conformance,
)

pytestmark = pytest.mark.gate


@pytest.fixture
def sqlite_store(tmp_path: Path) -> SQLiteRuntimeEventStore:
    return SQLiteRuntimeEventStore(db_path=tmp_path / "runtime_events.db")


@pytest.mark.parametrize(
    ("label", "factory"),
    [
        ("memory", lambda: InMemoryRuntimeEventStore()),
        (
            "document",
            lambda: DocumentBackedRuntimeEventStore(InMemoryDocumentStore()),
        ),
        (
            "cassandra_doc",
            lambda: runtime_event_persistence_from_document_store(InMemoryDocumentStore()),
        ),
        (
            "elasticsearch_lab",
            lambda: runtime_event_persistence_for_elasticsearch_lab(),
        ),
    ],
)
def test_runtime_event_persistence_conformance_matrix(
    label: str,
    factory,
) -> None:
    store: RuntimeEventPersistence = factory()
    try:
        assert_runtime_event_persistence_conformance(store, label=label)
    finally:
        store.close()


def test_sqlite_runtime_event_persistence_conformance(sqlite_store: SQLiteRuntimeEventStore) -> None:
    assert_runtime_event_persistence_conformance(sqlite_store, label="sqlite")
    sqlite_store.close()

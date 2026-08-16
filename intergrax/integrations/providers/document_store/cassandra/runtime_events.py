# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Cassandra-backed ``RuntimeEventPersistence`` via ``ConditionalDocumentStore`` (OBS-BUS-5)."""

from __future__ import annotations

from intergrax.integrations.contracts.document_store import ConditionalDocumentStore
from intergrax.integrations.providers.document_store.cassandra.adapter import _CassandraDocumentStore
from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.events.stores.document_backed_runtime_event_store import (
    DocumentBackedRuntimeEventStore,
)


def runtime_event_persistence_from_document_store(
    store: ConditionalDocumentStore,
) -> RuntimeEventPersistence:
    """Wrap any ``ConditionalDocumentStore`` as canonical runtime event persistence."""
    return DocumentBackedRuntimeEventStore(store)


def runtime_event_persistence_from_cassandra(
    store: _CassandraDocumentStore,
) -> RuntimeEventPersistence:
    """Factory used by ``IntegrationProfile`` when ``document_store=cassandra``."""
    return runtime_event_persistence_from_document_store(store)

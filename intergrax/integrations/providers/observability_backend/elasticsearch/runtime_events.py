# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Elasticsearch-aligned runtime event persistence (OBS-BUS-5).

Primary journal remains ``DocumentStore`` / Cassandra for append throughput.
Elasticsearch receives the same canonical ``RuntimeEvent`` envelope for search
(dual-write export ships in OBS-BUS-6). Lab profile uses an in-memory document
index implementing the same ``RuntimeEventPersistence`` contract.
"""

from __future__ import annotations

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.integrations.providers.observability_backend.elasticsearch.adapter import (
    ElasticsearchObservabilityBackend,
)
from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.events.stores.document_backed_runtime_event_store import (
    DocumentBackedRuntimeEventStore,
)


def runtime_event_persistence_from_document_store(
    store: DocumentStore,
) -> RuntimeEventPersistence:
    return DocumentBackedRuntimeEventStore(store)


def runtime_event_persistence_for_elasticsearch_lab(
    *,
    document_store: DocumentStore | None = None,
) -> RuntimeEventPersistence:
    """
    Lab/test persistence implementing the same protocol as production ES index path.

    Pass an explicit ``document_store`` to share storage with conformance harnesses.
    """
    store = document_store or InMemoryDocumentStore()
    return runtime_event_persistence_from_document_store(store)


def runtime_event_persistence_from_elasticsearch_backend(
    backend: ElasticsearchObservabilityBackend,
    *,
    document_store: DocumentStore | None = None,
) -> RuntimeEventPersistence:
    """
    Resolve runtime event persistence for ``observability_backend=elasticsearch``.

    The ``backend`` is retained for OBS-BUS-6 export/search wiring; persistence uses
  the document index contract until bulk index hooks land.
    """
    _ = backend
    return runtime_event_persistence_for_elasticsearch_lab(document_store=document_store)

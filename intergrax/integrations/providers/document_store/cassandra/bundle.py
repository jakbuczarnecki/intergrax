# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Complete Cassandra integration bundle — the single composition root for Cassandra in Intergrax.

Driver sessions are opened only in ``opens.py``. Tier-3 code MUST use
``create_cassandra_document_store()``, ``create_cassandra_integration()``, or
``profile.resolve(IntegrationCategory.DOCUMENT_STORE)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.integrations.providers.document_store.cassandra.adapter import CassandraDocumentStore
from intergrax.integrations.providers.document_store.cassandra.client import CassandraCqlClient
from intergrax.integrations.providers.document_store.cassandra.config import CassandraIntegrationConfig
from intergrax.integrations.providers.document_store.cassandra.opens import open_cassandra_document_store


@dataclass(frozen=True)
class CassandraIntegrationBundle:
    config: CassandraIntegrationConfig
    document_store: CassandraDocumentStore
    cql_client: CassandraCqlClient


def resolve_cassandra_config(**overrides: object) -> CassandraIntegrationConfig:
    return CassandraIntegrationConfig.from_env(**overrides)


def create_cassandra_integration(
    *,
    document_store: Optional[DocumentStore] = None,
    session: Optional[object] = None,
    session_factory: Optional[Callable[[], object]] = None,
    **config_overrides: object,
) -> CassandraIntegrationBundle:
    config = resolve_cassandra_config(**config_overrides)
    store = open_cassandra_document_store(
        config,
        implementation=document_store,
        session=session,
        session_factory=session_factory,
    )
    assert isinstance(store, CassandraDocumentStore)
    return CassandraIntegrationBundle(
        config=config,
        document_store=store,
        cql_client=store.cql_client,
    )


def create_cassandra_document_store(
    *,
    document_store: Optional[DocumentStore] = None,
    session: Optional[object] = None,
    session_factory: Optional[Callable[[], object]] = None,
    **config_overrides: object,
) -> CassandraDocumentStore:
    """Catalog factory for ``IntegrationSlug.CASSANDRA`` / ``DOCUMENT_STORE``."""
    return create_cassandra_integration(
        document_store=document_store,
        session=session,
        session_factory=session_factory,
        **config_overrides,
    ).document_store

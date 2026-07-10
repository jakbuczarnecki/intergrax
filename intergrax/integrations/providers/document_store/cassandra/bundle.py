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
from intergrax.integrations.providers.document_store.cassandra.adapter import _CassandraDocumentStore
from intergrax.integrations.providers.document_store.cassandra.client import CassandraCqlClient
from intergrax.integrations.providers.document_store.cassandra.config import CassandraIntegrationConfig
from intergrax.integrations.providers.document_store.cassandra.opens import open_cassandra_document_store


@dataclass(frozen=True)
class CassandraIntegrationBundle:
    config: CassandraIntegrationConfig
    document_store: CassandraDocumentStoreIntegration
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
    integration = open_cassandra_document_store(
        config,
        implementation=document_store,
        session=session,
        session_factory=session_factory,
    )
    assert isinstance(integration, CassandraDocumentStoreIntegration)
    adapter = integration.as_document_store()
    from intergrax.integrations.providers.document_store.cassandra.adapter import _CassandraDocumentStore

    assert isinstance(adapter, _CassandraDocumentStore)
    return CassandraIntegrationBundle(
        config=config,
        document_store=integration,
        cql_client=adapter.cql_client,
    )


def create_cassandra_document_store(
    *,
    document_store: Optional[DocumentStore] = None,
    session: Optional[object] = None,
    session_factory: Optional[Callable[[], object]] = None,
    **config_overrides: object,
) -> DocumentStore:
    """Catalog factory for ``"cassandra"`` / ``DOCUMENT_STORE``."""
    return create_cassandra_integration(
        document_store=document_store,
        session=session,
        session_factory=session_factory,
        **config_overrides,
    ).document_store.as_document_store()

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.document_store.cassandra.integration import (
    CASSANDRA_DOCUMENT_STORE_PROVIDER_ID,
    CassandraDocumentStoreIntegration,
    CassandraDocumentStoreIntegrationConfig,
    CassandraDocumentStoreClient,
)


def create_cassandra_document_store_integration(
    *,
    client: CassandraDocumentStoreClient | None = None,
    enabled: bool = False,
) -> CassandraDocumentStoreIntegration:
    """
    Build a contract-based Cassandra document store integration.

    Compatibility shim — constructs Integration via from_store (create_cassandra_integration) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Cassandra document store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return CassandraDocumentStoreIntegration.from_client(client, enabled=enabled)
    return CassandraDocumentStoreIntegration.for_provider(
        provider_id=CASSANDRA_DOCUMENT_STORE_PROVIDER_ID,
        display_name="Cassandra",
        config=CassandraDocumentStoreIntegrationConfig(enabled=enabled),
    )

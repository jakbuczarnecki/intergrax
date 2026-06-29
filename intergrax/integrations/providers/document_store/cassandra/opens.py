# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Low-level Cassandra session openers — internal to the cassandra integration package.

Only this module may construct the Cassandra driver ``Cluster`` / ``Session``.
All composition roots use ``bundle.create_cassandra_*`` or
``profile.resolve(DOCUMENT_STORE)``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.integrations.providers.document_store.cassandra.adapter import _CassandraDocumentStore
from intergrax.integrations.providers.document_store.cassandra.integration import CassandraDocumentStoreIntegration
from intergrax.integrations.providers.document_store.cassandra.client import CassandraCqlClient
from intergrax.integrations.providers.document_store.cassandra.config import CassandraIntegrationConfig


def _import_cassandra() -> tuple[Any, Any]:
    try:
        from cassandra.auth import PlainTextAuthProvider
        from cassandra.cluster import Cluster
    except ImportError as exc:
        raise IntegrationConfigurationError(
            "Cassandra integration requires cassandra-driver. "
            "Install with: uv sync --extra dev  (includes cassandra-driver)"
        ) from exc
    return Cluster, PlainTextAuthProvider


def _open_session(
    config: CassandraIntegrationConfig,
    *,
    session_factory: Optional[Callable[[], Any]] = None,
) -> Any:
    if session_factory is not None:
        return session_factory()
    Cluster, PlainTextAuthProvider = _import_cassandra()
    auth_provider = None
    if config.user:
        auth_provider = PlainTextAuthProvider(username=config.user, password=config.password)
    cluster_kwargs: dict[str, Any] = {
        "contact_points": config.contact_points_list(),
        "port": config.port,
        "auth_provider": auth_provider,
    }
    if config.local_datacenter:
        from cassandra.policies import DCAwareRoundRobinPolicy

        cluster_kwargs["load_balancing_policy"] = DCAwareRoundRobinPolicy(
            local_dc=config.local_datacenter
        )
    cluster = Cluster(**cluster_kwargs)
    return cluster.connect(config.keyspace)


def open_cassandra_cql_client(
    config: CassandraIntegrationConfig,
    *,
    session: Optional[Any] = None,
    session_factory: Optional[Callable[[], Any]] = None,
) -> CassandraCqlClient:
    cql_session = session or _open_session(config, session_factory=session_factory)
    return CassandraCqlClient(config, session=cql_session)


def open_cassandra_document_store(
    config: CassandraIntegrationConfig,
    *,
    implementation: Optional[DocumentStore] = None,
    session: Optional[Any] = None,
    session_factory: Optional[Callable[[], Any]] = None,
) -> DocumentStore:
    if implementation is not None:
        return implementation
    client = open_cassandra_cql_client(
        config,
        session=session,
        session_factory=session_factory,
    )
    return CassandraDocumentStoreIntegration.from_client(_CassandraDocumentStore(client))
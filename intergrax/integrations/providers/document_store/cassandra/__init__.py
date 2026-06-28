# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Cassandra document store integration (Phase M.6 P2)."""

from intergrax.utils.lazy_export import export_from_bundle
from intergrax.integrations.providers.document_store.cassandra.config import (
    ENV_CASSANDRA_CONTACT_POINTS,
    ENV_CASSANDRA_KEYSPACE,
    ENV_CASSANDRA_PASSWORD,
    ENV_CASSANDRA_PORT,
    ENV_CASSANDRA_TABLE,
    ENV_CASSANDRA_USER,
    CassandraIntegrationConfig,
)

__all__ = [
    "ENV_CASSANDRA_CONTACT_POINTS",
    "ENV_CASSANDRA_KEYSPACE",
    "ENV_CASSANDRA_PASSWORD",
    "ENV_CASSANDRA_PORT",
    "ENV_CASSANDRA_TABLE",
    "ENV_CASSANDRA_USER",
    "CassandraDocumentStore",
    "CassandraIntegrationBundle",
    "CassandraIntegrationConfig",
    "create_cassandra_document_store",
    "create_cassandra_integration",
    "register_cassandra_integration",
    "resolve_cassandra_config",
    "create_cassandra_document_store_integration",
]

_LAZY_EXPORTS = frozenset(
    {
        "CassandraIntegrationBundle",
        "CassandraDocumentStore",
        "create_cassandra_integration",
        "create_cassandra_document_store",
        "register_cassandra_integration",
        "resolve_cassandra_config",
        "create_cassandra_document_store_integration",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "CASSANDRA_DOCUMENT_STORE_PROVIDER_ID",
        "CassandraDocumentStoreIntegration",
        "CassandraDocumentStoreIntegrationConfig",
        "CassandraDocumentStoreClient",
    }
)

def __getattr__(name: str):
    if name == "register_cassandra_integration":
        from intergrax.integrations.providers.document_store.cassandra.register import register_cassandra_integration

        return register_cassandra_integration
    if name in _LAZY_EXPORTS:
        from intergrax.integrations.providers.document_store.cassandra import bundle as _bundle

        return export_from_bundle(_bundle, name, _LAZY_EXPORTS)
    if name == "CassandraDocumentStore":
        from intergrax.integrations.providers.document_store.cassandra.adapter import CassandraDocumentStore

        return CassandraDocumentStore
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.document_store.cassandra import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Cassandra document store integration (Phase M.6 P2)."""

from intergrax.integrations.providers.cassandra.config import (
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
]

_LAZY_EXPORTS = frozenset(
    {
        "CassandraIntegrationBundle",
        "CassandraDocumentStore",
        "create_cassandra_integration",
        "create_cassandra_document_store",
        "register_cassandra_integration",
        "resolve_cassandra_config",
    }
)


def __getattr__(name: str):
    if name == "register_cassandra_integration":
        from intergrax.integrations.providers.cassandra.register import register_cassandra_integration

        return register_cassandra_integration
    if name in _LAZY_EXPORTS:
        from intergrax.integrations.providers.cassandra import bundle as _bundle

        return getattr(_bundle, name)
    if name == "CassandraDocumentStore":
        from intergrax.integrations.providers.cassandra.adapter import CassandraDocumentStore

        return CassandraDocumentStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

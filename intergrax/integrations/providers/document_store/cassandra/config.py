# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Cassandra document store integration configuration (Phase M.6 P2)."""

from __future__ import annotations

import os

from intergrax.integrations._shared.config import BaseIntegrationConfig
from intergrax.integrations.contracts.base import IntegrationConfigurationError

ENV_CASSANDRA_CONTACT_POINTS = "INTERGRAX_CASSANDRA_CONTACT_POINTS"
ENV_CASSANDRA_PORT = "INTERGRAX_CASSANDRA_PORT"
ENV_CASSANDRA_KEYSPACE = "INTERGRAX_CASSANDRA_KEYSPACE"
ENV_CASSANDRA_TABLE = "INTERGRAX_CASSANDRA_TABLE"
ENV_CASSANDRA_USER = "INTERGRAX_CASSANDRA_USER"
ENV_CASSANDRA_PASSWORD = "INTERGRAX_CASSANDRA_PASSWORD"
ENV_CASSANDRA_LOCAL_DC = "INTERGRAX_CASSANDRA_LOCAL_DC"

DEFAULT_PORT = 9042
DEFAULT_TABLE = "intergrax_documents"


def _validate_identifier(value: str, field: str) -> str:
    if not value or not all(ch.isalnum() or ch == "_" for ch in value):
        raise IntegrationConfigurationError(f"Invalid Cassandra {field}: {value!r}")
    return value


class CassandraIntegrationConfig(BaseIntegrationConfig):
    """
    Cassandra CQL settings.

    Expected table schema (default ``intergrax_documents``)::

        CREATE TABLE intergrax_documents (
            partition_key text,
            row_key text,
            payload text,
            PRIMARY KEY ((partition_key), row_key)
        );
    """

    contact_points: str = ""
    port: int = DEFAULT_PORT
    keyspace: str = ""
    table: str = DEFAULT_TABLE
    user: str = ""
    password: str = ""
    local_datacenter: str = ""

    def contact_points_list(self) -> list[str]:
        return [part.strip() for part in self.contact_points.split(",") if part.strip()]

    def qualified_table(self) -> str:
        keyspace = _validate_identifier(self.keyspace, "keyspace")
        table = _validate_identifier(self.table, "table")
        return f"{keyspace}.{table}"

    @classmethod
    def from_env(cls, **overrides: object) -> CassandraIntegrationConfig:
        contact_points = os.environ.get(ENV_CASSANDRA_CONTACT_POINTS, "").strip()
        port_raw = os.environ.get(ENV_CASSANDRA_PORT, "").strip()
        keyspace = os.environ.get(ENV_CASSANDRA_KEYSPACE, "").strip()
        table = os.environ.get(ENV_CASSANDRA_TABLE, DEFAULT_TABLE).strip() or DEFAULT_TABLE
        user = os.environ.get(ENV_CASSANDRA_USER, "").strip()
        password = os.environ.get(ENV_CASSANDRA_PASSWORD, "").strip()
        local_datacenter = os.environ.get(ENV_CASSANDRA_LOCAL_DC, "").strip()
        payload: dict[str, object] = {
            "contact_points": contact_points,
            "keyspace": keyspace,
            "table": table,
            "user": user,
            "password": password,
            "local_datacenter": local_datacenter,
        }
        if port_raw:
            payload["port"] = int(port_raw)
        else:
            payload["port"] = DEFAULT_PORT
        payload.update(overrides)
        return cls.model_validate(payload)

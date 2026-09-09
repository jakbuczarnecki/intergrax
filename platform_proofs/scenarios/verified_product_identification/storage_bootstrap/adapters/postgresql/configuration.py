"""Typed PostgreSQL configuration for VPI relational storage bootstrap."""

from __future__ import annotations

import re
from dataclasses import dataclass

from intergrax.integrations.providers.relational_store.postgresql.config import (
    PostgreSQLIntegrationConfig,
    validate_schema_identifier,
)

_TABLE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TABLE_VALIDATION_MESSAGE = (
    "table_name must be a simple SQL identifier (letters, digits, underscore)"
)

DEFAULT_APPLICATION_NAME = "vpi-relational-bootstrap"
DEFAULT_RELATIONAL_TABLE_NAME = "vpi_data_pack_relational_record"


def validate_table_identifier(value: str) -> str:
    stripped = value.strip()
    if not _TABLE_PATTERN.match(stripped):
        raise ValueError(_TABLE_VALIDATION_MESSAGE)
    return stripped


@dataclass(frozen=True, slots=True)
class PostgreSqlBootstrapConfiguration:
    integration: PostgreSQLIntegrationConfig
    schema_name: str
    table_name: str
    statement_timeout_ms: int | None = None
    connection_timeout_seconds: int | None = None
    application_name: str = DEFAULT_APPLICATION_NAME

    @classmethod
    def from_env(
        cls,
        *,
        schema_name: str,
        table_name: str = DEFAULT_RELATIONAL_TABLE_NAME,
        statement_timeout_ms: int | None = None,
        connection_timeout_seconds: int | None = None,
        application_name: str = DEFAULT_APPLICATION_NAME,
    ) -> PostgreSqlBootstrapConfiguration:
        if not application_name.strip():
            raise ValueError("application_name must be non-empty")
        integration = PostgreSQLIntegrationConfig.from_env(tenant_schema=schema_name)
        return cls(
            integration=integration,
            schema_name=validate_schema_identifier(schema_name),
            table_name=validate_table_identifier(table_name),
            statement_timeout_ms=statement_timeout_ms,
            connection_timeout_seconds=connection_timeout_seconds,
            application_name=application_name,
        )

    def __repr__(self) -> str:
        return (
            "PostgreSqlBootstrapConfiguration("
            f"schema_name={self.schema_name!r}, "
            f"table_name={self.table_name!r}, "
            f"statement_timeout_ms={self.statement_timeout_ms!r}, "
            f"connection_timeout_seconds={self.connection_timeout_seconds!r}, "
            f"application_name={self.application_name!r})"
        )

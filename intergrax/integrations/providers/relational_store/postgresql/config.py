# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""PostgreSQL integration configuration (Phase M.6)."""

from __future__ import annotations

import os
import re
from typing import Optional
from urllib.parse import quote_plus

from pydantic import field_validator

from intergrax.integrations._shared.config import BaseIntegrationConfig

ENV_POSTGRESQL_DSN = "INTERGRAX_POSTGRESQL_DSN"
ENV_POSTGRESQL_HOST = "INTERGRAX_POSTGRESQL_HOST"
ENV_POSTGRESQL_PORT = "INTERGRAX_POSTGRESQL_PORT"
ENV_POSTGRESQL_USER = "INTERGRAX_POSTGRESQL_USER"
ENV_POSTGRESQL_PASSWORD = "INTERGRAX_POSTGRESQL_PASSWORD"
ENV_POSTGRESQL_DATABASE = "INTERGRAX_POSTGRESQL_DATABASE"
ENV_POSTGRESQL_SSLMODE = "INTERGRAX_POSTGRESQL_SSLMODE"
ENV_POSTGRESQL_TENANT_SCHEMA = "INTERGRAX_POSTGRESQL_SCHEMA"

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 5432
DEFAULT_USER = "intergrax"
DEFAULT_DATABASE = "intergrax"
DEFAULT_SSLMODE = "prefer"

_SCHEMA_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SCHEMA_VALIDATION_MESSAGE = (
    "tenant_schema must be a simple SQL identifier (letters, digits, underscore)"
)


def validate_schema_identifier(value: str) -> str:
    """Canonical SQL schema identifier validation shared by config and session provider."""
    stripped = value.strip()
    if stripped == "public":
        return stripped
    if not _SCHEMA_PATTERN.match(stripped):
        raise ValueError(_SCHEMA_VALIDATION_MESSAGE)
    return stripped


class PostgreSQLIntegrationConfig(BaseIntegrationConfig):
    """
    Connection settings for production ``RelationalStore``.

    Prefer ``INTERGRAX_POSTGRESQL_DSN``; otherwise host/port/user/password/database are composed.
    """

    dsn: Optional[str] = None
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    user: str = DEFAULT_USER
    password: str = ""
    database: str = DEFAULT_DATABASE
    sslmode: str = DEFAULT_SSLMODE
    tenant_schema: Optional[str] = None

    @field_validator("tenant_schema")
    @classmethod
    def _validate_tenant_schema(cls, value: Optional[str]) -> Optional[str]:
        if value is None or value == "":
            return None
        return validate_schema_identifier(value)

    def connection_string(self) -> str:
        if self.dsn:
            return self.dsn.strip()
        user = quote_plus(self.user)
        password = quote_plus(self.password)
        auth = user if not password else f"{user}:{password}"
        return (
            f"postgresql://{auth}@{self.host}:{self.port}/{self.database}"
            f"?sslmode={quote_plus(self.sslmode)}"
        )

    @classmethod
    def from_env(cls, **overrides: object) -> PostgreSQLIntegrationConfig:
        payload: dict[str, object] = {}
        dsn = os.environ.get(ENV_POSTGRESQL_DSN, "").strip()
        if dsn:
            payload["dsn"] = dsn
        host = os.environ.get(ENV_POSTGRESQL_HOST, "").strip()
        if host:
            payload["host"] = host
        port_raw = os.environ.get(ENV_POSTGRESQL_PORT, "").strip()
        if port_raw:
            payload["port"] = int(port_raw)
        user = os.environ.get(ENV_POSTGRESQL_USER, "").strip()
        if user:
            payload["user"] = user
        if ENV_POSTGRESQL_PASSWORD in os.environ:
            payload["password"] = os.environ.get(ENV_POSTGRESQL_PASSWORD, "")
        database = os.environ.get(ENV_POSTGRESQL_DATABASE, "").strip()
        if database:
            payload["database"] = database
        sslmode = os.environ.get(ENV_POSTGRESQL_SSLMODE, "").strip()
        if sslmode:
            payload["sslmode"] = sslmode
        schema = os.environ.get(ENV_POSTGRESQL_TENANT_SCHEMA, "").strip()
        if schema:
            payload["tenant_schema"] = schema
        payload.update(overrides)
        return cls.model_validate(payload)

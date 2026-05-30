# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""MySQL integration configuration (Phase M.6)."""

from __future__ import annotations

import os
import re
from typing import Any, Optional
from urllib.parse import quote_plus, unquote, urlparse

from pydantic import field_validator

from intergrax.integrations._shared.config import BaseIntegrationConfig

ENV_MYSQL_DSN = "INTERGRAX_MYSQL_DSN"
ENV_MYSQL_HOST = "INTERGRAX_MYSQL_HOST"
ENV_MYSQL_PORT = "INTERGRAX_MYSQL_PORT"
ENV_MYSQL_USER = "INTERGRAX_MYSQL_USER"
ENV_MYSQL_PASSWORD = "INTERGRAX_MYSQL_PASSWORD"
ENV_MYSQL_DATABASE = "INTERGRAX_MYSQL_DATABASE"
ENV_MYSQL_CHARSET = "INTERGRAX_MYSQL_CHARSET"
ENV_MYSQL_TENANT_DATABASE = "INTERGRAX_MYSQL_TENANT_DATABASE"

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 3306
DEFAULT_USER = "intergrax"
DEFAULT_DATABASE = "intergrax"
DEFAULT_CHARSET = "utf8mb4"

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class MySQLIntegrationConfig(BaseIntegrationConfig):
    """
    Connection settings for production ``RelationalStore``.

    Prefer ``INTERGRAX_MYSQL_DSN`` (`mysql://user:pass@host:3306/db`); otherwise compose from parts.
    """

    dsn: Optional[str] = None
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    user: str = DEFAULT_USER
    password: str = ""
    database: str = DEFAULT_DATABASE
    charset: str = DEFAULT_CHARSET
    tenant_database: Optional[str] = None

    @field_validator("tenant_database")
    @classmethod
    def _validate_tenant_database(cls, value: Optional[str]) -> Optional[str]:
        if value is None or value == "":
            return None
        if not _IDENTIFIER_PATTERN.match(value):
            raise ValueError(
                "tenant_database must be a simple SQL identifier (letters, digits, underscore)"
            )
        return value

    def connection_string(self) -> str:
        if self.dsn:
            return self.dsn.strip()
        user = quote_plus(self.user)
        password = quote_plus(self.password)
        auth = user if not password else f"{user}:{password}"
        return f"mysql://{auth}@{self.host}:{self.port}/{self.database}?charset={quote_plus(self.charset)}"

    def connection_kwargs(self) -> dict[str, Any]:
        if self.dsn:
            return _parse_mysql_dsn(self.dsn, default_charset=self.charset)
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password,
            "database": self.database,
            "charset": self.charset,
        }

    @classmethod
    def from_env(cls, **overrides: object) -> MySQLIntegrationConfig:
        payload: dict[str, object] = {}
        dsn = os.environ.get(ENV_MYSQL_DSN, "").strip()
        if dsn:
            payload["dsn"] = dsn
        host = os.environ.get(ENV_MYSQL_HOST, "").strip()
        if host:
            payload["host"] = host
        port_raw = os.environ.get(ENV_MYSQL_PORT, "").strip()
        if port_raw:
            payload["port"] = int(port_raw)
        user = os.environ.get(ENV_MYSQL_USER, "").strip()
        if user:
            payload["user"] = user
        if ENV_MYSQL_PASSWORD in os.environ:
            payload["password"] = os.environ.get(ENV_MYSQL_PASSWORD, "")
        database = os.environ.get(ENV_MYSQL_DATABASE, "").strip()
        if database:
            payload["database"] = database
        charset = os.environ.get(ENV_MYSQL_CHARSET, "").strip()
        if charset:
            payload["charset"] = charset
        tenant_database = os.environ.get(ENV_MYSQL_TENANT_DATABASE, "").strip()
        if tenant_database:
            payload["tenant_database"] = tenant_database
        payload.update(overrides)
        return cls.model_validate(payload)


def _parse_mysql_dsn(dsn: str, *, default_charset: str) -> dict[str, Any]:
    parsed = urlparse(dsn.strip())
    if parsed.scheme not in {"mysql", "mysql+pymysql"}:
        raise ValueError(f"Unsupported MySQL DSN scheme: {parsed.scheme!r}")

    database = parsed.path.lstrip("/") or DEFAULT_DATABASE
    query = parsed.query or ""
    charset = default_charset
    if "charset=" in query:
        for part in query.split("&"):
            if part.startswith("charset="):
                charset = unquote(part.split("=", 1)[1]) or default_charset

    port = parsed.port or DEFAULT_PORT
    return {
        "host": parsed.hostname or DEFAULT_HOST,
        "port": port,
        "user": unquote(parsed.username or DEFAULT_USER),
        "password": unquote(parsed.password or ""),
        "database": unquote(database),
        "charset": charset,
    }

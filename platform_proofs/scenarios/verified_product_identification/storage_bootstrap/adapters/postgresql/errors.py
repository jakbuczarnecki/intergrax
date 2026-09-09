"""PostgreSQL relational bootstrap adapter errors — translated at port boundary."""

from __future__ import annotations


class PostgreSqlBootstrapAdapterError(Exception):
    """Base adapter error for PostgreSQL relational storage bootstrap."""


class PostgreSqlBootstrapConfigurationError(PostgreSqlBootstrapAdapterError):
    """Invalid adapter configuration or unmapped logical relational target."""


class PostgreSqlBootstrapSchemaError(PostgreSqlBootstrapAdapterError):
    """Schema preparation or compatibility failure."""


class PostgreSqlBootstrapIdentityConflictError(PostgreSqlBootstrapAdapterError):
    """Canonical identity already exists with incompatible immutable content."""


class PostgreSqlBootstrapOperationError(PostgreSqlBootstrapAdapterError):
    """Bounded PostgreSQL operation failure without leaking driver details."""
